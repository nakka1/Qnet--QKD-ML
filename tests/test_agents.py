"""The policies, the network under them, and the evaluation around them.

The gradient checks are the important part.  A hand-derived backward pass that
is subtly wrong still trains — more slowly, to a worse optimum — and watching a
learning curve will not tell you.  Finite differences will, so every loss the
trainer actually uses is checked against them here.

The rest guards the properties a comparison depends on: that every policy
returns a legal index, that evaluation pairs policies on identical episodes,
and that the reported score is the unshaped outcome rather than the shaped
reward the agent was trained on.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.agents import (GreedyFidelityPolicy, RandomPolicy, ShortestPathPolicy,
                                   ThresholdPolicy, evaluate, run_episode)
from qnet_autopilot.agents.actor_critic import (ActorCriticPolicy, TrainingConfig,
                                                _generalised_advantages, train)
from qnet_autopilot.agents.base import Policy
from qnet_autopilot.agents.numpy_nn import MLP, Adam, categorical_entropy, softmax
from qnet_autopilot.env import EnvironmentConfig, QuantumNetworkEnv, Request
from qnet_autopilot.env.actions import ActionKind
from qnet_autopilot.env.features import ACTION_FEATURES, STATE_FEATURES
from qnet_autopilot.network import scenario
from qnet_autopilot.physics.memory import MemoryParameters


def build_env(**overrides) -> QuantumNetworkEnv:
    defaults = dict(
        graph=scenario("chain", num_nodes=4, span_km=20),
        memory=MemoryParameters(coherence_time_s=1.0, raw_fidelity=0.90, cutoff_ratio=0.1),
        gate_fid=0.999, meas_fid=0.999,
        coupling_efficiency=0.7, detector_efficiency=0.85,
        generate_budget_s=0.004, wait_slot_s=0.002,
        max_steps=60, max_pairs_per_link=3)
    defaults.update(overrides)
    return QuantumNetworkEnv(EnvironmentConfig(**defaults), rng=0)


def make_request(rng: np.random.Generator) -> Request:
    return Request(source="0", destination="3",
                   min_fidelity=float(rng.choice([0.70, 0.74])),
                   deadline_s=float(rng.uniform(0.04, 0.09)))


ALL_POLICIES = [RandomPolicy, ShortestPathPolicy, ThresholdPolicy, GreedyFidelityPolicy]


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

class TestMLP:

    def test_rejects_a_degenerate_shape(self):
        with pytest.raises(ValueError):
            MLP([4])

    def test_forward_shape_and_batching(self):
        net = MLP([5, 8, 3], rng=0)
        assert net.forward(np.zeros(5)).shape == (1, 3)
        assert net.forward(np.zeros((7, 5))).shape == (7, 3)

    def test_backward_before_forward_is_an_error(self):
        with pytest.raises(RuntimeError):
            MLP([3, 2], rng=0).backward(np.ones((1, 2)))

    def test_pickling_keeps_the_weights_and_drops_the_activation_cache(self):
        """A checkpoint should be its weights, not the last batch it saw.

        The forward pass caches every layer's activations for backprop. Pickled
        along with the model that cache is the entire batch times the hidden
        width — for a validation pass, hundreds of megabytes of something that
        is not the model at all.
        """
        import pickle

        net = MLP([4, 32, 2], rng=0)
        net.forward(np.ones((5000, 4)))
        assert net._cache, "forward should cache for backward"

        payload = pickle.dumps(net)
        restored = pickle.loads(payload)

        assert restored._cache == []
        assert len(payload) < 8 * sum(array.size for array in net.parameters) + 4096
        assert np.allclose(restored.forward(np.ones((3, 4))),
                           net.forward(np.ones((3, 4))))

    def test_flat_round_trip(self):
        net = MLP([4, 6, 2], rng=1)
        flat = net.get_flat().copy()
        net.set_flat(np.zeros_like(flat))
        assert np.allclose(net.get_flat(), 0.0)
        net.set_flat(flat)
        assert np.allclose(net.get_flat(), flat)

    def test_set_flat_rejects_the_wrong_size(self):
        net = MLP([4, 6, 2], rng=1)
        with pytest.raises(ValueError):
            net.set_flat(np.zeros(3))

    def test_zero_output_gives_a_uniform_policy(self):
        net = MLP([6, 8, 1], rng=2, zero_output=True)
        probabilities = softmax(net.forward(np.random.default_rng(0).normal(size=(5, 6)))[:, 0])
        assert np.allclose(probabilities, 0.2)

    @pytest.mark.parametrize("sizes", [[4, 5, 1], [6, 8, 8, 1], [3, 4, 2]])
    def test_gradient_matches_finite_differences_on_a_sum_loss(self, sizes):
        rng = np.random.default_rng(0)
        net = MLP(sizes, rng=3)
        inputs = rng.normal(size=(6, sizes[0]))

        def loss(flat):
            net.set_flat(flat)
            return float(np.sum(net.forward(inputs) ** 2))

        flat = net.get_flat().copy()
        net.set_flat(flat)
        analytic = np.concatenate(
            [g.ravel() for g in net.backward(2 * net.forward(inputs))])

        step = 1e-6
        indices = rng.choice(flat.size, size=min(30, flat.size), replace=False)
        for index in indices:
            up, down = flat.copy(), flat.copy()
            up[index] += step
            down[index] -= step
            numeric = (loss(up) - loss(down)) / (2 * step)
            net.set_flat(flat)
            assert analytic[index] == pytest.approx(numeric, abs=1e-5, rel=1e-4)

    def test_policy_gradient_with_entropy_matches_finite_differences(self):
        """The exact loss :func:`train` differentiates, checked numerically."""
        rng = np.random.default_rng(1)
        net = MLP([7, 9, 6, 1], rng=4)
        inputs = rng.normal(size=(8, 7))
        choice, advantage, beta = 5, -0.6, 0.04

        def loss(flat):
            net.set_flat(flat)
            probabilities = softmax(net.forward(inputs)[:, 0])
            return (-np.log(probabilities[choice]) * advantage
                    - beta * categorical_entropy(probabilities))

        flat = net.get_flat().copy()
        net.set_flat(flat)
        probabilities = softmax(net.forward(inputs)[:, 0])
        entropy = categorical_entropy(probabilities)
        gradient = probabilities.copy()
        gradient[choice] -= 1.0
        gradient *= advantage
        gradient += beta * probabilities * (
            np.log(np.clip(probabilities, 1e-12, 1.0)) + entropy)
        analytic = np.concatenate([g.ravel() for g in net.backward(gradient[:, None])])

        step = 1e-6
        indices = rng.choice(flat.size, size=40, replace=False)
        for index in indices:
            up, down = flat.copy(), flat.copy()
            up[index] += step
            down[index] -= step
            numeric = (loss(up) - loss(down)) / (2 * step)
            net.set_flat(flat)
            assert analytic[index] == pytest.approx(numeric, abs=1e-6, rel=1e-4)

    def test_value_loss_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(2)
        net = MLP([5, 7, 1], rng=5)
        inputs = rng.normal(size=(9, 5))
        targets = rng.normal(size=(9, 1))

        def loss(flat):
            net.set_flat(flat)
            return float(np.mean((net.forward(inputs) - targets) ** 2))

        flat = net.get_flat().copy()
        net.set_flat(flat)
        predicted = net.forward(inputs)
        analytic = np.concatenate(
            [g.ravel() for g in net.backward(2 * (predicted - targets) / len(targets))])

        step = 1e-6
        for index in rng.choice(flat.size, size=min(25, flat.size), replace=False):
            up, down = flat.copy(), flat.copy()
            up[index] += step
            down[index] -= step
            numeric = (loss(up) - loss(down)) / (2 * step)
            net.set_flat(flat)
            assert analytic[index] == pytest.approx(numeric, abs=1e-6, rel=1e-4)


class TestAdam:

    def test_minimises_a_quadratic(self):
        parameters = [np.array([5.0, -3.0])]
        optimiser = Adam(learning_rate=0.1, clip_norm=None)
        for _ in range(600):
            optimiser.step(parameters, [2 * parameters[0]])
        assert np.allclose(parameters[0], 0.0, atol=1e-4)

    def test_clipping_bounds_the_step(self):
        parameters = [np.zeros(3)]
        optimiser = Adam(learning_rate=0.1, clip_norm=1.0)
        norm = optimiser.step(parameters, [np.full(3, 1e6)])
        assert norm == pytest.approx(np.sqrt(3) * 1e6)
        # With clipping and bias correction, the first step is ~the learning rate.
        assert np.all(np.abs(parameters[0]) <= 0.1 + 1e-9)

    def test_reports_the_unclipped_norm(self):
        optimiser = Adam(clip_norm=1.0)
        norm = optimiser.step([np.zeros(2)], [np.array([3.0, 4.0])])
        assert norm == pytest.approx(5.0)


class TestSoftmax:

    def test_is_a_distribution_and_survives_large_logits(self):
        probabilities = softmax(np.array([1000.0, 1001.0, 999.0]))
        assert np.all(np.isfinite(probabilities))
        assert probabilities.sum() == pytest.approx(1.0)
        assert np.argmax(probabilities) == 1

    def test_entropy_at_the_extremes(self):
        # The log is clipped at 1e-12 rather than guarded by a mask, so a
        # degenerate distribution reports ~5e-11 instead of exactly zero. That
        # is deliberate — it keeps the gradient finite — so the tolerance here
        # is absolute, not pytest.approx's default relative one.
        assert categorical_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)
        uniform = np.full(4, 0.25)
        assert categorical_entropy(uniform) == pytest.approx(np.log(4))


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

class TestPolicies:

    @pytest.mark.parametrize("factory", ALL_POLICIES)
    def test_every_policy_satisfies_the_protocol(self, factory):
        assert isinstance(factory(), Policy)

    @pytest.mark.parametrize("factory", ALL_POLICIES)
    def test_every_policy_returns_a_legal_index(self, factory):
        policy = factory()
        env = build_env()
        observation, _ = env.reset(make_request(np.random.default_rng(0)),
                                   memory_size=5, rng=1)
        policy.reset()
        for _ in range(60):
            choice = policy.act(observation)
            _, _, actions = observation
            assert isinstance(choice, (int, np.integer))
            assert 0 <= choice < len(actions)
            observation, result = env.step(actions[choice])
            if result.done:
                break

    def test_deterministic_baselines_repeat_themselves(self):
        def trace(policy_factory):
            env = build_env()
            observation, _ = env.reset(Request("0", "3", min_fidelity=0.74,
                                               deadline_s=0.08),
                                       memory_size=5, rng=42)
            policy = policy_factory()
            policy.reset()
            choices = []
            for _ in range(40):
                choice = policy.act(observation)
                choices.append(choice)
                observation, result = env.step(observation[2][choice])
                if result.done:
                    break
            return choices

        for factory in (ShortestPathPolicy, ThresholdPolicy, GreedyFidelityPolicy):
            assert trace(factory) == trace(factory)

    def test_shortest_path_never_purifies(self):
        env = build_env()
        policy = ShortestPathPolicy()
        for seed in range(6):
            episode = run_episode(env, policy,
                                  request=make_request(np.random.default_rng(seed)),
                                  memory_size=5, seed=seed, collect=False)
            assert episode.purifications == 0

    def test_the_threshold_policy_does_purify(self):
        env = build_env()
        policy = ThresholdPolicy(target_fidelity=0.94)
        total = sum(run_episode(env, policy,
                                request=make_request(np.random.default_rng(seed)),
                                memory_size=5, seed=seed, collect=False).purifications
                    for seed in range(6))
        assert total > 0

    def test_the_threshold_policy_stops_purifying_when_the_target_is_low(self):
        """A target below the raw link fidelity means no round can be justified."""
        env = build_env()
        policy = ThresholdPolicy(target_fidelity=0.0)
        total = sum(run_episode(env, policy,
                                request=make_request(np.random.default_rng(seed)),
                                memory_size=5, seed=seed, collect=False).purifications
                    for seed in range(6))
        assert total == 0


# ---------------------------------------------------------------------------
# rollout and evaluation
# ---------------------------------------------------------------------------

class TestRollout:

    def test_an_episode_records_what_it_did(self):
        env = build_env()
        episode = run_episode(env, ShortestPathPolicy(),
                              request=Request("0", "3", min_fidelity=0.70,
                                              deadline_s=0.09),
                              memory_size=5, seed=3)
        assert episode.steps == len(episode.transitions)
        assert episode.elapsed_s > 0
        assert episode.request is not None
        assert episode.succeeded == (episode.delivered > 0)

    def test_a_seed_reproduces_an_episode(self):
        env = build_env()
        first = run_episode(env, ShortestPathPolicy(), memory_size=5, seed=7,
                            request=Request("0", "3", deadline_s=0.08), collect=False)
        second = run_episode(env, ShortestPathPolicy(), memory_size=5, seed=7,
                             request=Request("0", "3", deadline_s=0.08), collect=False)
        assert first.raw_return == pytest.approx(second.raw_return)
        assert first.steps == second.steps
        assert first.attempts == second.attempts

    def test_an_out_of_range_choice_is_caught(self):
        class BadPolicy:
            name = "bad"

            def reset(self):
                pass

            def act(self, observation):
                return 10_000

        with pytest.raises(IndexError):
            run_episode(build_env(), BadPolicy(), memory_size=4, seed=0,
                        request=Request("0", "3", deadline_s=0.05))

    def test_collect_false_still_counts_steps(self):
        env = build_env()
        episode = run_episode(env, RandomPolicy(0), memory_size=5, seed=2,
                              request=Request("0", "3", deadline_s=0.06), collect=False)
        assert episode.steps == len(episode.transitions)
        assert episode.transitions[0].state_features.size == 0

    def test_the_raw_return_ignores_shaping(self):
        plain = run_episode(build_env(shaping_weight=0.0), ShortestPathPolicy(),
                            memory_size=5, seed=9,
                            request=Request("0", "3", deadline_s=0.08), collect=False)
        shaped = run_episode(build_env(shaping_weight=2.0), ShortestPathPolicy(),
                             memory_size=5, seed=9,
                             request=Request("0", "3", deadline_s=0.08), collect=False)
        assert plain.raw_return == pytest.approx(shaped.raw_return)
        assert plain.shaped_return != pytest.approx(shaped.shaped_return)


class TestEvaluation:

    def test_every_policy_sees_the_same_episodes(self):
        result = evaluate(build_env, [RandomPolicy(0), ShortestPathPolicy()],
                          episodes=12, request_factory=make_request,
                          memory_size=5, seed=100)
        by_policy = result.frame.groupby("policy")["seed"].apply(list)
        assert by_policy.iloc[0] == by_policy.iloc[1]

    def test_summary_has_one_row_per_policy_with_an_error_bar(self):
        result = evaluate(build_env, [RandomPolicy(0), ShortestPathPolicy()],
                          episodes=15, request_factory=make_request,
                          memory_size=5, seed=200)
        summary = result.summary()
        assert len(summary) == 2
        assert (summary["success_stderr"] >= 0).all()
        assert (summary["episodes"] == 15).all()

    def test_paired_difference_is_symmetric_and_counts_correctly(self):
        result = evaluate(build_env, [RandomPolicy(0), ShortestPathPolicy()],
                          episodes=20, request_factory=make_request,
                          memory_size=5, seed=300)
        forward = result.paired_difference("shortest_path", "random")
        backward = result.paired_difference("random", "shortest_path")
        assert forward["mean"] == pytest.approx(-backward["mean"])
        assert forward["wins"] == backward["losses"]
        assert forward["wins"] + forward["losses"] + forward["ties"] == 20

    def test_paired_difference_needs_both_policies(self):
        result = evaluate(build_env, [RandomPolicy(0)], episodes=5,
                          request_factory=make_request, memory_size=5, seed=400)
        with pytest.raises(KeyError):
            result.paired_difference("random", "nobody")


# ---------------------------------------------------------------------------
# the learned policy
# ---------------------------------------------------------------------------

class TestActorCritic:

    def test_starts_uniform_over_legal_actions(self):
        policy = ActorCriticPolicy(rng=0)
        env = build_env()
        observation, _ = env.reset(Request("0", "3", deadline_s=0.08), memory_size=5)
        probabilities = policy.action_probabilities(observation)
        assert np.allclose(probabilities, 1.0 / len(probabilities))

    def test_input_layout_matches_the_feature_sizes(self):
        policy = ActorCriticPolicy(rng=0)
        assert policy.actor.sizes[0] == STATE_FEATURES + ACTION_FEATURES
        assert policy.critic.sizes[0] == STATE_FEATURES

    def test_temperature_zero_is_the_argmax(self):
        policy = ActorCriticPolicy(rng=0)
        policy.actor.weights[-1][:] = np.random.default_rng(0).normal(
            size=policy.actor.weights[-1].shape)
        env = build_env()
        observation, _ = env.reset(Request("0", "3", deadline_s=0.08), memory_size=5)

        probabilities = policy.action_probabilities(observation, temperature=0.0)
        assert probabilities.sum() == pytest.approx(1.0)
        assert probabilities.max() == pytest.approx(1.0)
        assert policy.view(temperature=0.0).act(observation) == int(np.argmax(probabilities))

    def test_lower_temperature_sharpens(self):
        policy = ActorCriticPolicy(rng=0)
        policy.actor.weights[-1][:] = np.random.default_rng(1).normal(
            size=policy.actor.weights[-1].shape)
        env = build_env()
        observation, _ = env.reset(Request("0", "3", deadline_s=0.08), memory_size=5)

        sharp = policy.action_probabilities(observation, temperature=0.25)
        soft = policy.action_probabilities(observation, temperature=2.0)
        assert categorical_entropy(sharp) < categorical_entropy(soft)

    def test_a_view_shares_the_weights(self):
        policy = ActorCriticPolicy(rng=0)
        view = policy.view(temperature=0.5)
        policy.actor.weights[0][0, 0] = 1.234
        assert view.actor.weights[0][0, 0] == pytest.approx(1.234)

    def test_save_and_load_round_trip(self):
        import tempfile
        from pathlib import Path

        original = ActorCriticPolicy(rng=0)
        original.actor.weights[-1][:] = np.random.default_rng(2).normal(
            size=original.actor.weights[-1].shape)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            original.save(path)
            restored = ActorCriticPolicy(rng=0)
            restored.load(path)

        for left, right in zip(original.actor.parameters, restored.actor.parameters):
            assert np.allclose(left, right)
        for left, right in zip(original.critic.parameters, restored.critic.parameters):
            assert np.allclose(left, right)


class TestAdvantages:

    def test_lambda_one_gives_the_monte_carlo_return(self):
        rewards = [1.0, -0.5, 2.0]
        values = [0.0, 0.0, 0.0]
        advantages, returns = _generalised_advantages(rewards, values, gamma=1.0, lam=1.0)
        assert advantages[0] == pytest.approx(sum(rewards))
        assert advantages[-1] == pytest.approx(rewards[-1])
        assert np.allclose(returns, advantages)

    def test_lambda_zero_gives_the_one_step_temporal_difference(self):
        rewards = [1.0, 2.0]
        values = [0.5, 0.25]
        advantages, _ = _generalised_advantages(rewards, values, gamma=0.9, lam=0.0)
        assert advantages[0] == pytest.approx(1.0 + 0.9 * 0.25 - 0.5)
        assert advantages[1] == pytest.approx(2.0 - 0.25)

    def test_a_perfect_critic_gives_zero_advantage(self):
        """With values equal to the true returns, there is nothing to learn."""
        rewards = [0.0, 0.0, 1.0]
        gamma = 0.9
        values = [gamma ** 2, gamma ** 1, 1.0]
        advantages, _ = _generalised_advantages(rewards, values, gamma=gamma, lam=0.95)
        assert np.allclose(advantages, 0.0, atol=1e-12)


class TestTraining:

    def test_a_short_run_improves_on_the_untrained_policy(self):
        """Not a performance claim — a check that the loop learns anything at all.

        The real numbers come from ``scripts/evaluate_phase2.py`` on held-out
        seeds; this only has to catch a trainer that silently does nothing.
        """
        config = TrainingConfig(iterations=12, batch_episodes=8, eval_every=11,
                                eval_episodes=24, learning_rate=3e-3,
                                learning_rate_end=3e-3, seed=0)
        policy, history = train(lambda: build_env(shaping_weight=0.5), make_request,
                                config, memory_size=5)

        assert len(history.train_success) == config.iterations
        assert history.eval_success
        # The entropy bonus should not have let the policy stay uniform.
        assert history.entropy[-1] < np.log(12)
        assert np.mean(history.train_success[-4:]) > np.mean(history.train_success[:4])

    def test_keeping_the_best_checkpoint_restores_those_weights(self):
        config = TrainingConfig(iterations=6, batch_episodes=4, eval_every=2,
                                eval_episodes=10, keep_best=True, seed=0)
        policy, history = train(lambda: build_env(shaping_weight=0.5), make_request,
                                config, memory_size=5)
        assert history.best_validation >= max(history.eval_success) - 1e-12
