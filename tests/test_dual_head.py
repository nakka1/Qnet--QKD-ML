"""The refinements: input gradients, feature normalisation, and two heads.

The gradient tests are the ones that matter. A shared trunk is the first thing
in this package where a gradient flows through two paths and has to be summed,
and a wrong sum does not crash, does not warn, and does not show up in a
learning curve — it just trains to somewhere worse. Both paths are therefore
checked against finite differences, including the part that is easiest to get
wrong: the actor's embedding is broadcast across the legal actions, so the
gradient arriving back at it is a sum down those rows.

The second group checks that the two architectures are actually interchangeable.
The whole experiment rests on both being driven by the *same* training loop, so
if one of them quietly fails to implement the protocol the comparison stops
being about architecture.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.agents.actor_critic import ActorCriticPolicy, TrainingConfig, train
from qnet_autopilot.agents.dual_head import DualHeadPolicy
from qnet_autopilot.agents.normalise import FeatureNormaliser
from qnet_autopilot.agents.numpy_nn import MLP
from qnet_autopilot.env.features import ACTION_FEATURES, STATE_FEATURES


def finite_difference(function, arrays, analytic, epsilon: float = 1e-6) -> float:
    """Largest disagreement between an analytic gradient and a numeric one."""
    worst = 0.0
    for array, gradient in zip(arrays, analytic):
        iterator = np.nditer(array, flags=["multi_index"])
        while not iterator.finished:
            index = iterator.multi_index
            original = array[index]
            array[index] = original + epsilon
            up = function()
            array[index] = original - epsilon
            down = function()
            array[index] = original
            worst = max(worst, abs((up - down) / (2 * epsilon) - gradient[index]))
            iterator.iternext()
    return worst


class TestInputGradient:

    def test_it_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        net = MLP([5, 7, 4, 3], rng=1)
        inputs = rng.normal(size=(6, 5))
        weights = rng.normal(size=(6, 3))

        net.forward(inputs)
        _, input_gradient = net.backward_to_input(weights)

        numeric = np.zeros_like(inputs)
        epsilon = 1e-6
        for row in range(inputs.shape[0]):
            for column in range(inputs.shape[1]):
                perturbed = inputs.copy()
                perturbed[row, column] += epsilon
                up = float(np.sum(net.forward(perturbed) * weights))
                perturbed[row, column] -= 2 * epsilon
                down = float(np.sum(net.forward(perturbed) * weights))
                numeric[row, column] = (up - down) / (2 * epsilon)

        assert np.max(np.abs(input_gradient - numeric)) < 1e-8

    def test_parameter_gradients_are_unchanged(self):
        """The new method must not quietly disagree with the old one."""
        rng = np.random.default_rng(1)
        net = MLP([4, 6, 2], rng=2)
        inputs = rng.normal(size=(5, 4))
        gradient = rng.normal(size=(5, 2))

        net.forward(inputs)
        plain = net.backward(gradient)
        net.forward(inputs)
        packed, _ = net.backward_to_input(gradient)

        assert all(np.array_equal(left, right) for left, right in zip(plain, packed))

    def test_it_refuses_before_a_forward_pass(self):
        with pytest.raises(RuntimeError, match="forward"):
            MLP([3, 2], rng=0).backward_to_input(np.ones((1, 2)))


class TestFeatureNormaliser:

    def test_it_recovers_the_mean_and_scale(self):
        rng = np.random.default_rng(0)
        rows = rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(20_000, 2))
        normaliser = FeatureNormaliser(2)
        for start in range(0, len(rows), 512):
            normaliser.update(rows[start:start + 512])

        assert np.allclose(normaliser.mean, [3.0, -1.0], atol=0.05)
        assert np.allclose(normaliser.scale, [2.0, 0.5], rtol=0.05)

        standardised = normaliser.transform(rows)
        assert np.allclose(standardised.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(standardised.std(axis=0), 1.0, rtol=1e-6)

    def test_batched_updates_match_one_big_update(self):
        """Chan's parallel form has to agree with folding the lot at once."""
        rows = np.random.default_rng(1).normal(size=(500, 3))
        whole = FeatureNormaliser(3).update(rows)
        pieces = FeatureNormaliser(3)
        for start in range(0, 500, 37):
            pieces.update(rows[start:start + 37])

        assert np.allclose(whole.mean, pieces.mean, rtol=1e-12)
        assert np.allclose(whole.m2, pieces.m2, rtol=1e-10)

    def test_a_constant_feature_passes_through_rather_than_exploding(self):
        rows = np.column_stack([np.ones(100), np.arange(100.0)])
        normaliser = FeatureNormaliser(2).update(rows)
        standardised = normaliser.transform(rows)
        assert np.all(np.isfinite(standardised))

    def test_it_passes_data_through_until_it_has_seen_any(self):
        normaliser = FeatureNormaliser(3)
        rows = np.array([[1.0, 2.0, 3.0]])
        assert np.array_equal(normaliser.transform(rows), rows)

    def test_freezing_stops_the_statistics_moving(self):
        normaliser = FeatureNormaliser(2).update(np.zeros((10, 2)))
        normaliser.freeze()
        before = normaliser.mean.copy()
        normaliser.update(np.full((10, 2), 100.0))
        assert np.array_equal(normaliser.mean, before)

    def test_it_rejects_rows_of_the_wrong_width(self):
        with pytest.raises(ValueError, match="width"):
            FeatureNormaliser(3).update(np.zeros((4, 5)))

    def test_state_round_trips(self):
        rows = np.random.default_rng(2).normal(size=(200, 4))
        original = FeatureNormaliser(4).update(rows)
        restored = FeatureNormaliser(4).load_state(original.state())
        assert restored.count == original.count
        assert np.allclose(restored.transform(rows), original.transform(rows))


class TestDualHeadGradients:

    def build(self):
        policy = DualHeadPolicy(trunk=(9,), head=(7,), embedding=5, rng=2,
                                normalise=False)
        # The actor head starts at zero, so at initialisation no gradient
        # reaches the trunk through it — correct, and useless as a test.
        rng = np.random.default_rng(3)
        policy.actor_head.weights[-1][...] = rng.normal(
            size=policy.actor_head.weights[-1].shape)
        return policy

    def test_the_actor_path_matches_finite_differences(self):
        policy = self.build()
        rng = np.random.default_rng(4)
        state = rng.normal(size=STATE_FEATURES)
        actions = rng.normal(size=(4, ACTION_FEATURES))
        seed_gradient = rng.normal(size=4)

        def loss():
            return float(np.sum(policy.logits(state, actions) * seed_gradient))

        policy.logits(state, actions)
        analytic = policy.actor_backward(seed_gradient)["shared"]
        arrays = policy.parameter_groups()["shared"]
        assert finite_difference(loss, arrays, analytic) < 1e-8

    def test_the_critic_path_matches_finite_differences(self):
        policy = self.build()
        rng = np.random.default_rng(5)
        states = rng.normal(size=(6, STATE_FEATURES))
        seed_gradient = rng.normal(size=(6, 1))

        def loss():
            return float(np.sum(policy.critic_forward(states) * seed_gradient))

        policy.critic_forward(states)
        analytic = policy.critic_backward(seed_gradient)["shared"]
        arrays = policy.parameter_groups()["shared"]
        assert finite_difference(loss, arrays, analytic) < 1e-8

    def test_the_trunk_receives_gradient_from_both_heads(self):
        """Otherwise it is two networks that happen to share an array."""
        policy = self.build()
        rng = np.random.default_rng(6)
        state = rng.normal(size=STATE_FEATURES)
        actions = rng.normal(size=(3, ACTION_FEATURES))
        trunk_size = len(policy.trunk.parameters)

        policy.logits(state, actions)
        from_actor = policy.actor_backward(rng.normal(size=3))["shared"]
        policy.critic_forward(rng.normal(size=(4, STATE_FEATURES)))
        from_critic = policy.critic_backward(rng.normal(size=(4, 1)))["shared"]

        assert any(np.any(array != 0) for array in from_actor[:trunk_size])
        assert any(np.any(array != 0) for array in from_critic[:trunk_size])

    def test_the_actor_gradient_sums_over_the_action_rows(self):
        """The embedding is broadcast, so its gradient is a sum, not a mean.

        Doubling the number of identical action rows must double the gradient
        reaching the trunk. Averaging instead would halve it, train more slowly,
        and look like nothing at all.
        """
        policy = self.build()
        rng = np.random.default_rng(7)
        state = rng.normal(size=STATE_FEATURES)
        one = rng.normal(size=(1, ACTION_FEATURES))
        trunk_size = len(policy.trunk.parameters)

        policy.logits(state, one)
        single = policy.actor_backward(np.array([1.0]))["shared"][:trunk_size]
        policy.logits(state, np.vstack([one, one]))
        double = policy.actor_backward(np.array([1.0, 1.0]))["shared"][:trunk_size]

        for left, right in zip(single, double):
            assert np.allclose(2.0 * left, right, rtol=1e-12)

    def test_the_embedding_is_nonlinear(self):
        """Without a nonlinearity the trunk adds nothing over separate nets.

        A linear trunk composed with a linear first layer of each head is one
        linear map, so the architecture would be a reparametrisation rather than
        a change — and the experiment would measure parameter count.
        """
        policy = self.build()
        rng = np.random.default_rng(8)
        states = rng.normal(size=(200, STATE_FEATURES)) * 3.0
        embeddings = policy._embed(states)
        assert np.any(embeddings == 0.0), "ReLU should be clipping somewhere"
        assert np.all(embeddings >= 0.0)


class TestDualHeadInterface:

    def test_it_implements_the_protocol_the_trainer_uses(self):
        policy = DualHeadPolicy(rng=0)
        for method in ("logits", "value", "observe", "freeze_normalisers",
                       "parameter_groups", "actor_backward", "critic_forward",
                       "critic_backward", "inference_parameters", "act",
                       "action_probabilities", "reset", "save", "load", "view"):
            assert hasattr(policy, method), method

    def test_inference_parameters_include_the_trunk(self):
        """Restoring an actor head onto another iteration's trunk is half a policy."""
        separate = ActorCriticPolicy(rng=0)
        assert len(separate.inference_parameters()) == len(separate.actor.parameters)

        shared = DualHeadPolicy(rng=0)
        expected = len(shared.trunk.parameters) + len(shared.actor_head.parameters)
        assert len(shared.inference_parameters()) == expected

    def test_it_starts_with_a_uniform_policy(self):
        """The actor head is zero-initialised, as the separate actor is."""
        policy = DualHeadPolicy(rng=0, normalise=False)
        rng = np.random.default_rng(1)
        observation = (rng.normal(size=STATE_FEATURES),
                       rng.normal(size=(5, ACTION_FEATURES)), None)
        probabilities = policy.action_probabilities(observation)
        assert np.allclose(probabilities, 1.0 / 5)

    def test_a_view_shares_the_weights(self):
        policy = DualHeadPolicy(rng=0)
        view = policy.view(temperature=0.5)
        assert view.trunk is policy.trunk
        assert view.actor_head is policy.actor_head
        assert view.temperature == 0.5

    def test_save_and_load_round_trip(self):
        import tempfile

        policy = DualHeadPolicy(rng=0, normalise=True)
        rng = np.random.default_rng(2)
        policy.observe(rng.normal(size=STATE_FEATURES),
                       rng.normal(size=(4, ACTION_FEATURES)))
        for array in policy.actor_head.parameters:
            array += rng.normal(size=array.shape) * 0.1

        state = rng.normal(size=STATE_FEATURES)
        actions = rng.normal(size=(4, ACTION_FEATURES))
        before = policy.logits(state, actions)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/policy.npz"
            policy.save(path)
            restored = DualHeadPolicy(rng=9, normalise=True)
            restored.load(path)
            assert np.allclose(restored.logits(state, actions), before)

    def test_the_shared_group_holds_every_parameter(self):
        policy = DualHeadPolicy(rng=0)
        group = policy.parameter_groups()["shared"]
        expected = (len(policy.trunk.parameters) + len(policy.actor_head.parameters)
                    + len(policy.critic_head.parameters))
        assert len(group) == expected

    def test_gradient_packing_lines_up_with_the_group(self):
        """A mis-ordered pack would apply the trunk's gradient to a head."""
        policy = DualHeadPolicy(trunk=(6,), head=(5,), embedding=4, rng=0,
                                normalise=False)
        rng = np.random.default_rng(3)
        policy.logits(rng.normal(size=STATE_FEATURES),
                      rng.normal(size=(3, ACTION_FEATURES)))
        packed = policy.actor_backward(rng.normal(size=3))["shared"]
        for array, gradient in zip(policy.parameter_groups()["shared"], packed):
            assert array.shape == gradient.shape


class TestTrainingBothArchitectures:

    def config(self):
        return TrainingConfig(iterations=3, batch_episodes=4, eval_every=99,
                              eval_episodes=8, seed=0)

    def test_the_same_loop_trains_both(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from task import make_env, make_request

        for policy in (ActorCriticPolicy(hidden=(16, 16), rng=0),
                       DualHeadPolicy(trunk=(16,), head=(16,), embedding=16, rng=0)):
            before = [array.copy() for array in policy.inference_parameters()]
            trained, history = train(make_env, make_request, self.config(),
                                     policy=policy)
            after = trained.inference_parameters()
            assert any(not np.array_equal(left, right)
                       for left, right in zip(before, after)), \
                f"{type(policy).__name__} did not train"
            assert len(history.train_success) == 3

    def test_normalisation_is_off_by_default_on_the_shipped_policy(self):
        """The phase-2 number was measured without it; enabling it silently
        would change a reported result."""
        assert ActorCriticPolicy(rng=0).normaliser is None
        assert ActorCriticPolicy(rng=0, normalise=True).normaliser is not None
