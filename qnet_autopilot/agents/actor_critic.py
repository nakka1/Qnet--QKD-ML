"""The learned policy: an action-scoring actor with a state-value critic.

Architecture
------------
The actor scores **each legal action independently** from
``concat(state_features, action_features)`` and takes a softmax over the
scores.  There is no fixed-size output layer, so the number of legal actions
can change every step — which it does, constantly — and the same weights work
on a topology the agent never trained on.  This is the pointer-network shape,
and it is the natural one for an action set that is enumerated rather than
indexed.

The critic is an ordinary value network on the state features alone.  It does
not see the action set, because its job is to say how good the *situation* is,
and giving it the action list would let it launder information the advantage
estimate should not contain.

Training
--------
Advantage actor-critic with GAE(λ), an entropy bonus, and gradient clipping.
Not PPO: with a hand-written backward pass, the extra machinery of a clipped
surrogate and multiple epochs over a buffer is more places to be subtly wrong
than it is worth at this scale, and A2C with a well-tuned entropy schedule
solves this task.

Three details that matter more here than the algorithm choice:

* **Advantages are normalised per batch.**  Returns on this task are dominated
  by whether a rare delivery happened; without normalisation the update scale
  swings by an order of magnitude between batches.
* **The entropy bonus decays.**  Early on the agent must try purifying at all
  — the action is only reachable after building two pairs on one link, which a
  greedy-by-immediate-reward policy never does. Late on, entropy costs
  precision.
* **Evaluation is unshaped, and it samples.**  Training uses the shaped reward;
  every reported number uses the raw outcome, which is the only way a
  comparison against the heuristics means anything. Actions are *sampled* at
  the temperature the policy was trained at, not greedified — see below.

Why the policy is not greedified
--------------------------------
The obvious thing to do with a trained softmax policy is take the argmax. Here
that costs a third of the performance: on held-out episodes the same weights
score 0.55 greedy and 0.82 sampled. Two reasons, and neither is a bug:

* A2C optimises the return of the **stochastic** policy. The advantages, the
  value function, and the state distribution they were all estimated on belong
  to the soft policy. Greedifying produces a different policy that training
  never evaluated, and nothing guarantees it is better — the policy-improvement
  theorem needs the value function of the policy being greedified, which is not
  what the critic learned.
* The features are aggregates, so distinct physical situations can produce
  identical vectors. Under that aliasing the problem is a POMDP from the
  policy's point of view, and there a stochastic policy can strictly dominate
  every deterministic one.

``temperature`` exposes the knob: 0 is greedy, 1 is as trained.
``scripts/temperature_sweep.py`` measures the curve, and it peaks at the
temperature the policy was trained at, which is what the first explanation
predicts.

Model selection
---------------
Policy gradient on this task finds a good policy and then walks away from it:
greedy success can swing from 0.80 to 0.40 within twenty updates. :func:`train`
therefore keeps the best-scoring weights it has seen, on a fixed *validation*
seed range, and returns those rather than whatever the last update happened to
produce.

That makes the validation score an optimistic estimate — it is a maximum over
many checkpoints — so it is not the number to quote.  The honest figure comes
from re-evaluating the selected policy on a **disjoint test seed range**, which
is what ``scripts/evaluate_phase2.py`` reports and what the results table shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..env.features import ACTION_FEATURES, STATE_FEATURES
from .base import Observation
from .normalise import FeatureNormaliser
from .numpy_nn import MLP, Adam, categorical_entropy, softmax

__all__ = ["ActorCriticPolicy", "TrainingConfig", "train"]


@dataclass
class TrainingConfig:
    """Hyperparameters, with the reasoning for the non-obvious ones.

    Attributes:
        hidden: hidden layer widths, shared by actor and critic.
        learning_rate: Adam step size at the start of training.
        learning_rate_end: step size at the end, decayed linearly.  ``None``
            holds the rate constant.  Policy gradient on this task finds a good
            policy quickly and then knocks it over; decaying the step is the
            cheapest way to keep what it found.
        gamma: discount.  Close to 1 because episodes are short and the reward
            is at the end; discounting hard would make the agent short-sighted
            about exactly the decision under study.
        gae_lambda: GAE trace decay.
        entropy_start, entropy_end: entropy bonus, decayed linearly over
            training.  The start value has to be high enough that the agent
            explores purification, which requires two pairs on one link and is
            therefore several unrewarded steps away from any state it starts in.
        value_coefficient: weight on the critic loss.
        batch_episodes: episodes per update.  Larger batches trade wall-clock
            for a less noisy gradient; on a task where success is rare early,
            too small a batch means most updates see no successes at all.
        iterations: number of updates.
        clip_norm: global gradient-norm clip.
        eval_every: run a validation evaluation every this many iterations.
        eval_episodes: episodes per validation evaluation.
        eval_temperature: temperature to validate at.  Defaults to 1.0, the
            temperature the policy is trained at — validating greedily selects
            checkpoints for a policy that will not be the one deployed.
        keep_best: return the best-scoring checkpoint rather than the last.
    """

    hidden: Sequence[int] = (64, 64)
    learning_rate: float = 3e-3
    learning_rate_end: float | None = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_start: float = 0.03
    entropy_end: float = 0.003
    value_coefficient: float = 0.5
    batch_episodes: int = 24
    iterations: int = 300
    clip_norm: float = 1.0
    eval_every: int = 20
    eval_episodes: int = 120
    eval_temperature: float = 1.0
    keep_best: bool = True
    seed: int = 0


class ActorCriticPolicy:
    """Scores legal actions with an MLP; trained by :func:`train`.

    Attributes:
        actor: scores ``concat(state, action)`` to a scalar.
        critic: maps state features to a value estimate.
        normaliser, action_normaliser: running feature statistics, or ``None``.
            Off by default so the shipped result is reproducible; see
            :mod:`qnet_autopilot.agents.normalise` for why turning it on helps.
        greedy: when True, :meth:`act` takes the argmax instead of sampling.
        temperature: softmax temperature for sampling.  1.0 is the policy as
            trained; lower sharpens it, and 0 is equivalent to ``greedy``.
    """

    def __init__(self, hidden: Sequence[int] = (64, 64),
                 rng: np.random.Generator | int | None = None,
                 name: str = "actor_critic", normalise: bool = False):
        generator = np.random.default_rng(rng)
        self.name = name
        self.rng = generator
        self.actor = MLP([STATE_FEATURES + ACTION_FEATURES, *hidden, 1],
                         rng=generator, zero_output=True)
        self.critic = MLP([STATE_FEATURES, *hidden, 1], rng=generator)
        self.greedy = False
        self.temperature = 1.0
        # Off by default: the shipped phase-2 result was measured without it,
        # and turning it on silently would change that number.
        self.normaliser = FeatureNormaliser(STATE_FEATURES) if normalise else None
        self.action_normaliser = (FeatureNormaliser(ACTION_FEATURES) if normalise
                                  else None)

    # -- inference ----------------------------------------------------------

    def _inputs(self, state_features: NDArray[np.float64],
                action_features: NDArray[np.float64]) -> NDArray[np.float64]:
        """One row per action: the state repeated, concatenated with the action."""
        repeated = np.broadcast_to(state_features, (action_features.shape[0],
                                                    state_features.shape[0]))
        return np.concatenate([repeated, action_features], axis=1)

    def action_probabilities(self, observation: Observation,
                             temperature: float | None = None) -> NDArray[np.float64]:
        """Softmax over the legal actions at the given temperature."""
        state_features, action_features, _ = observation
        logits = self.logits(state_features, action_features)
        scale = self.temperature if temperature is None else temperature
        if scale <= 0:
            probabilities = np.zeros_like(logits)
            probabilities[int(np.argmax(logits))] = 1.0
            return probabilities
        return softmax(logits / scale)

    def act(self, observation: Observation) -> int:
        if self.greedy or self.temperature <= 0:
            state_features, action_features, _ = observation
            return int(np.argmax(self.logits(state_features, action_features)))
        probabilities = self.action_probabilities(observation)
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def value(self, state_features: NDArray[np.float64]) -> float:
        return float(self.critic.forward(self._encode_state(state_features)[None, :])[0, 0])

    def reset(self) -> None:
        pass

    # -- training protocol --------------------------------------------------
    #
    # The four methods below are the whole interface `train` uses to touch the
    # networks. They exist so that an architecture with a *shared* trunk can be
    # dropped into the same loop — see :mod:`qnet_autopilot.agents.dual_head`.
    # Without them the comparison between architectures would be a comparison
    # between two training loops as well, and no result could be attributed to
    # either.

    def _encode_state(self, state_features: NDArray[np.float64]
                      ) -> NDArray[np.float64]:
        """Normalise the state, if a normaliser is fitted. Identity otherwise."""
        if self.normaliser is None:
            return np.asarray(state_features, dtype=float)
        return self.normaliser.transform(state_features)

    def _encode_actions(self, action_features: NDArray[np.float64]
                        ) -> NDArray[np.float64]:
        if self.action_normaliser is None:
            return np.asarray(action_features, dtype=float)
        return self.action_normaliser.transform(action_features)

    def observe(self, state_features: NDArray[np.float64],
                action_features: NDArray[np.float64]) -> None:
        """Fold observed features into the running statistics, during training."""
        if self.normaliser is not None:
            self.normaliser.update(np.atleast_2d(state_features))
        if self.action_normaliser is not None:
            self.action_normaliser.update(action_features)

    def freeze_normalisers(self) -> None:
        for normaliser in (self.normaliser, self.action_normaliser):
            if normaliser is not None:
                normaliser.freeze()

    def logits(self, state_features: NDArray[np.float64],
               action_features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score every legal action, caching what :meth:`actor_backward` needs."""
        inputs = self._inputs(self._encode_state(state_features),
                              self._encode_actions(action_features))
        return self.actor.forward(inputs)[:, 0]

    def parameter_groups(self) -> dict[str, list[NDArray[np.float64]]]:
        """Parameter sets that get their own optimiser.

        Two here, because the actor and the critic share nothing: updating one
        cannot move the other, so they can carry independent Adam state.
        """
        return {"actor": self.actor.parameters, "critic": self.critic.parameters}

    def inference_parameters(self) -> list[NDArray[np.float64]]:
        """Every array that changes what the policy *does*.

        This is what the best-checkpoint logic saves and restores. The critic
        is not in it: it shapes training and is never consulted once the policy
        is acting, so keeping the critic from a different iteration than the
        actor costs nothing and saves copying it every evaluation.
        """
        return self.actor.parameters

    def actor_backward(self, logit_gradient: NDArray[np.float64]
                       ) -> dict[str, list[NDArray[np.float64]]]:
        return {"actor": self.actor.backward(logit_gradient[:, None])}

    def critic_forward(self, state_batch: NDArray[np.float64]) -> NDArray[np.float64]:
        encoded = np.stack([self._encode_state(row) for row in state_batch])
        return self.critic.forward(encoded)

    def critic_backward(self, value_gradient: NDArray[np.float64]
                        ) -> dict[str, list[NDArray[np.float64]]]:
        return {"critic": self.critic.backward(value_gradient)}

    # -- persistence --------------------------------------------------------

    def save(self, path) -> None:
        """Write the weights to a ``.npz``."""
        arrays = {}
        for index, array in enumerate(self.actor.parameters):
            arrays[f"actor_{index}"] = array
        for index, array in enumerate(self.critic.parameters):
            arrays[f"critic_{index}"] = array
        for prefix, normaliser in (("state_norm", self.normaliser),
                                   ("action_norm", self.action_normaliser)):
            if normaliser is not None:
                for key, value in normaliser.state().items():
                    arrays[f"{prefix}_{key}"] = value
        np.savez(path, **arrays)

    def load(self, path) -> None:
        """Read weights written by :meth:`save`."""
        data = np.load(path)
        for index, array in enumerate(self.actor.parameters):
            array[...] = data[f"actor_{index}"]
        for index, array in enumerate(self.critic.parameters):
            array[...] = data[f"critic_{index}"]
        for prefix, normaliser in (("state_norm", self.normaliser),
                                   ("action_norm", self.action_normaliser)):
            if normaliser is not None and f"{prefix}_count" in data:
                normaliser.load_state({key: data[f"{prefix}_{key}"]
                                       for key in ("count", "mean", "m2")})
                normaliser.freeze()

    def view(self, temperature: float = 1.0, rng=None,
             name: str | None = None) -> "ActorCriticPolicy":
        """Another handle on the same weights, at a different temperature.

        Shares the parameter arrays, so a view of a policy being trained tracks
        it. Used for evaluation without copying the network.
        """
        clone = ActorCriticPolicy.__new__(ActorCriticPolicy)
        clone.name = name or self.name
        clone.rng = self.rng if rng is None else np.random.default_rng(rng)
        clone.actor = self.actor
        clone.critic = self.critic
        clone.normaliser = self.normaliser
        clone.action_normaliser = self.action_normaliser
        clone.temperature = temperature
        clone.greedy = temperature <= 0
        return clone


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _generalised_advantages(rewards: Sequence[float], values: Sequence[float],
                            gamma: float, lam: float) -> tuple[NDArray, NDArray]:
    """GAE(λ) advantages and the value targets that go with them.

    The episode always ends in a terminal state — delivered, or out of time —
    so the bootstrap value past the end is zero, with no truncation correction
    needed.
    """
    length = len(rewards)
    advantages = np.zeros(length)
    running = 0.0
    for index in reversed(range(length)):
        next_value = values[index + 1] if index + 1 < length else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
    returns = advantages + np.asarray(values)
    return advantages, returns


@dataclass
class TrainingHistory:
    """What happened during training, for the learning curve."""

    iteration: list[int] = field(default_factory=list)
    train_return: list[float] = field(default_factory=list)
    train_success: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    eval_iteration: list[int] = field(default_factory=list)
    eval_success: list[float] = field(default_factory=list)
    eval_return: list[float] = field(default_factory=list)
    eval_fidelity: list[float] = field(default_factory=list)
    best_validation: float = float("nan")

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame({
            "iteration": self.iteration,
            "train_return": self.train_return,
            "train_success": self.train_success,
            "entropy": self.entropy,
            "grad_norm": self.grad_norm,
        })

    def eval_frame(self):
        import pandas as pd

        return pd.DataFrame({
            "iteration": self.eval_iteration,
            "eval_success": self.eval_success,
            "eval_return": self.eval_return,
            "eval_fidelity": self.eval_fidelity,
        })


def train(env_factory: Callable[[], "object"],
          request_factory: Callable[[np.random.Generator], "object"],
          config: TrainingConfig | None = None,
          memory_size: int = 6,
          policy: ActorCriticPolicy | None = None,
          eval_seed: int = 900_000,
          progress: Callable[[str], None] | None = None,
          ) -> tuple[ActorCriticPolicy, TrainingHistory]:
    """Train an actor-critic policy on the environment.

    Args:
        env_factory: builds a fresh environment.
        request_factory: draws a request from a seeded generator.
        config: hyperparameters.
        memory_size: memory slots per node.
        policy: continue training an existing policy instead of starting fresh.
        eval_seed: base seed for the held-out evaluation episodes.  Disjoint
            from the training seeds, so the reported curve is not the training
            set.
        progress: called with a one-line summary at each evaluation.

    Returns:
        ``(policy, history)``.
    """
    from .rollout import run_episode

    config = config or TrainingConfig()
    rng = np.random.default_rng(config.seed)
    policy = policy or ActorCriticPolicy(hidden=config.hidden, rng=config.seed)

    groups = policy.parameter_groups()
    optimisers = {name: Adam(learning_rate=config.learning_rate,
                             clip_norm=config.clip_norm) for name in groups}

    env = env_factory()
    history = TrainingHistory()
    episode_seed = 0
    best_score = -np.inf
    best_weights: list[NDArray[np.float64]] | None = None

    for iteration in range(config.iterations):
        fraction = iteration / max(config.iterations - 1, 1)
        entropy_coefficient = (config.entropy_start
                               + fraction * (config.entropy_end - config.entropy_start))
        if config.learning_rate_end is not None:
            rate = (config.learning_rate
                    + fraction * (config.learning_rate_end - config.learning_rate))
            for optimiser in optimisers.values():
                optimiser.learning_rate = rate

        batch_inputs: list[NDArray] = []
        batch_choice: list[int] = []
        batch_advantage: list[float] = []
        batch_states: list[NDArray] = []
        batch_returns: list[float] = []
        batch_success: list[bool] = []
        batch_return_value: list[float] = []

        policy.greedy = False
        for _ in range(config.batch_episodes):
            episode_seed += 1
            request = request_factory(np.random.default_rng(episode_seed))
            episode = run_episode(env, policy, request=request,
                                  memory_size=memory_size, seed=episode_seed,
                                  collect=True)

            rewards = [step.reward for step in episode.transitions]
            states = [step.state_features for step in episode.transitions]
            values = [policy.value(state) for state in states]
            advantages, returns = _generalised_advantages(
                rewards, values, config.gamma, config.gae_lambda)

            for step, advantage, target in zip(episode.transitions, advantages, returns):
                # Stored as (state, actions) rather than the concatenated matrix
                # the actor eats: with a shared trunk the state has to go
                # through the encoder, and the encoder moves between updates.
                batch_inputs.append((step.state_features, step.action_features))
                batch_choice.append(step.action_index)
                batch_advantage.append(float(advantage))
                policy.observe(step.state_features, step.action_features)
            batch_states.extend(states)
            batch_returns.extend(returns.tolist())
            batch_success.append(episode.succeeded)
            batch_return_value.append(episode.raw_return)

        advantage_array = np.asarray(batch_advantage)
        # Per-batch normalisation: the scale of the advantage on this task is
        # set by how many rare deliveries the batch happened to contain, which
        # is not information the step size should inherit.
        advantage_array = ((advantage_array - advantage_array.mean())
                           / (advantage_array.std() + 1e-8))

        # One accumulator per parameter group. With separate actor and critic
        # networks each group is written by exactly one of the two losses; with
        # a shared trunk both write into the same group, and summing them here
        # is what makes it a shared trunk rather than two networks that happen
        # to overlap.
        accumulated = {name: [np.zeros_like(array) for array in arrays]
                       for name, arrays in groups.items()}
        entropies = []

        for (state_features, action_features), choice, advantage in zip(
                batch_inputs, batch_choice, advantage_array):
            logits = policy.logits(state_features, action_features)
            probabilities = softmax(logits)
            entropy = categorical_entropy(probabilities)
            entropies.append(entropy)

            # Policy gradient plus the entropy bonus, both as d(loss)/d(logits).
            gradient = probabilities.copy()
            gradient[choice] -= 1.0
            gradient *= advantage
            log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
            gradient += entropy_coefficient * probabilities * (log_probabilities + entropy)

            for name, arrays in policy.actor_backward(gradient).items():
                for target, source in zip(accumulated[name], arrays):
                    target += source

        scale = 1.0 / max(len(batch_inputs), 1)
        for name in ("actor", "shared"):
            if name in accumulated:
                accumulated[name] = [array * scale for array in accumulated[name]]

        state_batch = np.stack(batch_states)
        target_batch = np.asarray(batch_returns)[:, None]
        predicted = policy.critic_forward(state_batch)
        value_gradient = (2.0 * config.value_coefficient
                          * (predicted - target_batch) / len(target_batch))
        for name, arrays in policy.critic_backward(value_gradient).items():
            for target, source in zip(accumulated[name], arrays):
                target += source

        norms = {name: optimisers[name].step(groups[name], accumulated[name])
                 for name in groups}
        grad_norm = norms.get("actor", norms.get("shared", 0.0))

        history.iteration.append(iteration)
        history.train_return.append(float(np.mean(batch_return_value)))
        history.train_success.append(float(np.mean(batch_success)))
        history.entropy.append(float(np.mean(entropies)))
        history.grad_norm.append(float(grad_norm))

        should_evaluate = (iteration % config.eval_every == 0
                           or iteration == config.iterations - 1)
        if should_evaluate:
            success, mean_return, fidelity = _validate(
                env_factory, request_factory, policy, config.eval_episodes,
                memory_size, eval_seed, config.eval_temperature)
            history.eval_iteration.append(iteration)
            history.eval_success.append(success)
            history.eval_return.append(mean_return)
            history.eval_fidelity.append(fidelity)
            if config.keep_best and success > best_score:
                best_score = success
                best_weights = [array.copy()
                                for array in policy.inference_parameters()]

            if progress is not None:
                marker = " *" if success >= best_score else "  "
                progress(f"iter {iteration:4d}  train {history.train_success[-1]:.2f}  "
                         f"val {success:.3f}{marker} return {mean_return:+.3f}  "
                         f"H {history.entropy[-1]:.3f}  |g| {grad_norm:.2f}")

    if config.keep_best and best_weights is not None:
        for array, saved in zip(policy.inference_parameters(), best_weights):
            array[...] = saved
        history.best_validation = best_score

    return policy, history


def _validate(env_factory, request_factory, policy: ActorCriticPolicy,
              episodes: int, memory_size: int, seed: int,
              temperature: float = 1.0) -> tuple[float, float, float]:
    """Held-out and unshaped, at the temperature the policy is trained at.

    Fixed seeds, so the score moves only when the policy does.
    """
    from .rollout import run_episode

    env = env_factory()
    # A fixed generator: the validation score should reflect the policy, not
    # which way the action sampling happened to fall this time.
    evaluation_policy = policy.view(temperature=temperature, rng=20_260_911)
    successes, returns, fidelities = [], [], []

    for index in range(episodes):
        episode_seed = seed + index
        request = request_factory(np.random.default_rng(episode_seed))
        episode = run_episode(env, evaluation_policy, request=request,
                              memory_size=memory_size, seed=episode_seed, collect=False)
        successes.append(episode.succeeded)
        returns.append(episode.raw_return)
        fidelities.extend(episode.fidelities)

    return (float(np.mean(successes)), float(np.mean(returns)),
            float(np.mean(fidelities)) if fidelities else float("nan"))
