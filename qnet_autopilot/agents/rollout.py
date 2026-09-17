"""Running episodes, and measuring them on terms that are not the agent's own.

Two things matter here and both are easy to get wrong.

**Evaluation must not use the shaped reward.**  Potential-based shaping is safe
for training and meaningless as a score — an agent and a heuristic shaped
differently are not comparable on it.  Every metric reported by
:func:`evaluate` comes from what actually happened: whether a qualifying pair
was delivered, at what fidelity, how long it took, what it consumed.

**Every policy must see the same episodes.**  Success rates on this task swing
by tens of percent between random seeds, because whether a link herald arrives
early or late dominates short episodes.  :func:`evaluate` therefore drives each
policy through the *same* list of seeds, so a comparison is paired rather than
two independent samples.  With 200 episodes the paired difference is worth
reading; the unpaired one, at these variances, is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..env.actions import Action, ActionKind
from ..env.network_env import EnvironmentConfig, QuantumNetworkEnv
from ..env.state import Request
from .base import Observation, Policy

__all__ = ["Transition", "Episode", "run_episode", "evaluate", "EvaluationResult"]


@dataclass
class Transition:
    """One step, kept for training."""

    state_features: np.ndarray
    action_features: np.ndarray
    action_index: int
    reward: float
    raw_reward: float
    done: bool


@dataclass
class Episode:
    """One completed episode and what it cost.

    Attributes:
        transitions: the steps, in order.
        delivered: how many qualifying pairs were delivered.
        succeeded: whether the request was fulfilled.
        fidelities: delivered pair fidelities.
        elapsed_s: simulated time consumed.
        attempts, purifications, swaps, discarded: resource counters.
        request: what the episode was asked for.
    """

    transitions: list[Transition] = field(default_factory=list)
    delivered: int = 0
    succeeded: bool = False
    fidelities: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    steps: int = 0
    attempts: int = 0
    purifications: int = 0
    swaps: int = 0
    discarded: int = 0
    request: Request | None = None

    @property
    def shaped_return(self) -> float:
        return float(sum(step.reward for step in self.transitions))

    @property
    def raw_return(self) -> float:
        """The score. Shaping never enters this."""
        return float(sum(step.raw_reward for step in self.transitions))

    @property
    def mean_fidelity(self) -> float:
        return float(np.mean(self.fidelities)) if self.fidelities else float("nan")


def run_episode(env: QuantumNetworkEnv, policy: Policy,
                request: Request | None = None,
                memory_size: int = 6,
                seed: int | None = None,
                collect: bool = True) -> Episode:
    """Run one episode to termination.

    Args:
        env: the environment.
        policy: anything satisfying :class:`~.base.Policy`.
        request: what to deliver; ``None`` draws one at random.
        memory_size: memory slots per node.
        seed: reseeds the environment, so the same seed gives the same link
            outcomes to every policy compared on it.
        collect: keep transitions.  Turn off for pure evaluation to save memory
            on long sweeps.

    Returns:
        Episode: the record.
    """
    observation, _ = env.reset(request=request, memory_size=memory_size, rng=seed)
    policy.reset()
    episode = Episode(request=env.request)

    while True:
        action_index = policy.act(observation)
        _, _, actions = observation
        if not 0 <= action_index < len(actions):
            raise IndexError(
                f"{policy.name} chose action {action_index} of {len(actions)} legal")

        state_features, action_features, _ = observation
        chosen = actions[action_index]
        observation, result = env.step(chosen)

        if collect:
            episode.transitions.append(Transition(
                state_features=state_features,
                action_features=action_features,
                action_index=action_index,
                reward=result.reward,
                raw_reward=result.raw_reward,
                done=result.done))
        else:
            episode.transitions.append(Transition(
                state_features=np.empty(0), action_features=np.empty(0),
                action_index=action_index, reward=result.reward,
                raw_reward=result.raw_reward, done=result.done))

        if result.info.get("outcome") == "delivered":
            episode.fidelities.append(float(result.info["fidelity"]))

        if result.done:
            episode.succeeded = result.terminated
            break

    state = env.state
    episode.delivered = len(state.delivered)
    episode.elapsed_s = state.now_s
    episode.steps = env.steps
    episode.attempts = state.attempts
    episode.purifications = state.purifications
    episode.swaps = state.swaps
    episode.discarded = state.discarded
    return episode


@dataclass
class EvaluationResult:
    """Per-policy outcomes over a shared set of episodes."""

    frame: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """One row per policy: success rate, timing, fidelity, resources."""
        grouped = self.frame.groupby("policy", sort=False)
        summary = grouped.agg(
            episodes=("succeeded", "size"),
            success_rate=("succeeded", "mean"),
            raw_return=("raw_return", "mean"),
            mean_time_ms=("elapsed_ms", "mean"),
            mean_fidelity=("mean_fidelity", "mean"),
            attempts=("attempts", "mean"),
            purifications=("purifications", "mean"),
            swaps=("swaps", "mean"),
            steps=("steps", "mean"),
        )
        # Standard error of the success rate, so a difference can be read as
        # meaningful or not without squinting.
        counts = grouped["succeeded"].size()
        rates = grouped["succeeded"].mean()
        summary["success_stderr"] = np.sqrt(rates * (1 - rates) / counts)
        return summary.reset_index()

    def paired_difference(self, policy: str, reference: str) -> dict[str, float]:
        """Per-episode difference against a reference policy, with its error.

        Paired on the episode seed: both policies met the same link outcomes,
        so the variance of the difference is far smaller than the variance of
        either rate, and a modest improvement is detectable with hundreds
        rather than thousands of episodes.
        """
        wide = self.frame.pivot(index="seed", columns="policy", values="succeeded")
        if policy not in wide or reference not in wide:
            raise KeyError(f"need both {policy!r} and {reference!r} in the results")
        difference = wide[policy].astype(float) - wide[reference].astype(float)
        return {
            "mean": float(difference.mean()),
            "stderr": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            "wins": int((difference > 0).sum()),
            "losses": int((difference < 0).sum()),
            "ties": int((difference == 0).sum()),
        }


def evaluate(env_factory: Callable[[], QuantumNetworkEnv],
             policies: Sequence[Policy],
             episodes: int = 200,
             request_factory: Callable[[np.random.Generator], Request] | None = None,
             memory_size: int = 6,
             seed: int = 10_000) -> EvaluationResult:
    """Run every policy over the same episodes and record what happened.

    Args:
        env_factory: builds a fresh environment.  A factory rather than an
            instance so each policy gets its own, with no state carried across.
        policies: the policies to compare.
        episodes: how many episodes each policy runs.
        request_factory: draws the request for an episode from a generator
            seeded per episode, so every policy gets identical requests.
        memory_size: memory slots per node.
        seed: base seed; episode *i* uses ``seed + i``.

    Returns:
        EvaluationResult: one row per (policy, episode).
    """
    records = []
    for policy in policies:
        env = env_factory()
        for index in range(episodes):
            episode_seed = seed + index
            request = (request_factory(np.random.default_rng(episode_seed))
                       if request_factory is not None else None)
            episode = run_episode(env, policy, request=request,
                                  memory_size=memory_size, seed=episode_seed,
                                  collect=False)
            records.append({
                "policy": policy.name,
                "seed": episode_seed,
                "succeeded": bool(episode.succeeded),
                "raw_return": episode.raw_return,
                "elapsed_ms": episode.elapsed_s * 1e3,
                "mean_fidelity": episode.mean_fidelity,
                "attempts": episode.attempts,
                "purifications": episode.purifications,
                "swaps": episode.swaps,
                "discarded": episode.discarded,
                "steps": episode.steps,
            })

    return EvaluationResult(frame=pd.DataFrame(records))
