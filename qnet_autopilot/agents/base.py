"""The seam between a policy and everything else.

One small interface, so the environment never learns anything about how a
decision was made and a policy never learns anything about how the environment
is implemented.  A heuristic, a numpy network, and a PyTorch network you write
on your own machine are all the same shape from here.

That seam is the reason this package can train something real in an
environment with no deep-learning framework available: swapping in a Torch
implementation later means writing one class with two methods, not rewriting
the environment, the rollout loop, the evaluation, or the baselines.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..env.actions import Action

__all__ = ["Policy", "Observation"]

#: What the environment hands a policy: global features, per-action features,
#: and the actions those rows describe, in the same order.
Observation = tuple[NDArray[np.float64], NDArray[np.float64], list[Action]]


@runtime_checkable
class Policy(Protocol):
    """Anything that can choose among the legal actions.

    Implementations must be deterministic given their own RNG state, so a run
    is reproducible from its seed.
    """

    name: str

    def act(self, observation: Observation) -> int:
        """Choose one action.

        Args:
            observation: ``(state_features, action_features, actions)``.  The
                action list is never empty — WAIT is always legal.

        Returns:
            The index of the chosen action within that list.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode.  Stateless policies do nothing."""
