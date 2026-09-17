"""Heuristics the learned policy has to beat.

A reinforcement-learning result without a competent baseline is not a result.
The temptation is to compare against a random policy, which anything beats, and
call the gap "learning" — so the baselines here are the ones an engineer would
actually deploy, written to be good rather than to be beaten:

:class:`RandomPolicy`
    The floor.  Useful only to show the task is not trivially solvable.

:class:`ShortestPathPolicy`
    What a classical network would do, adapted to repeaters: cover the
    fewest-kilometres path segment by segment, swap as soon as two adjacent
    pairs exist, deliver when the end-to-end pair clears the threshold, discard
    what is in the way.  Never purifies.  The honest "no repeater intelligence"
    comparison, and on short paths with good hardware it is hard to beat.

:class:`ThresholdPolicy`
    The same, plus the standard repeater rule: purify while a pair is below a
    fidelity target and the closed form says the round would help.  This is the
    policy most papers describe, with the threshold as a tuned constant, and it
    is the real bar.  A learned policy earns its keep by deciding *when* to
    purify from the state — how much time is left, how full memory is, how many
    swaps remain — rather than from a constant that can see none of that.

:class:`GreedyFidelityPolicy`
    A negative control: optimise quality, ignore the clock.

One lesson is baked into all of them.  The first version generated on whichever
path link was most likely to herald, which on a uniform chain is always the
same link; it refilled the near end of the path forever, never attempted the
far end, filled node 0's memory with useless two-hop pairs and deadlocked.  It
scored **below random**.  Covering *uncovered* segments first is the fix, and
it is why :data:`~qnet_autopilot.env.features.ActionFeature.SEGMENT_UNCOVERED`
exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..env.actions import Action, ActionKind
from ..env.features import ActionFeature as F
from .base import Observation

__all__ = ["RandomPolicy", "ShortestPathPolicy", "ThresholdPolicy", "GreedyFidelityPolicy"]

#: Fidelity below which a pair is not worth the memory slot it occupies.
_DEAD_PAIR_FIDELITY = 0.5


def _indices_of(actions: Sequence[Action], kind: ActionKind) -> list[int]:
    return [index for index, action in enumerate(actions) if action.kind is kind]


class RandomPolicy:
    """Uniform over legal actions. The floor, not a contender."""

    name = "random"

    def __init__(self, rng: np.random.Generator | int | None = None):
        self.rng = np.random.default_rng(rng)

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> int:
        _, _, actions = observation
        return int(self.rng.integers(len(actions)))


@dataclass
class _PathPolicy:
    """Shared machinery: the deliver / swap / generate / clean-up ladder."""

    name: str = "path"
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def reset(self) -> None:
        pass

    # -- rungs, highest priority first --------------------------------------

    def _deliver(self, features, actions) -> int | None:
        candidates = _indices_of(actions, ActionKind.DELIVER)
        if not candidates:
            return None
        # A deliverable pair only gets worse from here.
        return max(candidates, key=lambda index: features[index, F.FIDELITY_MARGIN])

    def _swap(self, features, actions) -> int | None:
        candidates = [index for index in _indices_of(actions, ActionKind.SWAP)
                      if features[index, F.PROGRESS_GAIN] > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda index: (features[index, F.REACHES_ENDPOINTS],
                                                  features[index, F.PROGRESS_GAIN],
                                                  features[index, F.PREDICTED_FIDELITY]))

    def _generate(self, features, actions) -> int | None:
        candidates = _indices_of(actions, ActionKind.GENERATE)
        if not candidates:
            return None
        uncovered = [index for index in candidates
                     if features[index, F.SEGMENT_UNCOVERED] > 0]
        if uncovered:
            # Cover the path before deepening any part of it.
            return max(uncovered, key=lambda index: features[index, F.LINK_SUCCESS_PROB])

        on_path = [index for index in candidates
                   if features[index, F.ON_REFERENCE_PATH] > 0]
        if on_path:
            # Everything is covered, so deepen the *emptiest* link. A chain has
            # a bottleneck and it is whichever link has fewest pairs; piling
            # more onto the link that already has some cannot help, because a
            # purification round needs two pairs on the *same* link and a swap
            # needs one on every link.
            return min(on_path, key=lambda index: (features[index, F.LINK_OCCUPANCY],
                                                   -features[index, F.LINK_SUCCESS_PROB]))
        return None

    def _discard(self, features, actions) -> int | None:
        """Free a slot held by a pair that can no longer contribute.

        Two kinds: pairs that have decohered past usefulness, and pairs that
        wander off the reference path (a swap at an endpoint produces these).
        Holding them is how a memory bank deadlocks.
        """
        candidates = _indices_of(actions, ActionKind.DISCARD)
        if not candidates:
            return None
        useless = [index for index in candidates
                   if features[index, F.INPUT_FIDELITY_A] < _DEAD_PAIR_FIDELITY
                   or features[index, F.ON_REFERENCE_PATH] < 1.0]
        if not useless:
            return None
        return min(useless, key=lambda index: features[index, F.INPUT_FIDELITY_A])

    def act(self, observation: Observation) -> int:
        _, features, actions = observation
        for rung in self._ladder():
            choice = rung(features, actions)
            if choice is not None:
                return choice
        return 0  # WAIT is always index 0

    def _ladder(self):
        return (self._deliver, self._swap, self._generate, self._discard)


@dataclass
class ShortestPathPolicy(_PathPolicy):
    """Cover the reference path, swap greedily, deliver, never purify."""

    name: str = "shortest_path"


@dataclass
class ThresholdPolicy(_PathPolicy):
    """Shortest-path, plus purify while a pair is below a fidelity target.

    Attributes:
        target_fidelity: purify while the best input is below this.  Set above
            the request threshold, because fidelity falls further during the
            swaps still ahead.
        min_gain: refuse rounds whose predicted gain is below this.  Not a straw
            man — omitting it is the failure mode of the naive version, which
            burns pairs on rounds that cannot help.
        min_time_remaining: refuse to purify when the deadline is close enough
            that the round cannot pay for itself.  The one piece of temporal
            awareness a constant-threshold rule can be given; the learned
            policy gets to discover the rest.
    """

    name: str = "threshold"
    target_fidelity: float = 0.94
    min_gain: float = 0.002
    min_time_remaining: float = 0.15
    salvage_time_remaining: float = 0.35

    def _purify(self, features, actions) -> int | None:
        candidates = []
        for index in _indices_of(actions, ActionKind.PURIFY):
            best_input = max(features[index, F.INPUT_FIDELITY_A],
                             features[index, F.INPUT_FIDELITY_B])
            if best_input >= self.target_fidelity:
                continue
            if features[index, F.FIDELITY_GAIN] <= self.min_gain:
                continue
            candidates.append(index)
        if not candidates:
            return None
        # Expected gain: the improvement, weighted by the chance of getting it.
        return max(candidates, key=lambda index: features[index, F.FIDELITY_GAIN]
                   * features[index, F.SUCCESS_PROB])

    def _quality_swap(self, features, actions) -> int | None:
        """Swap only once both inputs have reached the target.

        This is the nested purify-then-swap protocol, and the ordering is the
        whole point of it: swapping first locks in the error of both inputs,
        and purifying the multi-hop result afterwards is far less effective
        than purifying the links would have been. A heuristic that swaps
        eagerly and purifies whatever it ends up with looks like it is doing
        repeater control and is not — the first version of this class did
        exactly that and never once cleared the fidelity threshold.
        """
        candidates = [index for index in _indices_of(actions, ActionKind.SWAP)
                      if features[index, F.PROGRESS_GAIN] > 0
                      and min(features[index, F.INPUT_FIDELITY_A],
                              features[index, F.INPUT_FIDELITY_B]) >= self.target_fidelity]
        if not candidates:
            return None
        return max(candidates, key=lambda index: (features[index, F.REACHES_ENDPOINTS],
                                                  features[index, F.PROGRESS_GAIN],
                                                  features[index, F.PREDICTED_FIDELITY]))

    def act(self, observation: Observation) -> int:
        state_features, features, actions = observation
        from ..env.features import StateFeature

        time_remaining = state_features[StateFeature.TIME_REMAINING]
        ladder = [self._deliver]
        if time_remaining > self.min_time_remaining:
            ladder.append(self._purify)
        if time_remaining > self.salvage_time_remaining:
            # Plenty of time: hold the quality bar before swapping.
            ladder += [self._quality_swap, self._generate, self._swap]
        else:
            # Running out: take what is on the table rather than nothing.
            ladder += [self._swap, self._generate]
        ladder.append(self._discard)

        for rung in ladder:
            choice = rung(features, actions)
            if choice is not None:
                return choice
        return 0


@dataclass
class GreedyFidelityPolicy(_PathPolicy):
    """Optimise predicted fidelity, ignore the clock entirely.

    A negative control rather than a contender: it shows what a fidelity-only
    objective trains into.  It purifies relentlessly, produces excellent pairs,
    and misses deadlines — which is precisely what the time cost in the reward
    exists to punish.
    """

    name: str = "greedy_fidelity"

    def _improve(self, features, actions) -> int | None:
        scored = [(index, features[index, F.FIDELITY_GAIN] * features[index, F.SUCCESS_PROB])
                  for index, action in enumerate(actions)
                  if action.kind is ActionKind.PURIFY]
        positive = [item for item in scored if item[1] > 0]
        if not positive:
            return None
        return max(positive, key=lambda item: item[1])[0]

    def _ladder(self):
        return (self._deliver, self._improve, self._swap, self._generate, self._discard)
