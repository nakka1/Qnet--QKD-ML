"""The simulation state an agent acts on: pairs, memories, and the clock.

The state is deliberately small and explicit.  Everything the physics needs is
here and nothing else is: an entangled pair is a Bell-diagonal 4-vector, the two
nodes holding it, and the time it was created.  Fidelity is *not* stored — it is
derived from the state vector and the elapsed time, so it can never drift out of
sync with the clock, which is the single easiest bug to write in a repeater
simulator and the hardest to notice.

Memory is the binding constraint and it is modelled as such.  Each node has a
fixed number of slots; every pair occupies one slot at each of its two ends.
Purification consumes two pairs to make one, swapping consumes two to make one,
and generation needs a free slot at both ends before it can even be attempted.
An agent that does not respect that will find its legal action set empty, which
is a more honest failure than a simulator that quietly conjures memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from ..physics import bell_diagonal as bds
from ..physics.memory import MemoryParameters

__all__ = ["EntangledPair", "NetworkState", "Request"]


@dataclass
class EntangledPair:
    """One entangled pair held between two nodes.

    Attributes:
        pair_id: unique within the episode, assigned by
            :meth:`NetworkState.add_pair`.  Per-episode rather than global, so
            two runs from the same seed produce identical traces — which is
            what makes a paired comparison between two policies a comparison of
            their decisions rather than of their luck.  ``-1`` until stored.
        node_a, node_b: the two nodes holding it.  Ordered so ``node_a`` is the
            one closer to the request source along the pair's history — this is
            only a labelling convenience and carries no physics.
        state: Bell-diagonal 4-vector **as of** ``last_update_s``.
        created_s: when the pair was first heralded.  Its age drives the cutoff.
        last_update_s: when ``state`` was last brought up to date.  Decoherence
            between this and "now" has not been applied yet.
        hops: how many elementary links this pair spans.  One when heralded,
            summed on each swap.  A useful feature and a useful invariant.
        distance_km: total fibre length the pair spans.  Not cosmetic — it sets
            the classical signalling latency of every operation on the pair,
            which is what makes purifying a long pair expensive in a way that
            purifying a short one is not.
        purifications: how many purification rounds went into it.
    """

    pair_id: int
    node_a: str
    node_b: str
    state: NDArray[np.float64]
    created_s: float
    last_update_s: float
    hops: int = 1
    distance_km: float = 0.0
    purifications: int = 0

    @classmethod
    def new(cls, node_a: str, node_b: str, state: NDArray[np.float64],
            now_s: float, hops: int = 1, distance_km: float = 0.0) -> "EntangledPair":
        """An unstored pair.  :meth:`NetworkState.add_pair` gives it its id."""
        return cls(pair_id=-1, node_a=node_a, node_b=node_b,
                   state=np.asarray(state, dtype=float), created_s=now_s,
                   last_update_s=now_s, hops=hops, distance_km=distance_km)

    @property
    def nodes(self) -> tuple[str, str]:
        return (self.node_a, self.node_b)

    def other_end(self, node: str) -> str:
        """The node at the far end from ``node``.

        Raises:
            ValueError: if ``node`` does not hold this pair.
        """
        if node == self.node_a:
            return self.node_b
        if node == self.node_b:
            return self.node_a
        raise ValueError(f"{node!r} does not hold pair {self.pair_id}")

    def age_s(self, now_s: float) -> float:
        return now_s - self.created_s

    def state_at(self, now_s: float, memory: MemoryParameters) -> NDArray[np.float64]:
        """The pair's Bell-diagonal state brought forward to ``now_s``.

        Both halves idle, so the single-qubit channel applies at both ends.
        """
        elapsed = max(now_s - self.last_update_s, 0.0)
        if elapsed <= 0.0:
            return self.state
        x_rate, y_rate, z_rate = memory.pauli_rates
        return bds.decohere(self.state, x_rate, y_rate, z_rate, elapsed,
                            both_qubits=True)

    def fidelity_at(self, now_s: float, memory: MemoryParameters) -> float:
        return float(bds.fidelity(self.state_at(now_s, memory)))

    def advance_to(self, now_s: float, memory: MemoryParameters) -> None:
        """Bring the stored state up to ``now_s`` in place."""
        if now_s > self.last_update_s:
            self.state = self.state_at(now_s, memory)
            self.last_update_s = now_s


@dataclass(frozen=True)
class Request:
    """What the episode is trying to deliver.

    Attributes:
        source, destination: the two end nodes.
        min_fidelity: below this, a delivered pair does not count.  This is the
            whole reason purification is ever worth its cost.
        deadline_s: the episode fails at this time.
        pairs_needed: how many qualifying pairs to deliver.  More than one turns
            the problem from "find a path" into "keep a path running", which is
            a different and harder policy.
    """

    source: str
    destination: str
    min_fidelity: float = 0.8
    deadline_s: float = 1.0
    pairs_needed: int = 1


@dataclass
class NetworkState:
    """Everything that changes during an episode.

    Attributes:
        now_s: simulation clock, in seconds.
        pairs: live pairs by id.
        memory_used: occupied slots per node.
        memory_size: slots per node.
        delivered: pairs that met the request, in delivery order.
        discarded: count of pairs lost to the cutoff, for diagnostics.
        attempts: count of generation attempts, for diagnostics — the resource
            an agent spends most of and is least often charged for.
    """

    now_s: float = 0.0
    pairs: dict[int, EntangledPair] = field(default_factory=dict)
    memory_used: dict[str, int] = field(default_factory=dict)
    memory_size: int = 10
    delivered: list[EntangledPair] = field(default_factory=list)
    discarded: int = 0
    attempts: int = 0
    purifications: int = 0
    swaps: int = 0
    _next_pair_id: int = 0

    # -- memory -------------------------------------------------------------

    def free_slots(self, node: str) -> int:
        return self.memory_size - self.memory_used.get(node, 0)

    def can_hold(self, node_a: str, node_b: str) -> bool:
        """Is there room for one more pair between these two nodes?"""
        return self.free_slots(node_a) >= 1 and self.free_slots(node_b) >= 1

    def _occupy(self, pair: EntangledPair) -> None:
        for node in pair.nodes:
            self.memory_used[node] = self.memory_used.get(node, 0) + 1

    def _release(self, pair: EntangledPair) -> None:
        for node in pair.nodes:
            self.memory_used[node] = max(self.memory_used.get(node, 0) - 1, 0)

    # -- pairs --------------------------------------------------------------

    def add_pair(self, pair: EntangledPair) -> None:
        """Store a pair, assigning its id and occupying a slot at each end.

        Raises:
            RuntimeError: if either end is full.  Callers check
                :meth:`can_hold` first; reaching this means an action was
                emitted that should not have been legal.
        """
        if not self.can_hold(*pair.nodes):
            raise RuntimeError(
                f"no free memory for a pair between {pair.nodes}")
        pair.pair_id = self._next_pair_id
        self._next_pair_id += 1
        self.pairs[pair.pair_id] = pair
        self._occupy(pair)

    def remove_pair(self, pair_id: int) -> EntangledPair:
        pair = self.pairs.pop(pair_id)
        self._release(pair)
        return pair

    def pairs_on(self, node_a: str, node_b: str) -> list[EntangledPair]:
        """Every pair held between exactly these two nodes."""
        wanted = {node_a, node_b}
        return [pair for pair in self.pairs.values() if set(pair.nodes) == wanted]

    def pairs_at(self, node: str) -> list[EntangledPair]:
        """Every pair with one end at ``node``."""
        return [pair for pair in self.pairs.values() if node in pair.nodes]

    # -- time ---------------------------------------------------------------

    def advance(self, duration_s: float, memory: MemoryParameters) -> int:
        """Move the clock forward, decohere everything, drop what expired.

        Returns:
            How many pairs were discarded for exceeding the cutoff.

        Note:
            Decoherence is applied *before* the cutoff test, so a pair that
            crosses the cutoff during this interval is dropped rather than
            being usable at its decayed fidelity for the rest of the step.
        """
        if duration_s < 0:
            raise ValueError("time cannot run backwards")
        self.now_s += duration_s

        expired = [pair.pair_id for pair in self.pairs.values()
                   if pair.age_s(self.now_s) > memory.cutoff_time_s]
        for pair_id in expired:
            self.remove_pair(pair_id)
        self.discarded += len(expired)

        for pair in self.pairs.values():
            pair.advance_to(self.now_s, memory)

        return len(expired)

    # -- diagnostics --------------------------------------------------------

    def summary(self, memory: MemoryParameters) -> dict[str, float]:
        fidelities = [pair.fidelity_at(self.now_s, memory)
                      for pair in self.pairs.values()]
        return {
            "now_s": self.now_s,
            "live_pairs": len(self.pairs),
            "mean_fidelity": float(np.mean(fidelities)) if fidelities else float("nan"),
            "delivered": len(self.delivered),
            "discarded": self.discarded,
            "attempts": self.attempts,
            "purifications": self.purifications,
            "swaps": self.swaps,
        }
