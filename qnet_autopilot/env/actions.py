"""What an agent can do, and what each thing costs it.

The action space is **one primitive per step**, drawn from the set that is legal
right now.  Not a fixed vector with a mask over slots, and not a joint action
over every link at once — a single operation, chosen from a list that changes
shape as the state does.  Three reasons that formulation is the right one here:

* The joint action space over a real topology is astronomically large and
  almost entirely illegal.  Enumerating what is *actually* available is both
  cheaper and better-conditioned.
* Operations are not simultaneous in the physical network either.  A swap at a
  node and a purification on a link happen at different times and take
  different amounts of time.
* It makes the action space topology-independent.  The policy scores each
  legal action from that action's own features, so the same network runs on a
  five-node chain and an eleven-node metro graph without retraining the head.

**Time advances by the latency of the action chosen.**  This is the design
decision the whole environment turns on.  A purification on a 200-km pair costs
a millisecond of classical signalling during which every stored pair in the
network decoheres; a purification on a 25-km link costs an eighth of that.  If
operations were instantaneous, "purify everything, always" would be free and
optimal, and there would be no timing problem to learn.  The literature calls
this repeater *scheduling*; making latency the step cost is what puts it in the
MDP rather than in a footnote.

Every stochastic operation can also destroy what it touches.  Purification
consumes two pairs and returns one *only on success*; entanglement swapping
consumes two and returns one only if the Bell measurement succeeds.  An agent
that purifies a nearly-good pair is making a real bet.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .network_env import EnvironmentConfig
    from .state import NetworkState, Request

__all__ = ["ActionKind", "Action", "legal_actions"]


class ActionKind(enum.IntEnum):
    """The six primitives.  Integer-valued so they one-hot encode directly."""

    WAIT = 0
    GENERATE = 1
    PURIFY = 2
    SWAP = 3
    DELIVER = 4
    DISCARD = 5


@dataclass(frozen=True)
class Action:
    """One legal operation, fully specified.

    Attributes:
        kind: which primitive.
        link: the edge ``(u, v)`` for GENERATE.
        pair_ids: the pairs consumed — two for PURIFY and SWAP, one for
            DELIVER, none otherwise.
        node: the swapping node for SWAP.
    """

    kind: ActionKind
    link: tuple[str, str] | None = None
    pair_ids: tuple[int, ...] = ()
    node: str | None = None

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        if self.kind is ActionKind.WAIT:
            return "WAIT"
        if self.kind is ActionKind.GENERATE:
            return f"GENERATE({self.link[0]}-{self.link[1]})"
        if self.kind is ActionKind.PURIFY:
            return f"PURIFY({self.pair_ids[0]}+{self.pair_ids[1]})"
        if self.kind is ActionKind.SWAP:
            return f"SWAP@{self.node}({self.pair_ids[0]}+{self.pair_ids[1]})"
        if self.kind is ActionKind.DELIVER:
            return f"DELIVER({self.pair_ids[0]})"
        return f"DISCARD({self.pair_ids[0]})"


def legal_actions(state: "NetworkState", request: "Request",
                  config: "EnvironmentConfig") -> list[Action]:
    """Enumerate everything the agent may do in the current state.

    Order is deterministic — WAIT, then GENERATE by edge, then PURIFY, SWAP and
    DELIVER by pair id — so a run is reproducible from its seed and two agents
    can be compared on identical action sets.

    Args:
        state: the live network state.
        request: what the episode is trying to deliver.
        config: topology, hardware and the caps below.

    Returns:
        A non-empty list.  WAIT is always legal, so an agent can never be
        cornered into an illegal move; if everything else is impossible,
        waiting for a pair to expire and free memory is the only thing left,
        and that is a real situation the agent should have to recognise.
    """
    actions: list[Action] = [Action(ActionKind.WAIT)]

    # -- generation ---------------------------------------------------------
    # One action per edge, gated on memory at both ends and on a cap for how
    # many pairs may sit on one link at once. Without that cap an agent can
    # fill every slot with elementary pairs on one edge and deadlock the graph.
    for node_a, node_b in config.edges:
        if not state.can_hold(node_a, node_b):
            continue
        if len(state.pairs_on(node_a, node_b)) >= config.max_pairs_per_link:
            continue
        actions.append(Action(ActionKind.GENERATE, link=(node_a, node_b)))

    # -- purification -------------------------------------------------------
    # Any two pairs spanning the same node set. Sorted by id so the pair kept
    # (the first) is deterministic.
    by_endpoints: dict[frozenset[str], list[int]] = {}
    for pair in state.pairs.values():
        by_endpoints.setdefault(frozenset(pair.nodes), []).append(pair.pair_id)

    for pair_ids in by_endpoints.values():
        if len(pair_ids) < 2:
            continue
        ordered = sorted(pair_ids)
        for index, kept in enumerate(ordered):
            for consumed in ordered[index + 1:]:
                actions.append(Action(ActionKind.PURIFY,
                                      pair_ids=(kept, consumed)))

    # -- swapping -----------------------------------------------------------
    # Two pairs meeting at a node, with distinct far ends. The far ends already
    # hold a slot each for the consumed pairs, so the resulting pair always fits.
    for node in config.nodes:
        incident = sorted(state.pairs_at(node), key=lambda pair: pair.pair_id)
        for index, left in enumerate(incident):
            for right in incident[index + 1:]:
                far_left = left.other_end(node)
                far_right = right.other_end(node)
                if far_left == far_right:
                    # Would produce a pair from a node to itself: not a pair.
                    continue
                if left.hops + right.hops > config.max_hops:
                    continue
                actions.append(Action(ActionKind.SWAP, node=node,
                                      pair_ids=(left.pair_id, right.pair_id)))

    # -- delivery -----------------------------------------------------------
    endpoints = {request.source, request.destination}
    for pair in sorted(state.pairs.values(), key=lambda pair: pair.pair_id):
        if set(pair.nodes) != endpoints:
            continue
        if pair.fidelity_at(state.now_s, config.memory) < request.min_fidelity:
            # Deliverable in principle but not good enough: offering it would
            # let the agent bank a failure. Purify it or let it go.
            continue
        actions.append(Action(ActionKind.DELIVER, pair_ids=(pair.pair_id,)))

    # -- discard ------------------------------------------------------------
    # Releasing a memory is a real control action, and without it the only way
    # out of a full memory bank is to wait for the cutoff. An agent that has
    # filled a node with pairs it can no longer use should be able to say so
    # and move on; the first version of this environment could not, and both
    # the heuristics and the learned policy deadlocked because of it.
    if config.allow_discard:
        for pair in sorted(state.pairs.values(), key=lambda pair: pair.pair_id):
            actions.append(Action(ActionKind.DISCARD, pair_ids=(pair.pair_id,)))

    return actions
