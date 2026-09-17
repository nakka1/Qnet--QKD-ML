"""Turning state and actions into vectors, without leaking and without cheating.

The policy scores **each legal action from that action's own feature vector**,
then takes a softmax over the scores.  That is what makes one trained network
usable on a topology it never saw: nothing in the encoding refers to a node by
name or to a link by index, so a five-node chain and an eleven-node metro graph
produce feature vectors of the same shape and the same meaning.

Everything here is either a probability, a ratio, or a quantity normalised by a
scale the agent could actually know — the deadline, the cutoff time, the longest
span in the graph.  Raw seconds and raw kilometres never reach the network; with
link latencies spanning four orders of magnitude they would make the first
layer's conditioning hopeless.

What the agent is allowed to see
--------------------------------
Everything here is something a controller genuinely has.  It knows the fidelity
of a pair it holds — the pair was heralded with a known raw fidelity, has a
known age, and the decoherence model is calibrated hardware.  It knows the
classical topology and can compute shortest paths.  It can evaluate the closed
form for what a purification or a swap *would* produce, because that is
:mod:`qnet_autopilot.physics`, running locally in microseconds.

It does not know the outcome of a stochastic operation before performing it,
and nothing here tells it.  ``success_prob`` is the probability, not the draw.

Indices are named
-----------------
:class:`StateFeature` and :class:`ActionFeature` are ``IntEnum``\\s, so
baselines and diagnostics index by meaning.  Magic numbers in a heuristic that
reads a 22-element vector are how a policy ends up silently comparing latency
against fidelity.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from ..physics import bell_diagonal as bds
from .actions import Action, ActionKind

if TYPE_CHECKING:  # pragma: no cover
    from .network_env import QuantumNetworkEnv
    from .state import EntangledPair, NetworkState, Request

__all__ = ["StateFeature", "ActionFeature", "STATE_FEATURES", "ACTION_FEATURES",
           "encode_state", "encode_actions", "covered_segments"]


class StateFeature(enum.IntEnum):
    """Global situation, one vector per step."""

    TIME_REMAINING = 0        #: fraction of the deadline still available
    DELIVERED_FRACTION = 1    #: progress against pairs_needed
    LIVE_PAIRS = 2            #: live pairs, normalised by total memory
    MEAN_FIDELITY = 3         #: mean fidelity of live pairs, 0 when none
    BEST_FIDELITY = 4         #: best live pair's fidelity
    MEMORY_PRESSURE_MEAN = 5  #: mean slot occupancy across nodes
    MEMORY_PRESSURE_MAX = 6   #: worst node's occupancy — where deadlock starts
    PATH_PROGRESS = 7         #: reach of the best pair along the reference path
    PATH_COVERAGE = 8         #: fraction of path segments with a covering pair
    MIN_FIDELITY = 9          #: the request's threshold
    STEPS_USED = 10           #: fraction of the step budget consumed


class ActionFeature(enum.IntEnum):
    """One vector per legal action."""

    IS_WAIT = 0
    IS_GENERATE = 1
    IS_PURIFY = 2
    IS_SWAP = 3
    IS_DELIVER = 4
    IS_DISCARD = 5
    LATENCY = 6               #: action latency / deadline
    LINK_SUCCESS_PROB = 7     #: GENERATE: per-round heralding probability
    LINK_LENGTH = 8           #: GENERATE: span / longest span in the graph
    LINK_EXPECTED_ROUNDS = 9  #: GENERATE: expected rounds to herald / budget
    INPUT_FIDELITY_A = 10     #: fidelity of the first pair involved
    INPUT_FIDELITY_B = 11     #: fidelity of the second pair, where there is one
    PREDICTED_FIDELITY = 12   #: closed-form fidelity of the result
    FIDELITY_GAIN = 13        #: predicted minus the better input
    SUCCESS_PROB = 14         #: probability the operation works at all
    PAIR_AGE = 15             #: oldest input pair's age / cutoff time
    RESULT_HOPS = 16          #: hops the result spans / max_hops
    REACHES_ENDPOINTS = 17    #: 1 if the result connects the request's ends
    FIDELITY_MARGIN = 18      #: predicted fidelity minus the threshold, scaled
    ON_REFERENCE_PATH = 19    #: fraction of involved nodes on the reference path
    PROGRESS_GAIN = 20        #: extra path span the result would cover
    SEGMENT_UNCOVERED = 21    #: GENERATE: this path segment has no pair yet
    LINK_OCCUPANCY = 22       #: GENERATE: pairs already on this link / the cap


STATE_FEATURES = len(StateFeature)
ACTION_FEATURES = len(ActionFeature)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _span_on_path(pair: "EntangledPair", position: dict[str, int]) -> int:
    """How many reference-path hops a pair covers; zero if it leaves the path."""
    left = position.get(pair.node_a)
    right = position.get(pair.node_b)
    if left is None or right is None:
        return 0
    return abs(right - left)


def covered_segments(state: "NetworkState", position: dict[str, int]) -> set[int]:
    """Which reference-path segments already have a pair spanning them.

    Segment *i* runs between path positions *i* and *i+1*.  A pair from
    position 0 to position 2 covers segments 0 and 1.

    This is the piece of state a repeater controller most obviously needs and
    that a naive heuristic most obviously lacks: without it, "generate on the
    link most likely to herald" keeps re-generating on the same fast link while
    the far end of the path is never even attempted, filling memory with
    useless pairs until the network deadlocks.  That is not a hypothetical —
    it is what the first version of :class:`~qnet_autopilot.agents.baselines.ShortestPathPolicy`
    did, and it scored below random.
    """
    covered: set[int] = set()
    for pair in state.pairs.values():
        left = position.get(pair.node_a)
        right = position.get(pair.node_b)
        if left is None or right is None:
            continue
        for index in range(min(left, right), max(left, right)):
            covered.add(index)
    return covered


def encode_state(env: "QuantumNetworkEnv", state: "NetworkState",
                 request: "Request") -> NDArray[np.float64]:
    """Global features: the situation the agent is choosing within."""
    memory = env.config.memory
    now = state.now_s

    fidelities = [pair.fidelity_at(now, memory) for pair in state.pairs.values()]
    capacity = max(state.memory_size * len(env.config.nodes), 1)
    occupancy = [state.memory_used.get(node, 0) / max(state.memory_size, 1)
                 for node in env.config.nodes]

    segments = max(len(env._shortest_path) - 1, 1)
    coverage = len(covered_segments(state, env._path_position)) / segments

    features = np.zeros(STATE_FEATURES, dtype=float)
    features[StateFeature.TIME_REMAINING] = np.clip(
        1.0 - _safe_ratio(now, request.deadline_s), 0.0, 1.0)
    features[StateFeature.DELIVERED_FRACTION] = np.clip(
        _safe_ratio(len(state.delivered), request.pairs_needed), 0.0, 1.0)
    features[StateFeature.LIVE_PAIRS] = np.clip(2.0 * len(state.pairs) / capacity, 0.0, 1.0)
    features[StateFeature.MEAN_FIDELITY] = float(np.mean(fidelities)) if fidelities else 0.0
    features[StateFeature.BEST_FIDELITY] = float(np.max(fidelities)) if fidelities else 0.0
    features[StateFeature.MEMORY_PRESSURE_MEAN] = float(np.mean(occupancy)) if occupancy else 0.0
    features[StateFeature.MEMORY_PRESSURE_MAX] = float(np.max(occupancy)) if occupancy else 0.0
    features[StateFeature.PATH_PROGRESS] = env._potential()
    features[StateFeature.PATH_COVERAGE] = coverage
    features[StateFeature.MIN_FIDELITY] = request.min_fidelity
    features[StateFeature.STEPS_USED] = np.clip(
        _safe_ratio(env.steps, env.config.max_steps), 0.0, 1.0)
    return features


def encode_actions(env: "QuantumNetworkEnv", state: "NetworkState",
                   request: "Request", actions: Sequence[Action]
                   ) -> NDArray[np.float64]:
    """One row per legal action, in the order given."""
    if not actions:
        return np.zeros((0, ACTION_FEATURES))
    position = env._path_position
    covered = covered_segments(state, position)
    return np.stack([_encode_action(env, state, request, action, position, covered)
                     for action in actions])


def _encode_action(env: "QuantumNetworkEnv", state: "NetworkState",
                   request: "Request", action: Action,
                   position: dict[str, int], covered: set[int]
                   ) -> NDArray[np.float64]:
    config = env.config
    memory = config.memory
    now = state.now_s
    features = np.zeros(ACTION_FEATURES, dtype=float)
    features[int(action.kind)] = 1.0

    deadline = max(request.deadline_s, 1e-12)
    endpoints = {request.source, request.destination}
    span = max(len(env._shortest_path) - 1, 1)

    def on_path(*nodes: str) -> float:
        if not position:
            return 0.0
        return sum(node in position for node in nodes) / len(nodes)

    if action.kind is ActionKind.WAIT:
        features[ActionFeature.LATENCY] = min(config.wait_slot_s / deadline, 1.0)
        return features

    if action.kind is ActionKind.GENERATE:
        left, right = action.link
        key = config.key(left, right)
        probability = config.link_probability[key]
        round_trip = config.link_round_trip_s[key]
        max_rounds = max(int(config.generate_budget_s // round_trip), 1)

        features[ActionFeature.LATENCY] = min(max_rounds * round_trip / deadline, 1.0)
        features[ActionFeature.LINK_SUCCESS_PROB] = probability
        features[ActionFeature.LINK_LENGTH] = config.length_km[key] / config.max_length_km
        expected_rounds = 1.0 / probability if probability > 0 else float(max_rounds)
        features[ActionFeature.LINK_EXPECTED_ROUNDS] = min(expected_rounds / max_rounds, 1.0)

        # How full this link already is. Without it, "generate on the link most
        # likely to herald" keeps deepening whichever link is nominally fastest
        # — on a uniform chain, always the same one — so no other link ever gets
        # the second pair a purification round needs.
        features[ActionFeature.LINK_OCCUPANCY] = min(
            len(state.pairs_on(left, right)) / max(config.max_pairs_per_link, 1), 1.0)

        if left in position and right in position:
            lower, upper = sorted((position[left], position[right]))
            adjacent = upper - lower == 1
            features[ActionFeature.ON_REFERENCE_PATH] = float(adjacent)
            if adjacent:
                features[ActionFeature.SEGMENT_UNCOVERED] = float(lower not in covered)
        return features

    if action.kind is ActionKind.DELIVER:
        (pair_id,) = action.pair_ids
        pair = state.pairs[pair_id]
        fidelity = pair.fidelity_at(now, memory)
        features[ActionFeature.INPUT_FIDELITY_A] = fidelity
        features[ActionFeature.PREDICTED_FIDELITY] = fidelity
        features[ActionFeature.SUCCESS_PROB] = 1.0
        features[ActionFeature.PAIR_AGE] = min(
            _safe_ratio(pair.age_s(now), memory.cutoff_time_s), 1.0)
        features[ActionFeature.RESULT_HOPS] = min(pair.hops / max(config.max_hops, 1), 1.0)
        features[ActionFeature.REACHES_ENDPOINTS] = 1.0
        features[ActionFeature.FIDELITY_MARGIN] = _safe_ratio(
            fidelity - request.min_fidelity, max(1.0 - request.min_fidelity, 1e-9))
        features[ActionFeature.ON_REFERENCE_PATH] = 1.0
        features[ActionFeature.PROGRESS_GAIN] = 1.0
        return features

    if action.kind is ActionKind.DISCARD:
        (pair_id,) = action.pair_ids
        pair = state.pairs[pair_id]
        fidelity = pair.fidelity_at(now, memory)
        features[ActionFeature.INPUT_FIDELITY_A] = fidelity
        features[ActionFeature.PREDICTED_FIDELITY] = fidelity
        features[ActionFeature.SUCCESS_PROB] = 1.0
        features[ActionFeature.PAIR_AGE] = min(
            _safe_ratio(pair.age_s(now), memory.cutoff_time_s), 1.0)
        features[ActionFeature.RESULT_HOPS] = min(pair.hops / max(config.max_hops, 1), 1.0)
        features[ActionFeature.REACHES_ENDPOINTS] = float(set(pair.nodes) == endpoints)
        features[ActionFeature.ON_REFERENCE_PATH] = on_path(*pair.nodes)
        # Discarding costs progress, so the gain is the span given up.
        features[ActionFeature.PROGRESS_GAIN] = -_span_on_path(pair, position) / span
        return features

    first_id, second_id = action.pair_ids
    first = state.pairs[first_id]
    second = state.pairs[second_id]
    first_state = first.state_at(now, memory)
    second_state = second.state_at(now, memory)
    fidelity_a = float(bds.fidelity(first_state))
    fidelity_b = float(bds.fidelity(second_state))

    if action.kind is ActionKind.PURIFY:
        protocol = bds.dejmps if config.purification == "dejmps" else bds.bbpssw
        p_success, result = protocol(first_state, second_state,
                                     gate_fid=config.gate_fid, meas_fid=config.meas_fid)
        predicted = float(bds.fidelity(result))
        latency = config.signalling_delay_s(first.distance_km)
        result_hops = first.hops
        reaches = float(set(first.nodes) == endpoints)
        success = float(np.clip(p_success, 0.0, 1.0))
        path_relevance = on_path(*first.nodes)
        progress_gain = 0.0
    else:  # SWAP
        result = bds.swap(first_state, second_state, gate_fid=config.gate_fid,
                          meas_fid=config.meas_fid, is_twirled=config.twirl_on_swap)
        predicted = float(bds.fidelity(result))
        latency = max(config.signalling_delay_s(first.distance_km),
                      config.signalling_delay_s(second.distance_km))
        result_hops = first.hops + second.hops
        far_left = first.other_end(action.node)
        far_right = second.other_end(action.node)
        reaches = float({far_left, far_right} == endpoints)
        success = float(config.swap_success_prob)
        path_relevance = on_path(far_left, far_right)
        if far_left in position and far_right in position:
            merged_span = abs(position[far_left] - position[far_right])
            best_input = max(_span_on_path(first, position), _span_on_path(second, position))
            progress_gain = (merged_span - best_input) / span
        else:
            progress_gain = 0.0

    features[ActionFeature.LATENCY] = min(latency / deadline, 1.0)
    features[ActionFeature.INPUT_FIDELITY_A] = fidelity_a
    features[ActionFeature.INPUT_FIDELITY_B] = fidelity_b
    features[ActionFeature.PREDICTED_FIDELITY] = predicted
    features[ActionFeature.FIDELITY_GAIN] = predicted - max(fidelity_a, fidelity_b)
    features[ActionFeature.SUCCESS_PROB] = success
    features[ActionFeature.PAIR_AGE] = min(
        _safe_ratio(max(first.age_s(now), second.age_s(now)), memory.cutoff_time_s), 1.0)
    features[ActionFeature.RESULT_HOPS] = min(result_hops / max(config.max_hops, 1), 1.0)
    features[ActionFeature.REACHES_ENDPOINTS] = reaches
    features[ActionFeature.FIDELITY_MARGIN] = _safe_ratio(
        predicted - request.min_fidelity, max(1.0 - request.min_fidelity, 1e-9))
    features[ActionFeature.ON_REFERENCE_PATH] = path_relevance
    features[ActionFeature.PROGRESS_GAIN] = float(np.clip(progress_gain, -1.0, 1.0))
    return features
