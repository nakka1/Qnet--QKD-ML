"""The environment: an entanglement-distribution episode an agent can act in.

One episode is one request — deliver ``pairs_needed`` end-to-end pairs between
two nodes, each at or above a fidelity threshold, before a deadline.  The agent
issues one primitive per step (:mod:`.actions`) and the clock advances by that
primitive's latency.

What makes this a control problem rather than a routing puzzle
--------------------------------------------------------------
Three couplings, all of them physical, none of them separable:

1. **Every operation costs time, and time costs fidelity.**  Purifying a pair
   improves it, but the classical signalling the purification needs decoheres
   every *other* pair in the network while it runs.  On a long path that cost
   can exceed the gain, and where the crossover sits depends on the state of
   the whole network, not just the pair being purified.
2. **Memory is finite and shared.**  Generating aggressively on one link fills
   slots that a swap elsewhere needs.  The binding constraint moves as the
   episode runs.
3. **Both purification and swapping can fail and take their inputs with them.**
   Purification returns one pair from two only on success; a Bell measurement
   that misfires loses both. So every quality-improving move is a bet against
   the clock.

An agent that treats routing and repeater timing as separate problems cannot
see any of this, which is the reason they are one action space here.

Reward
------
Sparse and honest: reward arrives on delivery, scaled by how far the delivered
fidelity clears the threshold, minus a cost proportional to elapsed time.
Nothing is paid for looking busy.

Optional **potential-based shaping** (``shaping_weight``) adds
:math:`\\gamma\\Phi(s') - \\Phi(s)` where :math:`\\Phi` measures how much of the
source-to-destination path the best current pair spans.  Potential-based
shaping provably leaves the optimal policy unchanged (Ng, Harada & Russell
1999), which is why it is safe to train with — and evaluation always reports
the *unshaped* outcome, so a shaped agent and a heuristic are compared on the
same terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ..physics import bell_diagonal as bds
from ..physics import constants as C
from ..physics.fiber import propagation_delay_s
from ..physics.link import attempt_success_probability
from ..physics.memory import MemoryParameters
from .actions import Action, ActionKind, legal_actions
from .state import EntangledPair, NetworkState, Request

__all__ = ["EnvironmentConfig", "QuantumNetworkEnv", "StepResult"]


@dataclass
class EnvironmentConfig:
    """Topology, hardware and the rules of the episode.

    Attributes:
        graph: the network, with ``length`` (km) on every edge.
        memory: memory hardware, including the cutoff policy.
        gate_fid, meas_fid: local operation quality, feeding purification and
            swapping.  These are the numbers SeQUeNCe silently drops (see the
            README); here they matter and are applied.
        purification: ``"dejmps"`` or ``"bbpssw"``.  DEJMPS keeps the full
            Bell-diagonal structure and generally wins; BBPSSW twirls first.
        swap_success_prob: probability the Bell measurement at a repeater
            succeeds.  1.0 for deterministic gates on matter qubits; 0.5 for a
            linear-optics measurement.
        twirl_on_swap: twirl inputs and output of a swap into Werner form.
        scheme: heralding scheme for elementary link generation.
        coupling_efficiency, detector_efficiency: link budget inputs.
        generate_budget_s: how long one GENERATE action may keep retrying
            before giving up.  This is the macro-action's granularity: too
            small and episodes are thousands of steps of nothing, too large and
            the agent loses the ability to interleave.
        heralding_round_trips: one-way traversals of the span charged per
            heralding attempt. One is the flight time alone, which is what
            phases 2 to 4 were published with. SeQUeNCe's protocol costs
            :data:`~qnet_autopilot.physics.constants.MEASURED_HERALDING_ROUND_TRIPS`
            of them, because an attempt is a negotiation — the ends agree on a
            slot, emit, and hear back from the measurement node — and the
            measured factor is constant across span lengths. Left at one by
            default so the published numbers stay reproducible; set it to the
            measured value to charge what the protocol actually costs.
        wait_slot_s: how far WAIT advances the clock.
        max_pairs_per_link: cap on simultaneous pairs on one edge.  Without it
            an agent can fill every memory with elementary pairs on one link
            and deadlock.
        allow_discard: offer DISCARD, which releases a pair's memory at once.
            On by default: without it a full memory bank can only be cleared by
            waiting out the cutoff, which is a modelling artefact rather than a
            physical constraint.
        max_hops: refuse swaps that would build a pair spanning more links than
            this.
        max_steps: episode step cap, independent of the deadline.
        delivery_reward: paid once per qualifying delivered pair.
        fidelity_bonus: extra reward for clearing the threshold by a margin,
            scaled so delivering at fidelity 1.0 pays ``fidelity_bonus`` more.
        time_cost: subtracted over the episode in proportion to elapsed time,
            normalised by the deadline — so a full-length episode costs
            ``time_cost`` in total regardless of the deadline chosen.
        attempt_cost: charged per GENERATE.  Link time is a real resource and
            an agent that is not charged for it will spam generation.
        shaping_weight: strength of the potential-based shaping term.  Zero
            disables it.
        gamma: discount, used by the shaping term and by the trainer.
    """

    graph: nx.Graph
    memory: MemoryParameters = field(default_factory=MemoryParameters)
    gate_fid: float = C.DEFAULT_GATE_FIDELITY
    meas_fid: float = C.DEFAULT_MEAS_FIDELITY
    purification: str = "dejmps"
    swap_success_prob: float = 1.0
    twirl_on_swap: bool = False
    scheme: str = "single_heralded"
    coupling_efficiency: float = C.DEFAULT_COUPLING_EFFICIENCY
    detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY
    memory_efficiency: float = 1.0
    generate_budget_s: float = 0.02
    heralding_round_trips: float = 1.0
    wait_slot_s: float = 0.01
    max_pairs_per_link: int = 3
    allow_discard: bool = True
    max_hops: int = 8
    max_steps: int = 120
    delivery_reward: float = 1.0
    fidelity_bonus: float = 0.5
    time_cost: float = 0.5
    attempt_cost: float = 0.01
    shaping_weight: float = 0.0
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if self.purification not in ("dejmps", "bbpssw"):
            raise ValueError(f"unknown purification protocol {self.purification!r}")
        missing = [(u, v) for u, v, data in self.graph.edges(data=True)
                   if "length" not in data]
        if missing:
            raise ValueError(f"edges without a 'length': {missing[:3]}")

        self.nodes: list[str] = [str(node) for node in self.graph.nodes()]
        self._relabel = {node: str(node) for node in self.graph.nodes()}
        self.edges: list[tuple[str, str]] = [
            (self._relabel[u], self._relabel[v]) for u, v in self.graph.edges()]

        self.length_km: dict[frozenset[str], float] = {}
        self.link_delay_s: dict[frozenset[str], float] = {}
        self.link_probability: dict[frozenset[str], float] = {}
        self.link_round_trip_s: dict[frozenset[str], float] = {}

        for u, v, data in self.graph.edges(data=True):
            key = frozenset((self._relabel[u], self._relabel[v]))
            length_km = float(data["length"])
            attenuation = float(data.get("attenuation", 2.0e-4)) * 1000.0  # dB/m -> dB/km
            self.length_km[key] = length_km
            self.link_delay_s[key] = float(propagation_delay_s(length_km))
            # One heralding round is one full traversal of the span: each node
            # sends to the midpoint and the herald comes back.
            self.link_round_trip_s[key] = (self.link_delay_s[key]
                                           * self.heralding_round_trips)
            self.link_probability[key] = float(attempt_success_probability(
                length_km, self.scheme, attenuation,
                self.coupling_efficiency, self.detector_efficiency,
                self.memory_efficiency))

        self.max_length_km = max(self.length_km.values())
        self.hop_index: dict[str, dict[str, int]] = {}

    def key(self, node_a: str, node_b: str) -> frozenset[str]:
        return frozenset((node_a, node_b))

    def distance_km(self, node_a: str, node_b: str) -> float:
        return self.length_km[self.key(node_a, node_b)]

    def signalling_delay_s(self, distance_km: float) -> float:
        """One-way classical delay over a given fibre distance."""
        return float(propagation_delay_s(max(distance_km, 0.0)))


@dataclass
class StepResult:
    """What one step produced.

    Attributes:
        reward: shaped reward, what the trainer optimises.
        raw_reward: unshaped reward, what evaluation reports.
        terminated: the request was fulfilled.
        truncated: the deadline or the step cap was hit.
        info: diagnostics — the action taken, its latency, whether a stochastic
            operation succeeded, and the running state summary.
    """

    reward: float
    raw_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class QuantumNetworkEnv:
    """Entanglement distribution as a sequential decision problem.

    The API mirrors Gymnasium's (``reset`` / ``step`` returning observation,
    reward, terminated, truncated, info) without importing it — the package has
    no RL-framework dependency, and the observation here is a
    ``(state_features, action_features)`` pair rather than a fixed array,
    because the action set changes shape with the state.
    """

    def __init__(self, config: EnvironmentConfig, rng: np.random.Generator | int | None = None):
        self.config = config
        self.rng = np.random.default_rng(rng)
        self.state = NetworkState(memory_size=10)
        self.request: Request | None = None
        self.steps = 0
        self._shortest_path: list[str] = []
        self._path_position: dict[str, int] = {}
        # A primitive that creates a pair records it here rather than storing
        # it immediately: the pair only exists once the operation's latency has
        # elapsed, and the clock has not moved yet when the primitive runs.
        self._pending_pair: tuple[str, str, float] | None = None
        self._pending_merge: EntangledPair | None = None

    # -- episode lifecycle --------------------------------------------------

    def reset(self, request: Request | None = None,
              memory_size: int = 10,
              rng: np.random.Generator | int | None = None) -> tuple[Any, dict]:
        """Start a new episode.

        Args:
            request: what to deliver.  ``None`` draws a random source and
                destination from the topology.
            memory_size: memory slots per node.
            rng: reseed the environment's generator.

        Returns:
            ``(observation, info)``.
        """
        if rng is not None:
            self.rng = np.random.default_rng(rng)

        if request is None:
            source, destination = self.rng.choice(
                len(self.config.nodes), size=2, replace=False)
            request = Request(source=self.config.nodes[source],
                              destination=self.config.nodes[destination])
        self.request = request

        self.state = NetworkState(memory_size=memory_size)
        self.steps = 0
        self._shortest_path = self._compute_shortest_path(request)
        self._path_position = {node: index
                               for index, node in enumerate(self._shortest_path)}

        return self.observe(), {"request": request,
                                "shortest_path": list(self._shortest_path)}

    def _compute_shortest_path(self, request: Request) -> list[str]:
        """Fewest-kilometres path, used for the shaping potential only.

        Not a routing decision — the agent is free to build a pair along any
        path at all.  This only defines "how far along" a pair is for shaping.
        """
        relabelled = nx.relabel_nodes(self.config.graph, self.config._relabel, copy=True)
        try:
            return nx.shortest_path(relabelled, request.source, request.destination,
                                    weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [request.source, request.destination]

    # -- observation --------------------------------------------------------

    def legal_actions(self) -> list[Action]:
        return legal_actions(self.state, self.request, self.config)

    def observe(self) -> tuple[NDArray[np.float64], NDArray[np.float64], list[Action]]:
        """``(state_features, action_features, actions)`` for the current state."""
        from .features import encode_actions, encode_state

        actions = self.legal_actions()
        return (encode_state(self, self.state, self.request),
                encode_actions(self, self.state, self.request, actions),
                actions)

    # -- transition ---------------------------------------------------------

    def step(self, action: Action) -> tuple[Any, StepResult]:
        """Apply one primitive and advance the clock by its latency.

        Raises:
            ValueError: if the action is not currently legal.  This is a hard
                error rather than a no-op with a penalty: a silent no-op lets a
                buggy policy look like it is exploring when it is stalling.
        """
        if self.request is None:
            raise RuntimeError("call reset() before step()")

        potential_before = self._potential()
        handler = {
            ActionKind.WAIT: self._do_wait,
            ActionKind.GENERATE: self._do_generate,
            ActionKind.PURIFY: self._do_purify,
            ActionKind.SWAP: self._do_swap,
            ActionKind.DELIVER: self._do_deliver,
            ActionKind.DISCARD: self._do_discard,
        }[action.kind]

        duration_s, raw_reward, info = handler(action)
        expired = self.state.advance(duration_s, self.config.memory)
        expired += self._install_pending()

        self.steps += 1
        raw_reward -= self.config.time_cost * duration_s / self.request.deadline_s

        potential_after = self._potential()
        shaped = raw_reward + self.config.shaping_weight * (
            self.config.gamma * potential_after - potential_before)

        terminated = len(self.state.delivered) >= self.request.pairs_needed
        truncated = (not terminated
                     and (self.state.now_s >= self.request.deadline_s
                          or self.steps >= self.config.max_steps))

        info.update({
            "action": str(action),
            "duration_s": duration_s,
            "expired": expired,
            "now_s": self.state.now_s,
            "steps": self.steps,
        })

        return self.observe(), StepResult(reward=shaped, raw_reward=raw_reward,
                                          terminated=terminated, truncated=truncated,
                                          info=info)

    # -- primitives ---------------------------------------------------------

    def _do_wait(self, action: Action) -> tuple[float, float, dict]:
        return self.config.wait_slot_s, 0.0, {"outcome": "waited"}

    def _do_generate(self, action: Action) -> tuple[float, float, dict]:
        """Retry heralded generation on one link until success or budget.

        The number of rounds to the first success is geometric; truncating it
        at the budget gives the honest thing a link controller does — keep
        trying for a bounded time, then report back either way.  Sampling the
        actual number of rounds (rather than charging the mean) is what makes
        the agent face variance in link latency, which is most of what makes
        routing over a lossy network hard.
        """
        node_a, node_b = action.link
        key = self.config.key(node_a, node_b)
        probability = self.config.link_probability[key]
        round_trip = self.config.link_round_trip_s[key]

        max_rounds = max(int(self.config.generate_budget_s // round_trip), 1)
        self.state.attempts += 1

        if probability <= 0.0:
            return max_rounds * round_trip, -self.config.attempt_cost, {
                "outcome": "generate_failed", "rounds": max_rounds}

        success_probability = 1.0 - (1.0 - probability) ** max_rounds
        if self.rng.random() >= success_probability:
            return (max_rounds * round_trip, -self.config.attempt_cost,
                    {"outcome": "generate_failed", "rounds": max_rounds})

        # Rounds to first success, conditioned on succeeding within the budget.
        uniform = self.rng.random()
        rounds = int(np.ceil(
            np.log1p(-uniform * success_probability) / np.log1p(-probability)))
        rounds = int(np.clip(rounds, 1, max_rounds))
        duration_s = rounds * round_trip

        # The pair is heralded at the end of the elapsed time, so create it
        # after the clock moves. Record it here and let step() place it.
        self._pending_pair = (node_a, node_b, self.config.length_km[key])
        return duration_s, -self.config.attempt_cost, {
            "outcome": "generate_succeeded", "rounds": rounds}

    def _do_purify(self, action: Action) -> tuple[float, float, dict]:
        kept_id, consumed_id = action.pair_ids
        kept = self.state.pairs[kept_id]
        consumed = self.state.pairs[consumed_id]
        now = self.state.now_s

        kept_state = kept.state_at(now, self.config.memory)
        consumed_state = consumed.state_at(now, self.config.memory)

        protocol = bds.dejmps if self.config.purification == "dejmps" else bds.bbpssw
        p_success, purified = protocol(kept_state, consumed_state,
                                       gate_fid=self.config.gate_fid,
                                       meas_fid=self.config.meas_fid)
        p_success = float(np.clip(p_success, 0.0, 1.0))

        # Both nodes measure and exchange results; they learn the outcome one
        # one-way delay later, and everything decoheres meanwhile.
        latency_s = self.config.signalling_delay_s(kept.distance_km)
        self.state.purifications += 1

        self.state.remove_pair(consumed_id)
        if self.rng.random() < p_success:
            kept.state = np.asarray(purified, dtype=float)
            kept.last_update_s = now
            kept.purifications += 1
            outcome = "purify_succeeded"
        else:
            self.state.remove_pair(kept_id)
            outcome = "purify_failed"

        return latency_s, 0.0, {"outcome": outcome, "p_success": p_success}

    def _do_swap(self, action: Action) -> tuple[float, float, dict]:
        left_id, right_id = action.pair_ids
        left = self.state.pairs[left_id]
        right = self.state.pairs[right_id]
        node = action.node
        now = self.state.now_s

        far_left = left.other_end(node)
        far_right = right.other_end(node)

        left_state = left.state_at(now, self.config.memory)
        right_state = right.state_at(now, self.config.memory)
        swapped = bds.swap(left_state, right_state,
                           gate_fid=self.config.gate_fid,
                           meas_fid=self.config.meas_fid,
                           is_twirled=self.config.twirl_on_swap)

        distance_km = left.distance_km + right.distance_km
        # The middle node must tell both ends what correction to apply; the
        # pair is usable once the farther of the two has heard.
        latency_s = max(self.config.signalling_delay_s(left.distance_km),
                        self.config.signalling_delay_s(right.distance_km))

        self.state.swaps += 1
        self.state.remove_pair(left_id)
        self.state.remove_pair(right_id)

        if self.rng.random() >= self.config.swap_success_prob:
            return latency_s, 0.0, {"outcome": "swap_failed"}

        merged = EntangledPair.new(far_left, far_right, swapped, now,
                                   hops=left.hops + right.hops,
                                   distance_km=distance_km)
        # The merged pair is only as fresh as its oldest constituent: the
        # surviving qubits at the far ends have been sitting in memory since
        # their own links were heralded, and the cutoff applies to them.
        merged.created_s = min(left.created_s, right.created_s)
        self._pending_merge = merged
        return latency_s, 0.0, {"outcome": "swap_succeeded"}

    def _do_deliver(self, action: Action) -> tuple[float, float, dict]:
        (pair_id,) = action.pair_ids
        pair = self.state.pairs[pair_id]
        now = self.state.now_s
        fidelity = pair.fidelity_at(now, self.config.memory)

        self.state.remove_pair(pair_id)
        pair.state = pair.state_at(now, self.config.memory)
        pair.last_update_s = now
        self.state.delivered.append(pair)

        threshold = self.request.min_fidelity
        margin = (fidelity - threshold) / max(1.0 - threshold, 1e-9)
        reward = self.config.delivery_reward + self.config.fidelity_bonus * margin

        return 0.0, reward, {"outcome": "delivered", "fidelity": fidelity,
                             "hops": pair.hops, "age_s": pair.age_s(now)}

    def _do_discard(self, action: Action) -> tuple[float, float, dict]:
        """Release a pair's memory immediately.

        Free in time and in reward: the cost is the opportunity cost of the
        entanglement thrown away, which the shaping potential and the eventual
        outcome already charge for. Adding a penalty on top would double-count
        and would teach the agent to hoard.
        """
        (pair_id,) = action.pair_ids
        self.state.remove_pair(pair_id)
        self.state.discarded += 1
        return 0.0, 0.0, {"outcome": "discarded"}

    # -- shaping ------------------------------------------------------------

    def _potential(self) -> float:
        """How far along the reference path the best current pair reaches.

        Zero when nothing useful is held, one when an end-to-end pair exists.
        Only pairs whose *both* ends sit on the reference path count, which
        keeps the potential cheap and monotone.
        """
        if not self._path_position or len(self._shortest_path) < 2:
            return 0.0

        span = len(self._shortest_path) - 1
        best = 0
        for pair in self.state.pairs.values():
            left = self._path_position.get(pair.node_a)
            right = self._path_position.get(pair.node_b)
            if left is None or right is None:
                continue
            best = max(best, abs(right - left))
        return best / span

    # -- bookkeeping between the primitive and the clock --------------------

    def _install_pending(self) -> int:
        """Store pairs the last primitive produced, now that the clock has moved.

        A heralded pair exists only at the end of the round that produced it,
        and a swapped pair only once the correction has reached both ends.
        Creating them before :meth:`NetworkState.advance` would decohere them
        for time they did not exist, and would expose them to a cutoff test
        they should not face yet.

        Returns:
            How many produced pairs were dropped on arrival — a swap whose
            inputs were already so old that the merged pair is past the cutoff
            the moment it exists. That is a real outcome and worth counting.
        """
        dropped = 0

        if self._pending_pair is not None:
            node_a, node_b, length_km = self._pending_pair
            self._pending_pair = None
            if self.state.can_hold(node_a, node_b):
                self.state.add_pair(EntangledPair.new(
                    node_a, node_b, self.config.memory.fresh_state(),
                    self.state.now_s, hops=1, distance_km=length_km))
            else:
                dropped += 1

        if self._pending_merge is not None:
            merged = self._pending_merge
            self._pending_merge = None
            merged.advance_to(self.state.now_s, self.config.memory)
            if merged.age_s(self.state.now_s) > self.config.memory.cutoff_time_s:
                dropped += 1
            elif self.state.can_hold(*merged.nodes):
                self.state.add_pair(merged)
            else:
                dropped += 1

        self.state.discarded += dropped
        return dropped
