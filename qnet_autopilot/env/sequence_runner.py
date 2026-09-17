"""The phase-2 action space, executed inside SeQUeNCe's discrete-event loop.

What this is for
----------------
:mod:`qnet_autopilot.validation.twin` established that the analytic environment's
*state arithmetic* is SeQUeNCe's, step for step. It said nothing about the rest
of the simulator: photons that may or may not arrive, detectors that click or
do not, classical messages with real delays, protocols that negotiate before
they act, and memories that are occupied while all of that happens. Phase 2's
limits section, and every gap review since, named that as the piece still
missing.

This module closes it. It builds a real SeQUeNCe network — quantum routers with
memory arrays, single-heralded Bell-state-measurement nodes, quantum channels
with attenuation, classical channels with delay — and executes the same six
primitives the agent chooses between, by instantiating SeQUeNCe's own protocols
and letting the event loop resolve them. Nothing here is analytic: a GENERATE
succeeds when a detector clicks.

How the agent gets to make decisions inside an event loop
---------------------------------------------------------
``Timeline.run`` returns when the event queue empties or when it reaches
``stop_time``, and in the latter case it pushes the event it stopped on back
onto the queue. So it is resumable, and the control loop is stop-and-go: start
one protocol, raise the horizon, run until it resolves, read the outcome, ask
the policy for the next action. The simulator's clock is the only clock; this
module never advances time itself.

What it does not do
-------------------
It does not use SeQUeNCe's rule engine or its reservation protocol. Those are
SeQUeNCe's own policy — a rule set that reacts to memory states — and the whole
point here is to put a *learned* policy in that position instead. Routing
tables, network managers and applications are therefore all unused, and this
module drives protocols directly, the way the rule actions in SeQUeNCe's own
examples do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .actions import Action, ActionKind
from .network_env import EnvironmentConfig
from .state import Request

__all__ = ["SequencePair", "SequenceOutcome", "SequenceNetwork",
           "run_sequence_episode"]

#: Picoseconds per second. SeQUeNCe's clock is an integer number of
#: picoseconds; everything this package says out loud is in seconds.
PS_PER_S = 1e12


@dataclass
class SequencePair:
    """One entangled pair, as SeQUeNCe's memory managers describe it.

    Attributes:
        pair_id: assigned by this runner, stable while the pair lives.
        node_a, node_b: the two routers holding it.
        memory_a, memory_b: the memory objects, one at each end.
        created_s: when the pair was heralded, from the simulator's clock.
        hops: how many elementary links it spans.
        distance_km: total fibre it spans, which sets signalling latency.
        purifications: how many rounds went into it.
    """

    pair_id: int
    node_a: str
    node_b: str
    memory_a: Any
    memory_b: Any
    created_s: float
    hops: int = 1
    distance_km: float = 0.0
    purifications: int = 0

    @property
    def nodes(self) -> tuple[str, str]:
        return (self.node_a, self.node_b)

    def other_end(self, node: str) -> str:
        if node == self.node_a:
            return self.node_b
        if node == self.node_b:
            return self.node_a
        raise ValueError(f"{node!r} does not hold pair {self.pair_id}")

    def state(self, network: "SequenceNetwork") -> np.ndarray:
        """The Bell-diagonal state SeQUeNCe holds for this pair, brought to now.

        ``Memory.bds_decohere`` is what ages a stored pair in SeQUeNCe, and it
        runs only when a protocol touches the memory. Reading the state without
        calling it first reports the pair as it was when it was last acted on,
        which on an idle pair is a fidelity it no longer has.
        """
        self.memory_a.bds_decohere()
        self.memory_b.bds_decohere()
        return np.asarray(network.timeline.quantum_manager.get(
            self.memory_a.qstate_key).state, dtype=float)

    def fidelity(self, network: "SequenceNetwork") -> float:
        return float(self.state(network)[0])


@dataclass
class SequenceOutcome:
    """What one primitive did, once the event loop had finished with it.

    Attributes:
        outcome: the same vocabulary the analytic environment uses, so the two
            can be compared without a translation table.
        duration_s: elapsed simulator time, which is the whole point: it is
            produced by photon flight, detector dead time and message delay
            rather than by a formula.
        info: extra diagnostics — attempts made, fidelity delivered.
    """

    outcome: str
    duration_s: float
    info: dict[str, Any] = field(default_factory=dict)


class SequenceNetwork:
    """A chain of SeQUeNCe routers the phase-2 primitives can act on.

    Args:
        config: the same :class:`~qnet_autopilot.env.network_env.EnvironmentConfig`
            the analytic environment uses, so a comparison between the two is a
            comparison of simulators and not of parameters.
        memory_size: memories per router.
        seed: seeds every node's generator.
        classical_delay_s: one-way delay on the classical channels. ``None``
            derives it from each span's length, which is what a real network
            has and what makes purification expensive on a long pair.

    Note:
        The Bell-diagonal formalism cannot represent a single photon, so
        Barrett-Kok generation — which interferes two of them — is unavailable
        under it and raises inside the BSM. Single-heralded generation is the
        one that works, and it happens to be the scheme phase 1 validated the
        heralding probability of. The global protocol type has to be set before
        any node is constructed, because ``BSMNode`` builds its own generation
        protocol in its constructor and reads the global there.
    """

    def __init__(self, config: EnvironmentConfig, memory_size: int = 8,
                 seed: int = 0, classical_delay_s: float | None = None):
        from ..compat import bootstrap

        bootstrap()

        from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM, SINGLE_HERALDED
        from sequence.components.optical_channel import (ClassicalChannel,
                                                         QuantumChannel)
        from sequence.entanglement_management.generation import (
            EntanglementGenerationA, EntanglementGenerationB)
        from sequence.kernel.quantum_manager import QuantumManager
        from sequence.kernel.timeline import Timeline
        from sequence.topology.node import BSMNode, QuantumRouter

        self.config = config
        self.memory_size = memory_size

        self._previous_formalism = QuantumManager.get_active_formalism()
        QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
        EntanglementGenerationA.set_global_type(SINGLE_HERALDED)
        EntanglementGenerationB.set_global_type(SINGLE_HERALDED)

        self.timeline = Timeline(formalism=BELL_DIAGONAL_STATE_FORMALISM)
        self.timeline.show_progress = False

        memory = config.memory
        self.routers: dict[str, Any] = {}
        for index, name in enumerate(config.nodes):
            router = QuantumRouter(name, self.timeline, memory_size,
                                   gate_fid=config.gate_fid,
                                   meas_fid=config.meas_fid)
            router.set_seed(seed * 1000 + index)
            array = router.get_components_by_type("MemoryArray")[0]
            array.update_memory_params("coherence_time", memory.coherence_time_s)
            # A second instance of the same upstream trap the README documents
            # for gate fidelity: ``update_memory_params`` is a bare setattr, and
            # ``decoherence_rate`` is derived from ``coherence_time`` once, in
            # the constructor. Setting the coherence time after construction
            # therefore changes nothing — the rate stays at whatever the default
            # coherence time implied, which for a stock memory array is zero.
            # A stored pair then never decoheres, every fidelity in the run is
            # the raw one, and nothing anywhere reports an error.
            array.update_memory_params(
                "decoherence_rate", 1.0 / memory.coherence_time_s
                if memory.coherence_time_s > 0 else 0.0)
            array.update_memory_params("raw_fidelity", memory.raw_fidelity)
            array.update_memory_params("decoherence_errors",
                                       list(memory.decoherence_errors))
            # The analytic link budget multiplies coupling, detector and memory
            # efficiency into one arm efficiency. SeQUeNCe applies them in
            # three different places, so they have to be routed to the right
            # ones or the two simulators are running different hardware: the
            # emission side goes on the memory, the detection side on the BSM's
            # detectors below. Getting this wrong makes generation 1.3x easier
            # in one simulator than the other, which looks exactly like a
            # finding about discrete-event simulation and is not one.
            # Coupling is squared because the analytic arm efficiency applies
            # it at both fibre interfaces; the emission side of the arm is
            # therefore coupling^2 x memory, and the detection side is the
            # detector efficiency set on the BSM below.
            array.update_memory_params(
                "efficiency",
                config.coupling_efficiency ** 2 * config.memory_efficiency)
            self.routers[name] = router

        self.bsm_nodes: dict[frozenset[str], Any] = {}
        for index, (node_a, node_b) in enumerate(config.edges):
            key = config.key(node_a, node_b)
            detector = {"efficiency": config.detector_efficiency}
            bsm = BSMNode(
                f"bsm.{node_a}.{node_b}", self.timeline, [node_a, node_b],
                component_templates={
                    "encoding_type": "single_heralded",
                    "SingleHeraldedBSM": {"detectors": [dict(detector),
                                                        dict(detector)]}})
            bsm.set_seed(seed * 1000 + 500 + index)
            self.bsm_nodes[key] = bsm
            self.routers[node_a].add_bsm_node(bsm.name, node_b)
            self.routers[node_b].add_bsm_node(bsm.name, node_a)

        # Classical delay is per pair of nodes, from the fibre between them.
        # One flat delay for the whole network would make the router-to-BSM hop
        # — half a span — cost as much as the longest span in the graph, and
        # the heralding round is three of those hops. That is not a detail: it
        # sets how long a GENERATE takes, and therefore how many the agent can
        # afford before the deadline.
        self.classical_delay_s = self._build_classical_channels(
            ClassicalChannel, classical_delay_s)

        attenuations = {
            config.key(config._relabel[u], config._relabel[v]):
                float(data.get("attenuation", 2.0e-4))
            for u, v, data in config.graph.edges(data=True)}

        for node_a, node_b in config.edges:
            key = config.key(node_a, node_b)
            bsm = self.bsm_nodes[key]
            half_metres = config.length_km[key] * 1e3 / 2.0
            # dB/m, matching what the analytic link budget was given for this
            # edge. Read through the config's own relabelling, because the
            # graph may be keyed on integers while the config is on strings.
            attenuation = attenuations.get(key, 2.0e-4)
            for end in (node_a, node_b):
                quantum = QuantumChannel(f"qc.{end}.{bsm.name}", self.timeline,
                                         attenuation, half_metres)
                quantum.set_ends(self.routers[end], bsm.name)

        self.timeline.init()

        self.pairs: dict[int, SequencePair] = {}
        self.delivered: list[dict[str, Any]] = []
        self.discarded = 0
        self.attempts = 0
        self.purifications = 0
        self.swaps = 0
        self._next_pair_id = 0
        self._protocol_counter = 0

    def _build_classical_channels(self, ClassicalChannel,
                                  fixed_delay_s: float | None) -> float:
        """Wire every pair of nodes with a delay set by the fibre between them.

        Returns:
            The median one-way delay, used as the scale for the per-attempt
            event-loop horizon.
        """
        import networkx as nx

        config = self.config
        relabelled = nx.relabel_nodes(config.graph, config._relabel, copy=True)
        lengths = dict(nx.all_pairs_dijkstra_path_length(relabelled,
                                                         weight="length"))

        positions: dict[str, tuple[str, float]] = {
            name: (name, 0.0) for name in self.routers}
        for (node_a, node_b), bsm in (
                (tuple(sorted(key)), bsm) for key, bsm in self.bsm_nodes.items()):
            # A BSM sits at the midpoint of its span, so its distance to any
            # node is that node's distance to the nearer end plus half a span.
            positions[bsm.name] = (node_a, config.length_km[
                config.key(node_a, node_b)] / 2.0)

        every = list(self.routers.values()) + list(self.bsm_nodes.values())
        delays = []
        for source in every:
            anchor_a, offset_a = positions[source.name]
            for destination in every:
                if source is destination:
                    continue
                anchor_b, offset_b = positions[destination.name]
                if fixed_delay_s is not None:
                    delay_s = fixed_delay_s
                else:
                    span_km = lengths[anchor_a][anchor_b] + offset_a + offset_b
                    delay_s = config.signalling_delay_s(span_km)
                delays.append(delay_s)
                channel = ClassicalChannel(f"cc.{source.name}.{destination.name}",
                                           self.timeline, 1e3,
                                           delay=delay_s * PS_PER_S)
                channel.set_ends(source, destination.name)

        return float(np.median(delays)) if delays else 0.0

    # -- clock --------------------------------------------------------------

    @property
    def now_s(self) -> float:
        return self.timeline.now() / PS_PER_S

    def _run_until_quiet(self, horizon_s: float) -> None:
        """Let the event loop run, at most ``horizon_s`` further.

        ``Timeline.run`` returns either when the queue empties — every protocol
        this action started has finished — or when it reaches the horizon, in
        which case it puts the event it stopped on back. Both are resumable,
        which is what makes a decision loop possible at all.
        """
        self.timeline.stop_time = self.timeline.now() + int(horizon_s * PS_PER_S)
        self.timeline.run()

    def _name(self, prefix: str) -> str:
        self._protocol_counter += 1
        return f"{prefix}.{self._protocol_counter}"

    # -- memory bookkeeping -------------------------------------------------

    def _free_memories(self, node: str) -> list[Any]:
        """Memories on ``node`` that no live pair of ours is using."""
        held = {id(pair.memory_a) for pair in self.pairs.values()}
        held |= {id(pair.memory_b) for pair in self.pairs.values()}
        array = self.routers[node].get_components_by_type("MemoryArray")[0]
        return [memory for memory in array if id(memory) not in held]

    def free_slots(self, node: str) -> int:
        return len(self._free_memories(node))

    def can_hold(self, node_a: str, node_b: str) -> bool:
        return self.free_slots(node_a) >= 1 and self.free_slots(node_b) >= 1

    def pairs_on(self, node_a: str, node_b: str) -> list[SequencePair]:
        wanted = {node_a, node_b}
        return [pair for pair in self.pairs.values() if set(pair.nodes) == wanted]

    def pairs_at(self, node: str) -> list[SequencePair]:
        return [pair for pair in self.pairs.values() if node in pair.nodes]

    def _release(self, pair: SequencePair, state: str = "RAW") -> None:
        """Hand both memories back to their resource managers."""
        for node, memory in ((pair.node_a, pair.memory_a),
                             (pair.node_b, pair.memory_b)):
            manager = self.routers[node].resource_manager
            manager.update(None, memory, state)
        self.pairs.pop(pair.pair_id, None)

    def _add_pair(self, node_a, memory_a, node_b, memory_b, hops, distance_km,
                  purifications=0) -> SequencePair:
        pair = SequencePair(pair_id=self._next_pair_id, node_a=node_a,
                            node_b=node_b, memory_a=memory_a, memory_b=memory_b,
                            created_s=self.now_s, hops=hops,
                            distance_km=distance_km,
                            purifications=purifications)
        self._next_pair_id += 1
        self.pairs[pair.pair_id] = pair
        return pair

    def expire(self) -> int:
        """Drop pairs past the cutoff, as the analytic environment does.

        SeQUeNCe has its own expiry machinery driven by the reservation
        protocol, which is not running here; applying the same cutoff policy by
        hand keeps the two simulators comparable on the one axis that is
        policy rather than physics.
        """
        cutoff = self.config.memory.cutoff_time_s
        stale = [pair for pair in self.pairs.values()
                 if self.now_s - pair.created_s > cutoff]
        for pair in stale:
            self._release(pair)
        self.discarded += len(stale)
        return len(stale)

    # -- the primitives -----------------------------------------------------

    def generate(self, node_a: str, node_b: str) -> SequenceOutcome:
        """Retry single-heralded generation until it works or the budget ends.

        Each attempt is a fresh protocol pair, because a failed one takes
        itself out of the node's protocol list on the way down. The budget is
        the environment's ``generate_budget_s``, so the macro-action means the
        same thing in both simulators.
        """
        from sequence.entanglement_management.generation import SingleHeraldedA

        if not self.can_hold(node_a, node_b):
            return SequenceOutcome("generate_failed", 0.0,
                                   {"reason": "no free memory", "rounds": 0})

        key = self.config.key(node_a, node_b)
        bsm = self.bsm_nodes[key]
        memory_a = self._free_memories(node_a)[0]
        memory_b = self._free_memories(node_b)[0]

        started = self.now_s
        budget = self.config.generate_budget_s
        round_trip = max(self.config.link_round_trip_s[key], 1e-9)
        self.attempts += 1

        rounds = 0
        while self.now_s - started < budget:
            rounds += 1
            protocol_a = SingleHeraldedA.create(
                self.routers[node_a], self._name(f"eg.{node_a}"), bsm.name,
                node_b, memory_a)
            protocol_b = SingleHeraldedA.create(
                self.routers[node_b], self._name(f"eg.{node_b}"), bsm.name,
                node_a, memory_b)
            self.routers[node_a].protocols.append(protocol_a)
            self.routers[node_b].protocols.append(protocol_b)
            protocol_a.set_others(protocol_b.name, node_b, [memory_b.name])
            protocol_b.set_others(protocol_a.name, node_a, [memory_a.name])
            protocol_a.start()
            protocol_b.start()

            # A horizon of one attempt, not one budget. ``Timeline.run`` drains
            # every event it can reach, so a horizon of the whole budget lets
            # one call swallow the idle time between rounds and the loop then
            # counts one round for thirty times the elapsed clock. One
            # negotiation is three classical hops; ten is generous and bounded.
            self._run_until_quiet(max(10 * self.classical_delay_s,
                                      4 * round_trip))

            info = self.routers[node_a].resource_manager.memory_manager \
                .get_info_by_memory(memory_a)
            if info.state in ("ENTANGLED", "PURIFIED"):
                pair = self._add_pair(node_a, memory_a, node_b, memory_b,
                                      hops=1,
                                      distance_km=self.config.length_km[key])
                return SequenceOutcome(
                    "generate_succeeded", self.now_s - started,
                    {"rounds": rounds, "pair_id": pair.pair_id,
                     "fidelity": pair.fidelity(self)})

        return SequenceOutcome("generate_failed", self.now_s - started,
                               {"rounds": rounds})

    def purify(self, kept_id: int, consumed_id: int) -> SequenceOutcome:
        """One purification round, run by SeQUeNCe's own protocol objects."""
        from sequence.entanglement_management.purification import (BBPSSW_BDS,
                                                                   DEJMPS_BDS)

        kept = self.pairs[kept_id]
        consumed = self.pairs[consumed_id]
        if set(kept.nodes) != set(consumed.nodes):
            raise ValueError("purification needs two pairs between the same nodes")

        protocol_class = DEJMPS_BDS if self.config.purification == "dejmps" else BBPSSW_BDS
        started = self.now_s
        self.purifications += 1

        # Both ends run the protocol; whichever memory of the consumed pair
        # sits on a given node is that node's measured memory.
        left, right = kept.node_a, kept.node_b
        kept_left, kept_right = kept.memory_a, kept.memory_b
        consumed_left = (consumed.memory_a if consumed.node_a == left
                         else consumed.memory_b)
        consumed_right = (consumed.memory_b if consumed.node_b == right
                          else consumed.memory_a)

        protocol_left = protocol_class(self.routers[left], self._name(f"ep.{left}"),
                                       kept_left, consumed_left)
        protocol_right = protocol_class(self.routers[right],
                                        self._name(f"ep.{right}"),
                                        kept_right, consumed_right)
        self.routers[left].protocols.append(protocol_left)
        self.routers[right].protocols.append(protocol_right)
        protocol_left.set_others(protocol_right.name, right,
                                 [kept_right.name, consumed_right.name])
        protocol_right.set_others(protocol_left.name, left,
                                  [kept_left.name, consumed_left.name])
        protocol_left.start()
        protocol_right.start()

        latency = self.config.signalling_delay_s(kept.distance_km)
        self._run_until_quiet(max(10 * latency, 1e-3))

        info = self.routers[left].resource_manager.memory_manager \
            .get_info_by_memory(kept_left)
        self.pairs.pop(consumed_id, None)

        # SeQUeNCe moves a successfully purified memory to PURIFIED, not back
        # to ENTANGLED. Checking only for ENTANGLED reports every successful
        # purification as a failure, which is a silent and very believable bug:
        # the fidelities still look right, only the success rate is wrong.
        if info.state in ("ENTANGLED", "PURIFIED"):
            kept.purifications += 1
            return SequenceOutcome("purify_succeeded", self.now_s - started,
                                   {"fidelity": kept.fidelity(self)})

        self.pairs.pop(kept_id, None)
        return SequenceOutcome("purify_failed", self.now_s - started, {})

    def swap(self, node: str, left_id: int, right_id: int) -> SequenceOutcome:
        """A Bell measurement at ``node``, with corrections at both far ends."""
        from sequence.entanglement_management.swapping import (
            EntanglementSwappingA_BDS, EntanglementSwappingB_BDS)

        left = self.pairs[left_id]
        right = self.pairs[right_id]
        far_left = left.other_end(node)
        far_right = right.other_end(node)
        if far_left == far_right:
            raise ValueError("a swap between the same two nodes is not a pair")

        middle_left = left.memory_a if left.node_a == node else left.memory_b
        middle_right = right.memory_a if right.node_a == node else right.memory_b
        end_left = left.memory_b if left.node_a == node else left.memory_a
        end_right = right.memory_b if right.node_a == node else right.memory_a

        started = self.now_s
        self.swaps += 1

        middle = EntanglementSwappingA_BDS(
            self.routers[node], self._name(f"es.{node}"), middle_left,
            middle_right, success_prob=self.config.swap_success_prob,
            is_twirled=self.config.twirl_on_swap)
        hold_left = EntanglementSwappingB_BDS(
            self.routers[far_left], self._name(f"es.{far_left}"), end_left)
        hold_right = EntanglementSwappingB_BDS(
            self.routers[far_right], self._name(f"es.{far_right}"), end_right)

        self.routers[node].protocols.append(middle)
        self.routers[far_left].protocols.append(hold_left)
        self.routers[far_right].protocols.append(hold_right)

        middle.set_others(hold_left.name, far_left, [end_left.name])
        middle.set_others(hold_right.name, far_right, [end_right.name])
        hold_left.set_others(middle.name, node, [middle_left.name])
        hold_right.set_others(middle.name, node, [middle_right.name])

        middle.start()

        latency = self.config.signalling_delay_s(
            max(left.distance_km, right.distance_km))
        self._run_until_quiet(max(10 * latency, 1e-3))

        info = self.routers[far_left].resource_manager.memory_manager \
            .get_info_by_memory(end_left)

        created = min(left.created_s, right.created_s)
        distance = left.distance_km + right.distance_km
        hops = left.hops + right.hops
        self.pairs.pop(left_id, None)
        self.pairs.pop(right_id, None)
        for holder, memory in ((node, middle_left), (node, middle_right)):
            self.routers[holder].resource_manager.update(None, memory, "RAW")

        if info.state not in ("ENTANGLED", "PURIFIED"):
            for holder, memory in ((far_left, end_left), (far_right, end_right)):
                self.routers[holder].resource_manager.update(None, memory, "RAW")
            return SequenceOutcome("swap_failed", self.now_s - started, {})

        merged = self._add_pair(far_left, end_left, far_right, end_right,
                                hops=hops, distance_km=distance)
        merged.created_s = created
        return SequenceOutcome("swap_succeeded", self.now_s - started,
                               {"pair_id": merged.pair_id,
                                "fidelity": merged.fidelity(self)})

    def deliver(self, pair_id: int) -> SequenceOutcome:
        """Hand a pair to the application and free its memories."""
        pair = self.pairs[pair_id]
        record = {"fidelity": pair.fidelity(self), "hops": pair.hops,
                  "age_s": self.now_s - pair.created_s,
                  "purifications": pair.purifications}
        self.delivered.append(record)
        self._release(pair)
        return SequenceOutcome("delivered", 0.0, record)

    def discard(self, pair_id: int) -> SequenceOutcome:
        self._release(self.pairs[pair_id])
        self.discarded += 1
        return SequenceOutcome("discarded", 0.0, {})

    def wait(self, duration_s: float | None = None) -> SequenceOutcome:
        """Advance the clock with nothing scheduled.

        The event queue is empty between actions, so the loop cannot idle by
        running it; the clock is moved directly. This is the one place the
        runner touches ``Timeline.time``, and it is doing what a real node does
        when it decides to do nothing for a slot.
        """
        duration_s = (self.config.wait_slot_s if duration_s is None
                      else duration_s)
        self.timeline.time += int(duration_s * PS_PER_S)
        return SequenceOutcome("waited", duration_s, {})

    # -- the action interface ------------------------------------------------

    def legal_actions(self, request: Request) -> list[Action]:
        """The same enumeration the analytic environment offers.

        Deliberately a re-implementation against SeQUeNCe's memory managers
        rather than a call into the analytic one: if the two disagree about
        what is legal, that is a difference worth seeing rather than hiding.
        """
        actions: list[Action] = [Action(ActionKind.WAIT)]

        for node_a, node_b in self.config.edges:
            if not self.can_hold(node_a, node_b):
                continue
            if len(self.pairs_on(node_a, node_b)) >= self.config.max_pairs_per_link:
                continue
            actions.append(Action(ActionKind.GENERATE, link=(node_a, node_b)))

        by_endpoints: dict[frozenset[str], list[int]] = {}
        for pair in self.pairs.values():
            by_endpoints.setdefault(frozenset(pair.nodes), []).append(pair.pair_id)
        for pair_ids in by_endpoints.values():
            ordered = sorted(pair_ids)
            for index, kept in enumerate(ordered):
                for consumed in ordered[index + 1:]:
                    actions.append(Action(ActionKind.PURIFY,
                                          pair_ids=(kept, consumed)))

        for node in self.config.nodes:
            incident = sorted(self.pairs_at(node), key=lambda pair: pair.pair_id)
            for index, left in enumerate(incident):
                for right in incident[index + 1:]:
                    if left.other_end(node) == right.other_end(node):
                        continue
                    if left.hops + right.hops > self.config.max_hops:
                        continue
                    actions.append(Action(ActionKind.SWAP, node=node,
                                          pair_ids=(left.pair_id, right.pair_id)))

        endpoints = {request.source, request.destination}
        for pair in sorted(self.pairs.values(), key=lambda pair: pair.pair_id):
            if set(pair.nodes) != endpoints:
                continue
            if pair.fidelity(self) < request.min_fidelity:
                continue
            actions.append(Action(ActionKind.DELIVER, pair_ids=(pair.pair_id,)))

        if self.config.allow_discard:
            for pair in sorted(self.pairs.values(), key=lambda pair: pair.pair_id):
                actions.append(Action(ActionKind.DISCARD, pair_ids=(pair.pair_id,)))

        return actions

    def step(self, action: Action) -> SequenceOutcome:
        """Execute one primitive and expire whatever aged out while it ran."""
        if action.kind is ActionKind.WAIT:
            result = self.wait()
        elif action.kind is ActionKind.GENERATE:
            result = self.generate(*action.link)
        elif action.kind is ActionKind.PURIFY:
            result = self.purify(*action.pair_ids)
        elif action.kind is ActionKind.SWAP:
            result = self.swap(action.node, *action.pair_ids)
        elif action.kind is ActionKind.DELIVER:
            result = self.deliver(action.pair_ids[0])
        else:
            result = self.discard(action.pair_ids[0])

        result.info["expired"] = self.expire()
        result.info["now_s"] = self.now_s
        return result

    # -- the bridge to the policy's feature encoders -------------------------

    def as_state(self, delivered_pairs: Iterable[Any] = ()) -> "NetworkState":
        """Mirror SeQUeNCe's live pairs into the analytic state object.

        The policy's inputs are computed by
        :mod:`qnet_autopilot.env.features`, which reads a
        :class:`~qnet_autopilot.env.state.NetworkState`. Rebuilding that object
        from SeQUeNCe's memory managers — rather than re-implementing the
        encoders against SeQUeNCe's own types — means the agent sees exactly the
        features it was trained on, computed from the discrete-event
        simulator's state. The encoders are shared; the state behind them is
        not.

        Every mirrored pair carries ``last_update_s = now``, because the state
        read out of SeQUeNCe has already been aged by ``bds_decohere``. Leaving
        it at the creation time would age it a second time inside the encoder.
        """
        from .state import EntangledPair, NetworkState

        state = NetworkState(now_s=self.now_s, memory_size=self.memory_size)
        state.discarded = self.discarded
        state.attempts = self.attempts
        state.purifications = self.purifications
        state.swaps = self.swaps
        state.delivered = list(delivered_pairs)

        for pair in sorted(self.pairs.values(), key=lambda item: item.pair_id):
            mirrored = EntangledPair(
                pair_id=pair.pair_id, node_a=pair.node_a, node_b=pair.node_b,
                state=pair.state(self), created_s=pair.created_s,
                last_update_s=self.now_s, hops=pair.hops,
                distance_km=pair.distance_km,
                purifications=pair.purifications)
            state.pairs[pair.pair_id] = mirrored
            for node in mirrored.nodes:
                state.memory_used[node] = state.memory_used.get(node, 0) + 1

        # Memories SeQUeNCe considers busy that this runner has not accounted
        # for would make the two disagree about what is legal. There should be
        # none; counting them rather than assuming is what makes that a fact.
        for node in self.config.nodes:
            occupied = self.memory_size - self.free_slots(node)
            state.memory_used[node] = max(state.memory_used.get(node, 0), occupied)

        state._next_pair_id = self._next_pair_id
        return state

    def summary(self) -> dict[str, float]:
        """The numbers a comparison against the analytic environment needs."""
        fidelities = [record["fidelity"] for record in self.delivered]
        return {
            "now_s": self.now_s,
            "delivered": len(self.delivered),
            "mean_delivered_fidelity": (float(np.mean(fidelities))
                                        if fidelities else float("nan")),
            "live_pairs": len(self.pairs),
            "discarded": self.discarded,
            "attempts": self.attempts,
            "purifications": self.purifications,
            "swaps": self.swaps,
        }


def run_sequence_episode(config: EnvironmentConfig, policy, request: Request,
                         seed: int = 0, memory_size: int = 6,
                         max_steps: int | None = None) -> dict[str, Any]:
    """Run one episode of the phase-2 task inside SeQUeNCe's event loop.

    The policy is the trained one, unmodified. Its inputs come from the same
    encoders it was trained with, applied to a state mirrored out of SeQUeNCe's
    memory managers, so what changes between this and
    :func:`qnet_autopilot.agents.rollout.run_episode` is the simulator
    underneath and nothing else.

    Args:
        config: topology and hardware, shared with the analytic environment.
        policy: anything satisfying :class:`~qnet_autopilot.agents.base.Policy`.
        request: what to deliver.
        seed: seeds every SeQUeNCe node, and the encoder host.
        memory_size: memories per router.
        max_steps: step cap; defaults to the config's.

    Returns:
        A record with the same fields the analytic evaluation reports, plus
        ``legality_disagreements``: how many steps the two simulators offered
        different action sets on. That number should be zero, and it is
        reported rather than asserted because a disagreement is a finding.

    Note:
        The encoder host is a real :class:`QuantumNetworkEnv` whose state is
        replaced each step by the mirror. It is never stepped, so it contributes
        no physics — only the shortest path, the shaping potential and the
        feature layout, all of which are functions of the configuration.
    """
    from .features import encode_actions, encode_state
    from .network_env import QuantumNetworkEnv

    network = SequenceNetwork(config, memory_size=memory_size, seed=seed)
    encoder = QuantumNetworkEnv(config, rng=seed)
    encoder.reset(request=request, memory_size=memory_size)
    policy.reset()

    max_steps = config.max_steps if max_steps is None else max_steps
    disagreements = 0
    steps = 0
    fidelities: list[float] = []

    while steps < max_steps and network.now_s < request.deadline_s:
        if len(network.delivered) >= request.pairs_needed:
            break

        state = network.as_state(delivered_pairs=network.delivered)
        encoder.state = state
        encoder.steps = steps

        actions = network.legal_actions(request)
        mirrored = encoder.legal_actions()
        if [str(action) for action in actions] != [str(action) for action in mirrored]:
            disagreements += 1

        observation = (encode_state(encoder, state, request),
                       encode_actions(encoder, state, request, actions),
                       actions)
        index = policy.act(observation)
        if not 0 <= index < len(actions):
            raise IndexError(
                f"{policy.name} chose action {index} of {len(actions)} legal")

        result = network.step(actions[index])
        if result.outcome == "delivered":
            fidelities.append(float(result.info["fidelity"]))
        steps += 1

    summary = network.summary()
    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "seed": seed,
        "succeeded": len(network.delivered) >= request.pairs_needed,
        "delivered": len(network.delivered),
        "mean_fidelity": float(np.mean(fidelities)) if fidelities else float("nan"),
        "elapsed_s": network.now_s,
        "steps": steps,
        "attempts": summary["attempts"],
        "purifications": summary["purifications"],
        "swaps": summary["swaps"],
        "discarded": summary["discarded"],
        "legality_disagreements": disagreements,
    }
