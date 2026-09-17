"""The phase-2 environment carried alongside a SeQUeNCe reconstruction.

Phase 1 validated the closed forms one operation at a time: purification,
swapping and memory decoherence each return what SeQUeNCe's own protocol objects
return. That says nothing about the *environment*, which is separate software
deciding which function to call, on which pair, with which hardware, after how
much elapsed time. Phase 2's limits section admitted as much — "the environment
is the fast analytic model, not the discrete-event simulator" — and this module
is what closes it.

:class:`SequenceShadow` walks an episode step by step. Before each action it
computes what SeQUeNCe would produce for that operation, from *its own* copy of
the states; after each action it ages every surviving pair by calling
``Memory.bds_decohere`` once per end of the pair, driven by the elapsed time the
environment reports. Then it compares, pair by pair. The environment is never
asked what a state *is* — only what it *did*.

What this checks and what it does not
-------------------------------------
It checks that the state arithmetic along a trajectory is SeQUeNCe's arithmetic:
the right operation, on the right pair, with the right hardware parameters, aged
for the right interval, at both ends. It does not run an agent inside SeQUeNCe's
event loop, and it does not exercise SeQUeNCe's scheduling, messaging or
resource management. Those remain out of scope, and saying so is part of the
result.

Cost
----
Every reference operation builds a fresh :class:`~sequence.kernel.timeline.Timeline`.
That is deliberate: ``Timeline.init`` re-initialises every entity registered on
it, so reusing one timeline across a trajectory silently resets memories created
earlier, and the agreement that follows would be an artefact of the reset rather
than evidence of anything. The cost is a few milliseconds per operation, which
is why this is a validation tool and not something the trainer runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..env.actions import Action, ActionKind

__all__ = ["ShadowStep", "SequenceShadow", "sequence_purify", "sequence_swap",
           "sequence_decohere"]


# ---------------------------------------------------------------------------
# SeQUeNCe reference operations
# ---------------------------------------------------------------------------

def _fresh_timeline():
    """A timeline of its own, for the reason given in the module docstring."""
    from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
    from sequence.kernel.timeline import Timeline

    timeline = Timeline(formalism=BELL_DIAGONAL_STATE_FORMALISM)
    timeline.show_progress = False
    return timeline


class _FakeResourceManager:
    """The one method the entanglement protocols call on their owner's manager."""

    def update(self, protocol, memory, state):
        if state == "RAW":
            memory.reset()


def _fake_node(name: str, timeline, gate_fid: float, meas_fid: float):
    from sequence.message import Message
    from sequence.topology.node import Node

    class FakeNode(Node):
        def __init__(self, name, tl, gate_fid, meas_fid):
            Node.__init__(self, name, tl, gate_fid=gate_fid, meas_fid=meas_fid)
            self.resource_manager = _FakeResourceManager()

        def receive_message(self, src: str, msg: Message):
            for protocol in self.protocols:
                if protocol.name == msg.receiver:
                    protocol.received_message(src, msg)

    return FakeNode(name, timeline, gate_fid, meas_fid)


def _memory(name: str, timeline, coherence_time: float = 1.0, errors=None):
    from sequence.components.memory import Memory

    return Memory(name, timeline, fidelity=1.0, frequency=0, efficiency=1,
                  coherence_time=coherence_time, wavelength=500,
                  decoherence_errors=errors)


def sequence_purify(kept_state, measured_state, gate_fid: float, meas_fid: float,
                    kind: str = "dejmps") -> tuple[float, NDArray[np.float64]]:
    """SeQUeNCe's own purification of two Bell-diagonal states.

    Args:
        kept_state, measured_state: the pair kept and the pair sacrificed.
        gate_fid, meas_fid: local operation quality, passed into the node so
            that a value the environment fails to forward shows up as a
            disagreement rather than as nothing.
        kind: ``"dejmps"`` or ``"bbpssw"``.

    Returns:
        ``(p_success, state)`` exactly as the protocol object reports them.
    """
    from sequence.entanglement_management.purification import BBPSSW_BDS, DEJMPS_BDS

    timeline = _fresh_timeline()
    owner = _fake_node("p1", timeline, gate_fid, meas_fid)
    remote = _fake_node("p2", timeline, gate_fid, meas_fid)
    kept1, kept2 = _memory("k1", timeline), _memory("k2", timeline)
    meas1, meas2 = _memory("m1", timeline), _memory("m2", timeline)
    timeline.init()

    timeline.quantum_manager.set([kept1.qstate_key, kept2.qstate_key],
                                 np.asarray(kept_state))
    timeline.quantum_manager.set([meas1.qstate_key, meas2.qstate_key],
                                 np.asarray(measured_state))

    protocol_class = BBPSSW_BDS if kind == "bbpssw" else DEJMPS_BDS
    protocol = protocol_class(owner, "p1.ep", kept1, meas1)
    owner.protocols.append(protocol)
    protocol.set_others("p2.ep", remote.name, [kept2.name, meas2.name])
    probability, state = protocol.purification_res()
    return float(probability), np.asarray(state, dtype=float)


def sequence_swap(left_state, right_state, gate_fid: float, meas_fid: float,
                  is_twirled: bool = False) -> NDArray[np.float64]:
    """SeQUeNCe's own entanglement swapping: the merged state."""
    from sequence.entanglement_management.swapping import EntanglementSwappingA_BDS

    timeline = _fresh_timeline()
    middle = _fake_node("s", timeline, gate_fid, meas_fid)
    left, right = _memory("l", timeline), _memory("r", timeline)
    left_far, right_far = _memory("lf", timeline), _memory("rf", timeline)
    timeline.init()

    timeline.quantum_manager.set([left.qstate_key, left_far.qstate_key],
                                 np.asarray(left_state))
    timeline.quantum_manager.set([right.qstate_key, right_far.qstate_key],
                                 np.asarray(right_state))

    protocol = EntanglementSwappingA_BDS(middle, "s.es", left, right,
                                         is_twirled=is_twirled)
    middle.protocols.append(protocol)
    return np.asarray(protocol.swapping_res(), dtype=float)


def sequence_decohere(state, elapsed_s: float, coherence_time: float,
                      errors) -> NDArray[np.float64]:
    """Age a stored pair with ``Memory.bds_decohere`` at **both** of its ends.

    A pair occupies one memory at each end and neither is exempt from decay, so
    SeQUeNCe's single-memory routine is applied twice. This is what the
    environment expresses as ``both_qubits=True``; spelling it out with
    SeQUeNCe's own objects is what lets a mistake in either the rate or the
    number of ends be caught rather than assumed away.
    """
    if elapsed_s <= 0.0:
        return np.asarray(state, dtype=float)

    timeline = _fresh_timeline()
    node = _fake_node("d", timeline, 1.0, 1.0)
    near = _memory("d1", timeline, coherence_time, errors)
    far = _memory("d2", timeline, coherence_time, errors)
    near.owner = node
    far.owner = node
    timeline.init()
    timeline.quantum_manager.set([near.qstate_key, far.qstate_key],
                                 np.asarray(state))

    elapsed_ps = int(round(elapsed_s * 1e12))
    for memory in (near, far):
        memory.last_update_time = 1
        timeline.time = elapsed_ps + 1
        memory.bds_decohere()

    return np.asarray(timeline.quantum_manager.get(near.qstate_key).state,
                      dtype=float)


# ---------------------------------------------------------------------------
# the shadow
# ---------------------------------------------------------------------------

@dataclass
class ShadowStep:
    """One step, as the environment ran it and as SeQUeNCe rebuilt it.

    Attributes:
        action: what was done.
        outcome: the environment's own label for how it turned out.
        elapsed_s: how far the clock moved, read back from the environment.
        live_pairs: pairs held after the step.
        state_divergence: largest absolute disagreement over every component of
            every live pair. This is the number the whole exercise produces.
        probability_divergence: disagreement in the purification success
            probability, or ``nan`` when the step was not a purification. It is
            reported separately because it is a different claim: the state is
            what the pair *becomes*, the probability is how often it does.
    """

    action: Action
    outcome: str
    elapsed_s: float
    live_pairs: int
    state_divergence: float
    probability_divergence: float = float("nan")


@dataclass
class SequenceShadow:
    """Every live pair's state, carried forward with SeQUeNCe alone.

    Attributes:
        env: the environment being shadowed. It is stepped through
            :meth:`step`, never directly, or the two fall out of sync.
        states: the shadow's own copy of each live pair, by pair id.
        steps: one :class:`ShadowStep` per action taken.
        delivered: the shadow's copy of the most recently delivered pair.
    """

    env: Any
    states: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    steps: list[ShadowStep] = field(default_factory=list)
    delivered: NDArray[np.float64] | None = None

    @property
    def memory(self):
        return self.env.config.memory

    @property
    def worst_divergence(self) -> float:
        """The largest disagreement anywhere in the trajectory."""
        if not self.steps:
            return 0.0
        return max(step.state_divergence for step in self.steps)

    def step(self, action: Action):
        """Apply one action to the environment and to the shadow, and compare.

        Returns:
            ``(result, shadow_step)`` — the environment's own ``StepResult`` and
            the comparison this module produces.
        """
        config = self.env.config
        errors = list(self.memory.decoherence_errors)
        coherence = self.memory.coherence_time_s

        before = self.env.state.now_s
        live_before = set(self.env.state.pairs)

        # -- what SeQUeNCe says this operation produces, computed first -----
        expected: Any = None
        if action.kind is ActionKind.PURIFY:
            kept_id, consumed_id = action.pair_ids
            expected = sequence_purify(self.states[kept_id],
                                       self.states[consumed_id],
                                       config.gate_fid, config.meas_fid,
                                       kind=config.purification)
        elif action.kind is ActionKind.SWAP:
            left_id, right_id = action.pair_ids
            expected = sequence_swap(self.states[left_id], self.states[right_id],
                                     config.gate_fid, config.meas_fid,
                                     is_twirled=config.twirl_on_swap)

        _, result = self.env.step(action)
        outcome = result.info["outcome"]
        elapsed = self.env.state.now_s - before

        # -- the operation itself, at the instant it happened ---------------
        probability_divergence = float("nan")
        if action.kind is ActionKind.PURIFY:
            probability, purified = expected
            probability_divergence = abs(result.info["p_success"] - probability)
            kept_id, consumed_id = action.pair_ids
            self.states.pop(consumed_id, None)
            if outcome == "purify_succeeded":
                self.states[kept_id] = purified
            else:
                self.states.pop(kept_id, None)
        elif action.kind is ActionKind.SWAP:
            for pair_id in action.pair_ids:
                self.states.pop(pair_id, None)
        elif action.kind in (ActionKind.DELIVER, ActionKind.DISCARD):
            (pair_id,) = action.pair_ids
            self.delivered = self.states.pop(pair_id, None)

        # -- then the clock, which decoheres everything still held ----------
        for pair_id in list(self.states):
            self.states[pair_id] = sequence_decohere(
                self.states[pair_id], elapsed, coherence, errors)

        # -- and finally the pairs that exist only once the clock has moved -
        for pair_id in set(self.env.state.pairs) - live_before:
            if action.kind is ActionKind.SWAP:
                # Heralded at the swap instant, then aged like anything else.
                self.states[pair_id] = sequence_decohere(expected, elapsed,
                                                         coherence, errors)
            else:
                self.states[pair_id] = np.asarray(self.memory.fresh_state(),
                                                  dtype=float)

        # -- pairs the environment dropped for the cutoff -------------------
        for pair_id in set(self.states) - set(self.env.state.pairs):
            self.states.pop(pair_id)

        shadow_step = ShadowStep(
            action=action, outcome=outcome, elapsed_s=elapsed,
            live_pairs=len(self.states),
            state_divergence=self.divergence(),
            probability_divergence=probability_divergence)
        self.steps.append(shadow_step)
        return result, shadow_step

    def divergence(self) -> float:
        """Largest absolute disagreement over every component of every pair.

        Raises:
            AssertionError: if the two disagree about which pairs *exist*. That
                is not a numerical disagreement and averaging it into one would
                hide it.
        """
        if set(self.states) != set(self.env.state.pairs):
            raise AssertionError(
                f"shadow holds {sorted(self.states)} but the environment holds "
                f"{sorted(self.env.state.pairs)}")
        worst = 0.0
        for pair_id, state in self.states.items():
            actual = np.asarray(self.env.state.pairs[pair_id].state, dtype=float)
            worst = max(worst, float(np.max(np.abs(actual - state))))
        return worst
