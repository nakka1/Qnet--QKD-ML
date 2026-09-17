"""The load-bearing test: our closed forms must equal SeQUeNCe's, exactly.

:mod:`qnet_autopilot.physics.bell_diagonal` exists to be fast, and it is only
worth having if it is also *right*.  "Right" here has a precise meaning: it
must return what SeQUeNCe's own protocol objects return, to machine precision,
for every input — not approximately, not in the regime we happened to test.

So these tests build the real thing — a ``Timeline`` with the Bell-diagonal
quantum manager, ``FakeNode`` instances carrying ``gate_fid`` and ``meas_fid``,
``Memory`` objects holding states — call SeQUeNCe's ``purification_res()`` and
``swapping_res()``, and compare against our functions over a grid of fidelities
and hardware imperfections.

If SeQUeNCe changes a formula upstream, this test goes red and the transcription
gets updated.  That is the point: the fast path is allowed to be a copy, but it
is not allowed to silently become a stale copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.physics import bell_diagonal as bds

TOLERANCE = 1e-12

FIDELITIES = [0.55, 0.7, 0.85, 0.95, 1.0]
HARDWARE = [(1.0, 1.0), (0.99, 0.99), (0.95, 0.98), (0.9, 0.9)]


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def _fake_node(name, timeline, gate_fid, meas_fid):
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


class _FakeResourceManager:
    def update(self, protocol, memory, state):
        if state == "RAW":
            memory.reset()


def _memory(name, timeline):
    from sequence.components.memory import Memory

    return Memory(name, timeline, fidelity=1.0, frequency=0, efficiency=1,
                  coherence_time=1, wavelength=500)


def _purification_protocol(kind, timeline, kept_state, meas_state, gate_fid, meas_fid):
    """Build a SeQUeNCe purification protocol holding the two given states."""
    from sequence.entanglement_management.purification import BBPSSW_BDS, DEJMPS_BDS

    owner = _fake_node("a1", timeline, gate_fid, meas_fid)
    remote = _fake_node("a2", timeline, gate_fid, meas_fid)

    kept1, kept2 = _memory("kept1", timeline), _memory("kept2", timeline)
    meas1, meas2 = _memory("meas1", timeline), _memory("meas2", timeline)
    timeline.init()

    timeline.quantum_manager.set([kept1.qstate_key, kept2.qstate_key], np.asarray(kept_state))
    timeline.quantum_manager.set([meas1.qstate_key, meas2.qstate_key], np.asarray(meas_state))

    protocol_class = BBPSSW_BDS if kind == "bbpssw" else DEJMPS_BDS
    protocol = protocol_class(owner, "a1.ep1", kept1, meas1)
    owner.protocols.append(protocol)
    protocol.set_others("a2.ep2", remote.name, [kept2.name, meas2.name])
    return protocol


def _swapping_protocol(timeline, left_state, right_state, gate_fid, meas_fid, is_twirled):
    from sequence.entanglement_management.swapping import EntanglementSwappingA_BDS

    middle = _fake_node("b", timeline, gate_fid, meas_fid)
    left_memo, right_memo = _memory("left", timeline), _memory("right", timeline)
    left_far, right_far = _memory("left_far", timeline), _memory("right_far", timeline)
    timeline.init()

    timeline.quantum_manager.set([left_memo.qstate_key, left_far.qstate_key],
                                 np.asarray(left_state))
    timeline.quantum_manager.set([right_memo.qstate_key, right_far.qstate_key],
                                 np.asarray(right_state))

    protocol = EntanglementSwappingA_BDS(middle, "b.es", left_memo, right_memo,
                                         is_twirled=is_twirled)
    middle.protocols.append(protocol)
    return protocol


# ---------------------------------------------------------------------------
# purification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate_fid,meas_fid", HARDWARE)
@pytest.mark.parametrize("kept_fidelity", FIDELITIES)
@pytest.mark.parametrize("meas_fidelity", FIDELITIES)
def test_bbpssw_matches_sequence(bds_timeline, kept_fidelity, meas_fidelity,
                                 gate_fid, meas_fid):
    kept_state = bds.werner(kept_fidelity)
    meas_state = bds.werner(meas_fidelity)

    protocol = _purification_protocol("bbpssw", bds_timeline, kept_state, meas_state,
                                      gate_fid, meas_fid)
    expected_p, expected_state = protocol.purification_res()

    actual_p, actual_state = bds.bbpssw(kept_state, meas_state, gate_fid, meas_fid)

    assert actual_p == pytest.approx(float(expected_p), abs=TOLERANCE)
    assert np.allclose(actual_state, np.asarray(expected_state), atol=TOLERANCE)


@pytest.mark.parametrize("gate_fid,meas_fid", HARDWARE)
@pytest.mark.parametrize("kept_fidelity", FIDELITIES)
@pytest.mark.parametrize("meas_fidelity", FIDELITIES)
def test_dejmps_matches_sequence_on_werner_inputs(bds_timeline, kept_fidelity,
                                                  meas_fidelity, gate_fid, meas_fid):
    kept_state = bds.werner(kept_fidelity)
    meas_state = bds.werner(meas_fidelity)

    protocol = _purification_protocol("dejmps", bds_timeline, kept_state, meas_state,
                                      gate_fid, meas_fid)
    expected_p, expected_state = protocol.purification_res()

    actual_p, actual_state = bds.dejmps(kept_state, meas_state, gate_fid, meas_fid)

    assert actual_p == pytest.approx(float(expected_p), abs=TOLERANCE)
    assert np.allclose(actual_state, np.asarray(expected_state), atol=TOLERANCE)


@pytest.mark.parametrize("gate_fid,meas_fid", [(1.0, 1.0), (0.97, 0.98)])
def test_dejmps_matches_sequence_on_general_bell_diagonal_inputs(
        bds_timeline, gate_fid, meas_fid, rng):
    """The case that separates DEJMPS from BBPSSW: a non-Werner input."""
    kept_state = rng.dirichlet([8, 2, 1, 1])
    meas_state = rng.dirichlet([9, 1, 2, 1])

    protocol = _purification_protocol("dejmps", bds_timeline, kept_state, meas_state,
                                      gate_fid, meas_fid)
    expected_p, expected_state = protocol.purification_res()

    actual_p, actual_state = bds.dejmps(kept_state, meas_state, gate_fid, meas_fid)

    assert actual_p == pytest.approx(float(expected_p), abs=TOLERANCE)
    assert np.allclose(actual_state, np.asarray(expected_state), atol=TOLERANCE)


# ---------------------------------------------------------------------------
# swapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_twirled", [True, False])
@pytest.mark.parametrize("gate_fid,meas_fid", HARDWARE)
@pytest.mark.parametrize("left_fidelity", [0.6, 0.8, 0.95])
@pytest.mark.parametrize("right_fidelity", [0.6, 0.8, 0.95])
def test_swap_matches_sequence(bds_timeline, left_fidelity, right_fidelity,
                               gate_fid, meas_fid, is_twirled):
    left_state = bds.werner(left_fidelity)
    right_state = bds.werner(right_fidelity)

    protocol = _swapping_protocol(bds_timeline, left_state, right_state,
                                  gate_fid, meas_fid, is_twirled)
    expected_state = protocol.swapping_res()

    actual_state = bds.swap(left_state, right_state, gate_fid, meas_fid,
                            is_twirled=is_twirled)

    assert np.allclose(actual_state, np.asarray(expected_state), atol=TOLERANCE)


@pytest.mark.parametrize("is_twirled", [True, False])
def test_swap_matches_sequence_on_general_inputs(bds_timeline, is_twirled, rng):
    left_state = rng.dirichlet([10, 1, 2, 1])
    right_state = rng.dirichlet([7, 2, 1, 3])

    protocol = _swapping_protocol(bds_timeline, left_state, right_state,
                                  0.98, 0.99, is_twirled)
    expected_state = protocol.swapping_res()

    actual_state = bds.swap(left_state, right_state, 0.98, 0.99, is_twirled=is_twirled)

    assert np.allclose(actual_state, np.asarray(expected_state), atol=TOLERANCE)


# ---------------------------------------------------------------------------
# decoherence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coherence_time", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("elapsed_s", [1e-4, 1e-2, 0.5])
def test_decohere_matches_sequence_memory(bds_timeline, coherence_time, elapsed_s):
    """Compare against ``Memory.bds_decohere`` driven by a real timeline clock."""
    from sequence.components.memory import Memory

    errors = [0.2, 0.3, 0.5]
    initial_state = bds.werner(0.9)

    node = _fake_node("a1", bds_timeline, 1.0, 1.0)
    memory = Memory("m1", bds_timeline, fidelity=1.0, frequency=0, efficiency=1,
                    coherence_time=coherence_time, wavelength=500,
                    decoherence_errors=errors)
    partner = Memory("m2", bds_timeline, fidelity=1.0, frequency=0, efficiency=1,
                     coherence_time=coherence_time, wavelength=500,
                     decoherence_errors=errors)
    memory.owner = node
    bds_timeline.init()
    bds_timeline.quantum_manager.set([memory.qstate_key, partner.qstate_key], initial_state)

    # bds_decohere only acts once last_update_time is set and the clock has moved.
    memory.last_update_time = 1
    bds_timeline.time = int(round(elapsed_s * 1e12)) + 1
    memory.bds_decohere()
    expected_state = bds_timeline.quantum_manager.get(memory.qstate_key).state

    x_rate, y_rate, z_rate = (errors[0] / coherence_time,
                              errors[1] / coherence_time,
                              errors[2] / coherence_time)
    actual_state = bds.decohere(initial_state, x_rate, y_rate, z_rate, elapsed_s,
                                both_qubits=False)

    assert np.allclose(actual_state, np.asarray(expected_state), atol=1e-10)
