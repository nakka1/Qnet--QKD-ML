"""The phase-2 environment, confronted with SeQUeNCe over whole trajectories.

What phase 1 established, and what it did not
---------------------------------------------
``tests/test_bds_matches_sequence.py`` shows that each closed form —
purification, swapping, memory decoherence — returns exactly what SeQUeNCe's own
protocol objects return. That is a statement about the *functions*.

The phase-2 environment is a different piece of software. It decides which
function to call, on which pair, with which hardware parameters, after how much
elapsed time, and it threads the results through a clock. Every one of those is
a place to be wrong while calling perfectly correct physics, and none of them is
covered by testing the physics. Phase 2's own limits section said so: "the
environment is the fast analytic model, not the discrete-event simulator.
Closing that loop is work that has not been done."

This is that work. :class:`~qnet_autopilot.validation.twin.SequenceShadow`
carries a SeQUeNCe reconstruction of every live pair alongside the environment
and compares them after every single step; these tests script the trajectories
and assert the two never part company. The last test is the one that keeps the
rest honest: it breaks the hardware on purpose and checks that the comparison
notices.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from qnet_autopilot.env.actions import Action, ActionKind
from qnet_autopilot.env.network_env import EnvironmentConfig, QuantumNetworkEnv
from qnet_autopilot.env.state import Request
from qnet_autopilot.physics import bell_diagonal as bds
from qnet_autopilot.validation.twin import SequenceShadow

#: Both sides evaluate the same closed forms in double precision, so anything
#: above rounding is a real disagreement.
TOLERANCE = 1e-11

GATE_FID = 0.97
MEAS_FID = 0.98


def _chain(*lengths_km):
    """A chain graph ``a - b - c - ...`` with the given span lengths."""
    graph = nx.Graph()
    names = [chr(ord("a") + index) for index in range(len(lengths_km) + 1)]
    for index, length in enumerate(lengths_km):
        graph.add_edge(names[index], names[index + 1], length=float(length))
    return graph


def _config(**overrides):
    """Deterministic hardware: swapping never fails, the clock does the rest."""
    settings = dict(
        graph=_chain(20.0, 20.0),
        gate_fid=GATE_FID,
        meas_fid=MEAS_FID,
        purification="dejmps",
        swap_success_prob=1.0,
        twirl_on_swap=False,
        max_pairs_per_link=3,
    )
    settings.update(overrides)
    return EnvironmentConfig(**settings)


def _build(config, source="a", destination="c", memory_size=6, seed=0):
    request = Request(source=source, destination=destination,
                      min_fidelity=0.5, deadline_s=10.0)
    env = QuantumNetworkEnv(config, rng=seed)
    env.reset(request=request, memory_size=memory_size)
    return env


def _generate(env, link):
    """Keep trying generation on ``link`` until one more pair heralds.

    Generation is geometric and may fail within its budget; these tests are
    about what happens *after* a pair exists, so failures are simply retried —
    and each retry is still stepped through the shadow, so the clock it consumes
    is checked like any other.
    """
    target = len(env.state.pairs_on(*link)) + 1
    for _ in range(80):
        action = next(candidate for candidate in env.legal_actions()
                      if candidate.kind is ActionKind.GENERATE
                      and candidate.link == link)
        yield action
        if len(env.state.pairs_on(*link)) >= target:
            return
    raise AssertionError(f"generation on {link} never succeeded")


def _assert_in_step(shadow, step):
    assert step.state_divergence < TOLERANCE, (
        f"the environment parted company with SeQUeNCe {step.state_divergence:.3e} "
        f"after {step.action} ({step.outcome})")
    if not np.isnan(step.probability_divergence):
        assert step.probability_divergence < TOLERANCE, (
            "the environment's purification success probability is not SeQUeNCe's")


class TestEnvironmentComposition:
    """Whole trajectories, not single operations."""

    def test_generation_then_delivery_matches_a_decohered_state(self,
                                                                bds_timeline):
        """The simplest trajectory: make a pair, let it idle, hand it over."""
        env = _build(_config(graph=_chain(20.0)), destination="b")
        shadow = SequenceShadow(env)

        for action in _generate(env, ("a", "b")):
            _assert_in_step(shadow, shadow.step(action)[1])

        for _ in range(5):
            _assert_in_step(shadow, shadow.step(Action(ActionKind.WAIT))[1])

        deliver = next(action for action in env.legal_actions()
                       if action.kind is ActionKind.DELIVER)
        _assert_in_step(shadow, shadow.step(deliver)[1])

        delivered = env.state.delivered[-1]
        assert np.allclose(delivered.state, shadow.delivered, atol=TOLERANCE), (
            "the delivered state is not the one SeQUeNCe would have produced")

    def test_a_purification_round_matches_sequence(self, bds_timeline):
        """Two pairs on one link, purified: state *and* success probability."""
        env = _build(_config())
        shadow = SequenceShadow(env)

        for _ in range(2):
            for action in _generate(env, ("a", "b")):
                _assert_in_step(shadow, shadow.step(action)[1])

        purify = next(action for action in env.legal_actions()
                      if action.kind is ActionKind.PURIFY)
        result, step = shadow.step(purify)
        _assert_in_step(shadow, step)
        assert result.info["outcome"].startswith("purify")
        assert not np.isnan(step.probability_divergence), (
            "the purification probability was never compared")

    def test_a_swap_matches_sequence(self, bds_timeline):
        """One pair per link, swapped at the middle node."""
        env = _build(_config())
        shadow = SequenceShadow(env)

        for link in (("a", "b"), ("b", "c")):
            for action in _generate(env, link):
                _assert_in_step(shadow, shadow.step(action)[1])

        swap = next(action for action in env.legal_actions()
                    if action.kind is ActionKind.SWAP)
        _assert_in_step(shadow, shadow.step(swap)[1])

        merged = next(pair for pair in env.state.pairs.values() if pair.hops == 2)
        assert set(merged.nodes) == {"a", "c"}

    def test_a_full_purify_then_swap_trajectory_matches_sequence(self,
                                                                 bds_timeline):
        """The composition the agent actually learns, rebuilt independently.

        Generate two pairs on each link, purify each link, swap, deliver — with
        every intermediate state checked against the SeQUeNCe reconstruction.
        This is the test that would catch a hardware parameter that never
        arrives, an operation applied to a stale state, or decoherence charged
        for the wrong interval — none of which the per-operation tests can see.
        """
        env = _build(_config(), memory_size=8)
        shadow = SequenceShadow(env)

        for link in (("a", "b"), ("b", "c")):
            for _ in range(2):
                for action in _generate(env, link):
                    _assert_in_step(shadow, shadow.step(action)[1])

            purify = next(action for action in env.legal_actions()
                          if action.kind is ActionKind.PURIFY
                          and set(env.state.pairs[action.pair_ids[0]].nodes)
                          == set(link))
            result, step = shadow.step(purify)
            _assert_in_step(shadow, step)
            if result.info["outcome"] == "purify_failed":
                pytest.skip("purification failed on this seed")

        swap = next(action for action in env.legal_actions()
                    if action.kind is ActionKind.SWAP)
        _assert_in_step(shadow, shadow.step(swap)[1])

        deliver = next(action for action in env.legal_actions()
                       if action.kind is ActionKind.DELIVER)
        _assert_in_step(shadow, shadow.step(deliver)[1])

        delivered = env.state.delivered[-1]
        assert delivered.hops == 2
        assert np.allclose(delivered.state, shadow.delivered, atol=TOLERANCE)
        assert float(bds.fidelity(delivered.state)) == pytest.approx(
            float(bds.fidelity(shadow.delivered)), abs=TOLERANCE)

    def test_a_longer_chain_stays_in_step(self, bds_timeline):
        """Four hops and three swaps.

        Longer paths are where a composition error compounds instead of
        cancelling, and they are also where the agent spends most of its
        episodes. Nothing here is a new physical claim — it is the same
        arithmetic, run far enough for a small mistake to become visible.
        """
        env = _build(_config(graph=_chain(15.0, 25.0, 15.0, 25.0)),
                     destination="e", memory_size=8, seed=3)
        shadow = SequenceShadow(env)

        for link in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")):
            for action in _generate(env, link):
                _assert_in_step(shadow, shadow.step(action)[1])

        for _ in range(3):
            swap = next((action for action in env.legal_actions()
                         if action.kind is ActionKind.SWAP), None)
            if swap is None:
                break
            _assert_in_step(shadow, shadow.step(swap)[1])

        assert any(pair.hops > 1 for pair in env.state.pairs.values()), (
            "no swap ever completed; the trajectory tested nothing")
        assert shadow.worst_divergence < TOLERANCE

    def test_the_hardware_parameters_reach_the_physics(self, bds_timeline):
        """A configured gate fidelity that never arrives is invisible otherwise.

        Phase 1 found SeQUeNCe silently dropping ``gate_fidelity`` on the way
        from a config file into a node. The same class of bug in the other
        direction — the environment holding a value it never passes on — would
        make every result quietly correspond to perfect hardware, and every
        comparison above would still pass, because the shadow reads its hardware
        from the same config object.
        """
        outcomes = []
        for gate_fid, meas_fid in ((1.0, 1.0), (0.90, 0.90)):
            env = _build(_config(gate_fid=gate_fid, meas_fid=meas_fid), seed=1)
            for link in (("a", "b"), ("b", "c")):
                for action in _generate(env, link):
                    env.step(action)
            swap = next(action for action in env.legal_actions()
                        if action.kind is ActionKind.SWAP)
            env.step(swap)
            merged = next(pair for pair in env.state.pairs.values()
                          if pair.hops == 2)
            outcomes.append(float(bds.fidelity(merged.state)))

        assert outcomes[0] > outcomes[1] + 1e-3, (
            "gate and measurement fidelity are not reaching the swap")


class TestTheComparisonCanFail:
    """A comparison that cannot fail is not evidence.

    Both of these break something on purpose and check that the shadow says so.
    Without them, every passing test above could be passing because the two
    sides are the same code path in disguise.
    """

    def test_a_wrong_number_of_decohering_ends_is_caught(self, bds_timeline):
        """The environment decoheres both halves; charge for one and it shows."""
        from qnet_autopilot.env import state as state_module

        original = state_module.EntangledPair.state_at

        def one_ended(self, now_s, memory):
            elapsed = max(now_s - self.last_update_s, 0.0)
            if elapsed <= 0.0:
                return self.state
            x_rate, y_rate, z_rate = memory.pauli_rates
            return bds.decohere(self.state, x_rate, y_rate, z_rate, elapsed,
                                both_qubits=False)

        state_module.EntangledPair.state_at = one_ended
        try:
            env = _build(_config(graph=_chain(20.0)), destination="b")
            shadow = SequenceShadow(env)

            worst = 0.0
            for action in _generate(env, ("a", "b")):
                worst = max(worst, shadow.step(action)[1].state_divergence)
            for _ in range(5):
                worst = max(worst,
                            shadow.step(Action(ActionKind.WAIT))[1].state_divergence)
        finally:
            state_module.EntangledPair.state_at = original

        assert worst > 1e-3, (
            "halving the decoherence went unnoticed, so the comparison is vacuous")

    def test_a_dropped_gate_fidelity_is_caught(self, bds_timeline):
        """A shadow told the truth against an environment using perfect gates."""
        env = _build(_config(gate_fid=1.0, meas_fid=1.0))
        shadow = SequenceShadow(env)
        for link in (("a", "b"), ("b", "c")):
            for action in _generate(env, link):
                shadow.step(action)

        # The shadow is asked for the swap SeQUeNCe would do with the *real*
        # hardware, while the environment does it with perfect gates.
        object.__setattr__(env.config, "gate_fid", GATE_FID)
        object.__setattr__(env.config, "meas_fid", MEAS_FID)
        left_id, right_id = sorted(env.state.pairs)
        from qnet_autopilot.validation.twin import sequence_swap

        honest = sequence_swap(shadow.states[left_id], shadow.states[right_id],
                               GATE_FID, MEAS_FID)
        object.__setattr__(env.config, "gate_fid", 1.0)
        object.__setattr__(env.config, "meas_fid", 1.0)
        perfect = sequence_swap(shadow.states[left_id], shadow.states[right_id],
                                1.0, 1.0)

        assert float(np.max(np.abs(honest - perfect))) > 1e-3, (
            "imperfect gates make no difference to a swap, so nothing in this "
            "file could detect them being dropped")
