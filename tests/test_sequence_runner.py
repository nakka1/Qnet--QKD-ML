"""The phase-2 primitives, executed by SeQUeNCe rather than by a formula.

``tests/test_env_matches_sequence.py`` compares the analytic environment's
arithmetic with SeQUeNCe's. This file tests the other runner: the one that puts
the same action space on top of real quantum routers, real channels and real
detectors, and lets the event loop decide what happens.

The test that matters most here is not that any of it runs — it is
:meth:`TestHeraldingRateMatchesTheModel.test_the_measured_rate_matches_the_closed_form`.
Two simulators can only be compared if they are running the same hardware, and
the efficiencies that make up one arm of a heralded link live in one product in
the analytic model and in three different objects in SeQUeNCe. Getting that
mapping wrong changes the generation rate by a factor of two and looks exactly
like a result about discrete-event simulation. The test measures the rate and
holds it against the closed form.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from qnet_autopilot.env.actions import Action, ActionKind
from qnet_autopilot.env.network_env import EnvironmentConfig, QuantumNetworkEnv
from qnet_autopilot.env.sequence_runner import SequenceNetwork, run_sequence_episode
from qnet_autopilot.env.state import Request
from qnet_autopilot.physics.memory import MemoryParameters

#: Short spans and generous hardware: this file is testing the plumbing, and a
#: link that heralds once every few hundred rounds turns every test into a
#: benchmark.
def _config(**overrides) -> EnvironmentConfig:
    graph = nx.Graph()
    graph.add_edge("a", "b", length=20.0)
    graph.add_edge("b", "c", length=20.0)
    settings = dict(
        graph=graph,
        memory=MemoryParameters(coherence_time_s=1.0, raw_fidelity=0.92,
                                cutoff_ratio=0.5),
        gate_fid=0.97, meas_fid=0.98,
        coupling_efficiency=0.8, detector_efficiency=0.9,
        purification="dejmps", swap_success_prob=1.0,
        generate_budget_s=0.05, max_pairs_per_link=3,
    )
    settings.update(overrides)
    return EnvironmentConfig(**settings)


def _generated(network, node_a, node_b, tries=6):
    """Generate one pair, retrying whole budgets rather than failing a test."""
    for _ in range(tries):
        outcome = network.generate(node_a, node_b)
        if outcome.outcome == "generate_succeeded":
            return outcome
    raise AssertionError(f"no pair on {node_a}-{node_b} after {tries} budgets")


class TestTheNetworkItself:
    """Building it is most of the work, so the build is worth testing."""

    def test_it_builds_routers_and_a_bsm_node_per_edge(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=0)
        assert set(network.routers) == set(config.nodes)
        assert len(network.bsm_nodes) == len(config.edges)
        assert network.now_s == 0.0

    def test_the_configured_hardware_reaches_the_components(self, bds_timeline):
        """Every efficiency has to land on the object that applies it."""
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=0)

        array = network.routers["a"].get_components_by_type("MemoryArray")[0]
        assert array[0].efficiency == pytest.approx(
            config.coupling_efficiency ** 2 * config.memory_efficiency)
        assert array[0].coherence_time == pytest.approx(
            config.memory.coherence_time_s)
        assert array[0].raw_fidelity == pytest.approx(config.memory.raw_fidelity)

        bsm = network.bsm_nodes[config.key("a", "b")] \
            .get_components_by_type("SingleHeraldedBSM")[0]
        for detector in bsm.detectors:
            assert detector.efficiency == pytest.approx(config.detector_efficiency)

    def test_the_gate_fidelities_reach_the_routers(self, bds_timeline):
        """The upstream bug the README documents, checked on this path too."""
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=0)
        for router in network.routers.values():
            assert router.gate_fid == pytest.approx(config.gate_fid)
            assert router.meas_fid == pytest.approx(config.meas_fid)

    def test_the_decoherence_rate_is_set_and_not_merely_the_coherence_time(
            self, bds_timeline):
        """A second silently-discarded parameter, found by this runner.

        ``MemoryArray.update_memory_params`` is a bare ``setattr``, and
        ``decoherence_rate`` is computed from ``coherence_time`` once, in the
        constructor. Setting the coherence time afterwards leaves the rate at
        zero, stored pairs never decohere, and every fidelity in the run is the
        raw one. It is the same shape of bug as the gate-fidelity one and just
        as quiet.
        """
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=0)
        array = network.routers["a"].get_components_by_type("MemoryArray")[0]
        assert array[0].decoherence_rate == pytest.approx(
            1.0 / config.memory.coherence_time_s)
        assert array[0].decoherence_errors is not None


class TestHeraldingRateMatchesTheModel:
    """The alignment that makes a two-simulator comparison mean anything."""

    def test_the_measured_rate_matches_the_closed_form(self, bds_timeline):
        """Rounds to first success, against ``attempt_success_probability``.

        The number of rounds to the first herald is geometric, so its mean is
        the reciprocal of the per-round probability. Sixty successes gives a
        standard error near 13% of the estimate, which is tight enough to catch
        the factor-of-two errors this mapping invites and loose enough not to
        fail on luck.
        """
        config = _config(generate_budget_s=0.3)
        network = SequenceNetwork(config, memory_size=4, seed=0)

        rounds = []
        for _ in range(60):
            outcome = network.generate("a", "b")
            if outcome.outcome != "generate_succeeded":
                continue
            rounds.append(outcome.info["rounds"])
            network.discard(next(iter(network.pairs)))

        assert len(rounds) >= 40, "the link barely heralds; the test is not measuring"
        measured = 1.0 / float(np.mean(rounds))
        expected = config.link_probability[config.key("a", "b")]
        error = measured / np.sqrt(len(rounds))
        assert abs(measured - expected) < 4.0 * error, (
            f"SeQUeNCe heralds at {measured:.4f} per round, the closed form "
            f"says {expected:.4f} (+-{error:.4f}); the two simulators are not "
            f"running the same link")

    def test_a_heralded_pair_carries_the_configured_raw_fidelity(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        _generated(network, "a", "b")
        pair = next(iter(network.pairs.values()))
        assert pair.fidelity(network) == pytest.approx(
            config.memory.raw_fidelity, abs=5e-3)


class TestThePrimitives:
    """Each one, and the state SeQUeNCe is left in afterwards."""

    def test_generation_occupies_a_memory_at_each_end(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        before = (network.free_slots("a"), network.free_slots("b"))
        _generated(network, "a", "b")
        after = (network.free_slots("a"), network.free_slots("b"))
        assert after == (before[0] - 1, before[1] - 1)

    def test_purification_succeeds_and_is_not_reported_as_failure(self, bds_timeline):
        """The regression test for a real bug in this runner.

        SeQUeNCe moves a successfully purified memory to ``PURIFIED``, not back
        to ``ENTANGLED``. The first version of this runner checked only for
        ``ENTANGLED`` and reported every success as a failure — with the
        fidelities still correct, so nothing looked wrong except a success rate
        nobody had a baseline for.
        """
        config = _config()
        network = SequenceNetwork(config, memory_size=6, seed=2)
        for _ in range(2):
            _generated(network, "a", "b")

        kept, consumed = sorted(network.pairs)
        outcomes = [network.purify(kept, consumed).outcome]
        for seed in (3, 4, 5):
            other = SequenceNetwork(config, memory_size=6, seed=seed)
            for _ in range(2):
                _generated(other, "a", "b")
            first, second = sorted(other.pairs)
            outcomes.append(other.purify(first, second).outcome)

        assert any(outcome == "purify_succeeded" for outcome in outcomes), (
            f"no purification succeeded in four attempts: {outcomes}")

    def test_a_swap_produces_a_pair_between_the_far_ends(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=6, seed=1)
        _generated(network, "a", "b")
        _generated(network, "b", "c")
        left, right = sorted(network.pairs)

        outcome = network.swap("b", left, right)
        assert outcome.outcome == "swap_succeeded"
        merged = next(pair for pair in network.pairs.values() if pair.hops == 2)
        assert set(merged.nodes) == {"a", "c"}
        assert merged.distance_km == pytest.approx(40.0)
        assert network.free_slots("b") == 6, "the middle node kept its memories"

    def test_a_swap_costs_fidelity(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=6, seed=1)
        _generated(network, "a", "b")
        _generated(network, "b", "c")
        left, right = sorted(network.pairs)
        before = network.pairs[left].fidelity(network)
        network.swap("b", left, right)
        merged = next(pair for pair in network.pairs.values() if pair.hops == 2)
        assert merged.fidelity(network) < before

    def test_delivering_frees_both_memories(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        _generated(network, "a", "b")
        pair_id = next(iter(network.pairs))
        network.deliver(pair_id)
        assert network.pairs == {}
        assert network.free_slots("a") == 4
        assert len(network.delivered) == 1

    def test_waiting_moves_only_the_clock(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        before = network.now_s
        network.wait(0.01)
        assert network.now_s == pytest.approx(before + 0.01)

    def test_an_idle_pair_decoheres_with_the_clock(self, bds_timeline):
        """SeQUeNCe ages a memory only when something touches it."""
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        _generated(network, "a", "b")
        pair = next(iter(network.pairs.values()))
        before = pair.fidelity(network)
        network.wait(0.2)
        assert pair.fidelity(network) < before

    def test_a_pair_past_the_cutoff_is_dropped(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        _generated(network, "a", "b")
        network.wait(config.memory.cutoff_time_s * 1.5)
        assert network.expire() == 1
        assert network.pairs == {}
        assert network.discarded == 1


class TestTheBridgeToThePolicy:
    """The mirror the feature encoders read, and the action sets."""

    def test_the_mirror_reproduces_the_live_pairs(self, bds_timeline):
        config = _config()
        network = SequenceNetwork(config, memory_size=6, seed=1)
        _generated(network, "a", "b")
        _generated(network, "b", "c")

        state = network.as_state()
        assert set(state.pairs) == set(network.pairs)
        for pair_id, mirrored in state.pairs.items():
            original = network.pairs[pair_id]
            assert set(mirrored.nodes) == set(original.nodes)
            assert float(mirrored.state[0]) == pytest.approx(
                original.fidelity(network))
        assert state.memory_used["b"] == 2
        assert state.now_s == pytest.approx(network.now_s)

    def test_the_mirrored_state_is_not_aged_twice(self, bds_timeline):
        """The state read out of SeQUeNCe has already been brought to now."""
        config = _config()
        network = SequenceNetwork(config, memory_size=4, seed=1)
        _generated(network, "a", "b")
        network.wait(0.05)
        state = network.as_state()
        mirrored = next(iter(state.pairs.values()))
        assert mirrored.last_update_s == pytest.approx(network.now_s)
        assert mirrored.fidelity_at(network.now_s, config.memory) == pytest.approx(
            float(mirrored.state[0]))

    def test_both_simulators_offer_the_same_actions(self, bds_timeline):
        """A disagreement here would make the policy comparison meaningless."""
        config = _config()
        network = SequenceNetwork(config, memory_size=6, seed=1)
        request = Request(source="a", destination="c", min_fidelity=0.5,
                          deadline_s=10.0)

        encoder = QuantumNetworkEnv(config, rng=0)
        encoder.reset(request=request, memory_size=6)

        for step in range(4):
            if step < 2:
                _generated(network, *(("a", "b") if step == 0 else ("b", "c")))
            encoder.state = network.as_state()
            assert [str(action) for action in network.legal_actions(request)] == \
                   [str(action) for action in encoder.legal_actions()]

    def test_an_episode_runs_to_delivery(self, bds_timeline):
        from qnet_autopilot.agents.baselines import ShortestPathPolicy

        config = _config(max_steps=40)
        request = Request(source="a", destination="c", min_fidelity=0.6,
                          deadline_s=1.0)
        record = run_sequence_episode(config, ShortestPathPolicy(), request,
                                      seed=1, memory_size=6)
        assert record["legality_disagreements"] == 0
        assert record["delivered"] >= 1
        assert record["elapsed_s"] > 0.0
        assert 0.5 < record["mean_fidelity"] <= 1.0
