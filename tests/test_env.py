"""Invariants of the environment.

A reinforcement-learning environment is uniquely bad at failing loudly.  A
physics bug produces a wrong number; an environment bug produces a *learnable*
wrong number, and the agent finds it, exploits it, and reports a excellent
score.  These tests pin the invariants an exploit would have to break:

* **Memory is conserved.**  Every live pair holds exactly one slot at each end,
  always.  An accounting leak is free memory, and free memory makes the whole
  scheduling problem disappear.
* **Time only moves forward, and only by the chosen action's latency.**  If any
  path advanced the clock by less than it should, purification would become
  cheaper than physics allows.
* **Operations consume what they claim to.**  Purification takes two pairs and
  returns at most one; swapping likewise.  A conservation error here is
  entanglement from nothing.
* **Illegal actions are refused**, rather than silently becoming no-ops — a
  no-op with a penalty lets a broken policy look like it is exploring.
* **A seed reproduces a run exactly**, which is what makes a paired comparison
  between two policies a comparison of decisions rather than of luck.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.env import (ACTION_FEATURES, EnvironmentConfig, QuantumNetworkEnv,
                                Request, STATE_FEATURES)
from qnet_autopilot.env.actions import Action, ActionKind, legal_actions
from qnet_autopilot.env.features import ActionFeature, StateFeature, covered_segments
from qnet_autopilot.env.state import EntangledPair, NetworkState
from qnet_autopilot.network import scenario
from qnet_autopilot.physics import bell_diagonal as bds
from qnet_autopilot.physics.memory import MemoryParameters


def build_config(**overrides) -> EnvironmentConfig:
    defaults = dict(
        graph=scenario("chain", num_nodes=4, span_km=20),
        memory=MemoryParameters(coherence_time_s=1.0, raw_fidelity=0.90, cutoff_ratio=0.1),
        gate_fid=0.999, meas_fid=0.999,
        coupling_efficiency=0.7, detector_efficiency=0.85,
        generate_budget_s=0.004, wait_slot_s=0.002,
        max_steps=60, max_pairs_per_link=3)
    defaults.update(overrides)
    return EnvironmentConfig(**defaults)


def build_env(**overrides) -> QuantumNetworkEnv:
    return QuantumNetworkEnv(build_config(**overrides), rng=0)


def default_request() -> Request:
    return Request(source="0", destination="3", min_fidelity=0.74, deadline_s=0.08)


def memory_is_consistent(state: NetworkState) -> bool:
    """Recount slots from the live pairs and compare with the ledger."""
    counted: dict[str, int] = {}
    for pair in state.pairs.values():
        for node in pair.nodes:
            counted[node] = counted.get(node, 0) + 1
    for node, used in state.memory_used.items():
        if counted.get(node, 0) != used:
            return False
    return all(node in state.memory_used or count == 0 for node, count in counted.items())


# ---------------------------------------------------------------------------
# state bookkeeping
# ---------------------------------------------------------------------------

class TestNetworkState:

    def test_adding_and_removing_a_pair_balances_memory(self):
        state = NetworkState(memory_size=3)
        pair = EntangledPair.new("a", "b", bds.werner(0.9), 0.0)
        state.add_pair(pair)
        assert state.memory_used["a"] == 1 and state.memory_used["b"] == 1
        assert memory_is_consistent(state)

        state.remove_pair(pair.pair_id)
        assert state.free_slots("a") == 3
        assert memory_is_consistent(state)

    def test_a_full_node_refuses_another_pair(self):
        state = NetworkState(memory_size=1)
        state.add_pair(EntangledPair.new("a", "b", bds.werner(0.9), 0.0))
        assert not state.can_hold("a", "c")
        with pytest.raises(RuntimeError):
            state.add_pair(EntangledPair.new("a", "c", bds.werner(0.9), 0.0))

    def test_other_end_rejects_a_node_that_does_not_hold_the_pair(self):
        pair = EntangledPair.new("a", "b", bds.werner(0.9), 0.0)
        assert pair.other_end("a") == "b"
        with pytest.raises(ValueError):
            pair.other_end("c")

    def test_fidelity_is_derived_from_the_clock_not_stored(self):
        memory = MemoryParameters(coherence_time_s=0.5)
        pair = EntangledPair.new("a", "b", bds.werner(1.0), 0.0)
        assert pair.fidelity_at(0.0, memory) == pytest.approx(1.0)
        assert pair.fidelity_at(0.2, memory) < 1.0
        # Reading at a later time must not mutate the stored state.
        assert pair.last_update_s == 0.0
        assert pair.fidelity_at(0.0, memory) == pytest.approx(1.0)

    def test_advance_expires_pairs_past_the_cutoff(self):
        memory = MemoryParameters(coherence_time_s=1.0, cutoff_ratio=0.1)
        state = NetworkState(memory_size=4)
        state.add_pair(EntangledPair.new("a", "b", bds.werner(0.95), 0.0))
        assert state.advance(0.05, memory) == 0
        assert len(state.pairs) == 1
        assert state.advance(0.10, memory) == 1
        assert len(state.pairs) == 0
        assert memory_is_consistent(state)

    def test_time_cannot_run_backwards(self):
        state = NetworkState()
        with pytest.raises(ValueError):
            state.advance(-1.0, MemoryParameters())


# ---------------------------------------------------------------------------
# legal actions
# ---------------------------------------------------------------------------

class TestLegalActions:

    def test_wait_is_always_available(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        actions = env.legal_actions()
        assert actions[0].kind is ActionKind.WAIT

    def test_generation_is_offered_on_every_edge_at_the_start(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        generate = [a for a in env.legal_actions() if a.kind is ActionKind.GENERATE]
        assert len(generate) == env.config.graph.number_of_edges()

    def test_a_full_link_stops_being_offered(self):
        env = build_env(max_pairs_per_link=1)
        env.reset(default_request(), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0,
                                             distance_km=20.0))
        links = {a.link for a in env.legal_actions() if a.kind is ActionKind.GENERATE}
        assert ("0", "1") not in links

    def test_a_full_node_stops_being_offered(self):
        env = build_env()
        env.reset(default_request(), memory_size=1)
        env.state.add_pair(EntangledPair.new("1", "2", bds.werner(0.9), 0.0,
                                             distance_km=20.0))
        links = {a.link for a in env.legal_actions() if a.kind is ActionKind.GENERATE}
        assert ("0", "1") not in links and ("2", "3") not in links

    def test_purification_needs_two_pairs_on_the_same_span(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        assert not [a for a in env.legal_actions() if a.kind is ActionKind.PURIFY]

        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        purify = [a for a in env.legal_actions() if a.kind is ActionKind.PURIFY]
        assert len(purify) == 1

    def test_swapping_needs_two_pairs_meeting_at_a_node_with_distinct_far_ends(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        env.state.add_pair(EntangledPair.new("1", "2", bds.werner(0.9), 0.0, distance_km=20.0))
        swaps = [a for a in env.legal_actions() if a.kind is ActionKind.SWAP]
        assert len(swaps) == 1 and swaps[0].node == "1"

    def test_no_swap_that_would_make_a_pair_from_a_node_to_itself(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        for _ in range(2):
            env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0,
                                                 distance_km=20.0))
        assert not [a for a in env.legal_actions() if a.kind is ActionKind.SWAP]

    def test_delivery_requires_clearing_the_threshold(self):
        env = build_env()
        env.reset(Request("0", "3", min_fidelity=0.95, deadline_s=0.1), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "3", bds.werner(0.90), 0.0,
                                             hops=3, distance_km=60.0))
        assert not [a for a in env.legal_actions() if a.kind is ActionKind.DELIVER]

        env.state.add_pair(EntangledPair.new("0", "3", bds.werner(0.99), 0.0,
                                             hops=3, distance_km=60.0))
        deliver = [a for a in env.legal_actions() if a.kind is ActionKind.DELIVER]
        assert len(deliver) == 1

    def test_discard_can_be_turned_off(self):
        env = build_env(allow_discard=False)
        env.reset(default_request(), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        assert not [a for a in env.legal_actions() if a.kind is ActionKind.DISCARD]


# ---------------------------------------------------------------------------
# transitions
# ---------------------------------------------------------------------------

class TestTransitions:

    def test_wait_advances_by_exactly_the_slot(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        before = env.state.now_s
        env.step(Action(ActionKind.WAIT))
        assert env.state.now_s - before == pytest.approx(env.config.wait_slot_s)

    def test_every_step_advances_time_by_its_reported_duration(self, rng):
        env = build_env()
        observation, _ = env.reset(default_request(), memory_size=5, rng=3)
        for _ in range(40):
            _, _, actions = observation
            before = env.state.now_s
            action = actions[int(rng.integers(len(actions)))]
            observation, result = env.step(action)
            assert env.state.now_s - before == pytest.approx(result.info["duration_s"])
            assert env.state.now_s >= before
            if result.done:
                break

    def test_purification_consumes_one_pair_and_may_consume_both(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        first = EntangledPair.new("0", "1", bds.werner(0.85), 0.0, distance_km=20.0)
        second = EntangledPair.new("0", "1", bds.werner(0.85), 0.0, distance_km=20.0)
        env.state.add_pair(first)
        env.state.add_pair(second)

        env.step(Action(ActionKind.PURIFY, pair_ids=(first.pair_id, second.pair_id)))
        assert second.pair_id not in env.state.pairs
        assert len(env.state.pairs) <= 1
        assert memory_is_consistent(env.state)

    def test_successful_purification_improves_the_kept_pair(self):
        # gate and measurement perfect, so the round cannot make things worse.
        env = build_env(gate_fid=1.0, meas_fid=1.0)
        env.reset(default_request(), memory_size=4)
        first = EntangledPair.new("0", "1", bds.werner(0.80), 0.0, distance_km=20.0)
        second = EntangledPair.new("0", "1", bds.werner(0.80), 0.0, distance_km=20.0)
        env.state.add_pair(first)
        env.state.add_pair(second)

        for _ in range(30):
            env.rng = np.random.default_rng(0)
            _, result = env.step(Action(ActionKind.PURIFY,
                                        pair_ids=(first.pair_id, second.pair_id)))
            if result.info["outcome"] == "purify_succeeded":
                assert first.fidelity_at(env.state.now_s, env.config.memory) > 0.80
                return
            break
        pytest.skip("purification failed on this draw")

    def test_swapping_consumes_both_inputs(self):
        env = build_env(swap_success_prob=1.0)
        env.reset(default_request(), memory_size=4)
        left = EntangledPair.new("0", "1", bds.werner(0.95), 0.0, distance_km=20.0)
        right = EntangledPair.new("1", "2", bds.werner(0.95), 0.0, distance_km=20.0)
        env.state.add_pair(left)
        env.state.add_pair(right)

        env.step(Action(ActionKind.SWAP, node="1",
                        pair_ids=(left.pair_id, right.pair_id)))
        assert left.pair_id not in env.state.pairs
        assert right.pair_id not in env.state.pairs
        merged = list(env.state.pairs.values())
        assert len(merged) == 1
        assert set(merged[0].nodes) == {"0", "2"}
        assert merged[0].hops == 2
        assert merged[0].distance_km == pytest.approx(40.0)
        assert memory_is_consistent(env.state)

    def test_a_failed_swap_loses_both_pairs(self):
        env = build_env(swap_success_prob=0.0)
        env.reset(default_request(), memory_size=4)
        left = EntangledPair.new("0", "1", bds.werner(0.95), 0.0, distance_km=20.0)
        right = EntangledPair.new("1", "2", bds.werner(0.95), 0.0, distance_km=20.0)
        env.state.add_pair(left)
        env.state.add_pair(right)

        _, result = env.step(Action(ActionKind.SWAP, node="1",
                                    pair_ids=(left.pair_id, right.pair_id)))
        assert result.info["outcome"] == "swap_failed"
        assert not env.state.pairs
        assert memory_is_consistent(env.state)

    def test_a_swapped_pair_inherits_the_age_of_its_oldest_input(self):
        env = build_env(swap_success_prob=1.0)
        env.reset(default_request(), memory_size=4)
        env.state.now_s = 0.02
        old = EntangledPair.new("0", "1", bds.werner(0.95), 0.0, distance_km=20.0)
        fresh = EntangledPair.new("1", "2", bds.werner(0.95), 0.02, distance_km=20.0)
        env.state.add_pair(old)
        env.state.add_pair(fresh)

        env.step(Action(ActionKind.SWAP, node="1", pair_ids=(old.pair_id, fresh.pair_id)))
        merged = list(env.state.pairs.values())[0]
        assert merged.created_s == pytest.approx(0.0)

    def test_delivery_records_the_pair_and_pays_the_reward(self):
        env = build_env()
        env.reset(Request("0", "3", min_fidelity=0.7, deadline_s=0.1), memory_size=4)
        pair = EntangledPair.new("0", "3", bds.werner(0.95), 0.0, hops=3, distance_km=60.0)
        env.state.add_pair(pair)

        _, result = env.step(Action(ActionKind.DELIVER, pair_ids=(pair.pair_id,)))
        assert result.terminated
        assert len(env.state.delivered) == 1
        assert result.raw_reward > env.config.delivery_reward * 0.9
        assert memory_is_consistent(env.state)

    def test_discard_frees_memory_immediately_and_costs_nothing(self):
        env = build_env()
        env.reset(default_request(), memory_size=2)
        pair = EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0)
        env.state.add_pair(pair)

        _, result = env.step(Action(ActionKind.DISCARD, pair_ids=(pair.pair_id,)))
        assert pair.pair_id not in env.state.pairs
        assert env.state.free_slots("0") == 2
        assert result.info["duration_s"] == 0.0
        assert memory_is_consistent(env.state)

    def test_memory_stays_consistent_through_a_random_rollout(self, rng):
        env = build_env()
        observation, _ = env.reset(default_request(), memory_size=5, rng=11)
        for _ in range(60):
            _, _, actions = observation
            observation, result = env.step(actions[int(rng.integers(len(actions)))])
            assert memory_is_consistent(env.state)
            assert all(env.state.free_slots(node) >= 0 for node in env.config.nodes)
            if result.done:
                break

    def test_stepping_before_reset_is_an_error(self):
        env = build_env()
        with pytest.raises(RuntimeError):
            env.step(Action(ActionKind.WAIT))

    def test_the_same_seed_reproduces_the_run(self):
        def rollout(seed):
            env = build_env()
            observation, _ = env.reset(default_request(), memory_size=5, rng=seed)
            generator = np.random.default_rng(99)
            trace = []
            for _ in range(30):
                _, _, actions = observation
                action = actions[int(generator.integers(len(actions)))]
                observation, result = env.step(action)
                trace.append((str(action), round(env.state.now_s, 12),
                              result.info["outcome"]))
                if result.done:
                    break
            return trace

        assert rollout(17) == rollout(17)
        assert rollout(17) != rollout(18)


# ---------------------------------------------------------------------------
# episode outcome
# ---------------------------------------------------------------------------

class TestTermination:

    def test_the_deadline_truncates(self):
        env = build_env(wait_slot_s=0.01)
        observation, _ = env.reset(Request("0", "3", deadline_s=0.05), memory_size=4)
        for _ in range(20):
            observation, result = env.step(Action(ActionKind.WAIT))
            if result.done:
                break
        assert result.truncated and not result.terminated
        assert env.state.now_s >= 0.05

    def test_the_step_cap_truncates(self):
        env = build_env(max_steps=5, wait_slot_s=1e-6)
        env.reset(Request("0", "3", deadline_s=10.0), memory_size=4)
        for _ in range(5):
            _, result = env.step(Action(ActionKind.WAIT))
        assert result.truncated
        assert env.steps == 5

    def test_pairs_needed_greater_than_one_requires_more_than_one_delivery(self):
        env = build_env()
        request = Request("0", "3", min_fidelity=0.7, deadline_s=0.2, pairs_needed=2)
        env.reset(request, memory_size=4)
        for index in range(2):
            pair = EntangledPair.new("0", "3", bds.werner(0.95), env.state.now_s,
                                     hops=3, distance_km=60.0)
            env.state.add_pair(pair)
            _, result = env.step(Action(ActionKind.DELIVER, pair_ids=(pair.pair_id,)))
            assert result.terminated == (index == 1)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_shapes_match_the_declared_sizes(self):
        env = build_env()
        (state_features, action_features, actions), _ = env.reset(
            default_request(), memory_size=4)
        assert state_features.shape == (STATE_FEATURES,)
        assert action_features.shape == (len(actions), ACTION_FEATURES)

    def test_features_are_finite_and_bounded_through_a_rollout(self, rng):
        env = build_env()
        observation, _ = env.reset(default_request(), memory_size=5, rng=5)
        for _ in range(50):
            state_features, action_features, actions = observation
            assert np.all(np.isfinite(state_features))
            assert np.all(np.isfinite(action_features))
            assert np.all(state_features >= -1e-9) and np.all(state_features <= 1 + 1e-9)
            assert np.all(action_features >= -1 - 1e-9)
            assert np.all(action_features <= 1 + 1e-9)
            observation, result = env.step(actions[int(rng.integers(len(actions)))])
            if result.done:
                break

    def test_the_kind_one_hot_matches_the_action(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        env.state.add_pair(EntangledPair.new("1", "2", bds.werner(0.9), 0.0, distance_km=20.0))
        _, action_features, actions = env.observe()
        for row, action in zip(action_features, actions):
            one_hot = row[:len(ActionKind)]
            assert one_hot.sum() == pytest.approx(1.0)
            assert one_hot[int(action.kind)] == pytest.approx(1.0)

    def test_action_features_do_not_depend_on_topology_size(self):
        """The claim that makes transfer possible, checked directly."""
        small = build_env(graph=scenario("chain", num_nodes=3, span_km=20))
        large = build_env(graph=scenario("chain", num_nodes=7, span_km=20))
        small.reset(Request("0", "2", deadline_s=0.08), memory_size=4)
        large.reset(Request("0", "6", deadline_s=0.08), memory_size=4)
        assert small.observe()[1].shape[1] == large.observe()[1].shape[1]
        assert small.observe()[0].shape == large.observe()[0].shape

    def test_covered_segments_tracks_what_the_path_still_needs(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        position = env._path_position
        assert covered_segments(env.state, position) == set()

        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        assert covered_segments(env.state, position) == {0}

        env.state.add_pair(EntangledPair.new("1", "3", bds.werner(0.9), 0.0,
                                             hops=2, distance_km=40.0))
        assert covered_segments(env.state, position) == {0, 1, 2}

    def test_segment_uncovered_flags_only_the_missing_links(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0))
        _, action_features, actions = env.observe()

        for row, action in zip(action_features, actions):
            if action.kind is not ActionKind.GENERATE:
                continue
            expected = 0.0 if set(action.link) == {"0", "1"} else 1.0
            assert row[ActionFeature.SEGMENT_UNCOVERED] == pytest.approx(expected)

    def test_link_occupancy_reflects_stored_pairs(self):
        env = build_env(max_pairs_per_link=3)
        env.reset(default_request(), memory_size=6)
        for _ in range(2):
            env.state.add_pair(EntangledPair.new("0", "1", bds.werner(0.9), 0.0,
                                                 distance_km=20.0))
        _, action_features, actions = env.observe()
        for row, action in zip(action_features, actions):
            if action.kind is ActionKind.GENERATE and set(action.link) == {"0", "1"}:
                assert row[ActionFeature.LINK_OCCUPANCY] == pytest.approx(2 / 3)

    def test_predicted_fidelity_matches_the_physics_it_claims_to_predict(self):
        env = build_env(gate_fid=0.99, meas_fid=0.99)
        env.reset(default_request(), memory_size=4)
        left = EntangledPair.new("0", "1", bds.werner(0.92), 0.0, distance_km=20.0)
        right = EntangledPair.new("1", "2", bds.werner(0.88), 0.0, distance_km=20.0)
        env.state.add_pair(left)
        env.state.add_pair(right)

        _, action_features, actions = env.observe()
        for row, action in zip(action_features, actions):
            if action.kind is not ActionKind.SWAP:
                continue
            expected = bds.fidelity(bds.swap(left.state, right.state,
                                             gate_fid=0.99, meas_fid=0.99))
            assert row[ActionFeature.PREDICTED_FIDELITY] == pytest.approx(float(expected))

    def test_progress_gain_is_negative_for_a_swap_that_walks_backwards(self):
        env = build_env()
        env.reset(default_request(), memory_size=6)
        long_pair = EntangledPair.new("0", "3", bds.werner(0.9), 0.0, hops=3,
                                      distance_km=60.0)
        short_pair = EntangledPair.new("0", "1", bds.werner(0.9), 0.0, distance_km=20.0)
        env.state.add_pair(long_pair)
        env.state.add_pair(short_pair)

        _, action_features, actions = env.observe()
        swaps = [(row, action) for row, action in zip(action_features, actions)
                 if action.kind is ActionKind.SWAP]
        assert swaps
        for row, _ in swaps:
            assert row[ActionFeature.PROGRESS_GAIN] < 0


# ---------------------------------------------------------------------------
# reward and shaping
# ---------------------------------------------------------------------------

class TestReward:

    def test_shaping_does_not_change_the_raw_reward(self, rng):
        plain = build_env(shaping_weight=0.0)
        shaped = build_env(shaping_weight=1.0)
        first, _ = plain.reset(default_request(), memory_size=5, rng=21)
        second, _ = shaped.reset(default_request(), memory_size=5, rng=21)

        generator = np.random.default_rng(4)
        for _ in range(30):
            index = int(generator.integers(len(first[2])))
            first, result_a = plain.step(first[2][index])
            second, result_b = shaped.step(second[2][index])
            assert result_a.raw_reward == pytest.approx(result_b.raw_reward)
            if result_a.done or result_b.done:
                break

    def test_time_costs_reward(self):
        env = build_env()
        env.reset(default_request(), memory_size=4)
        _, result = env.step(Action(ActionKind.WAIT))
        assert result.raw_reward < 0

    def test_a_better_pair_pays_more(self):
        rewards = []
        for fidelity in (0.80, 0.99):
            env = build_env()
            env.reset(Request("0", "3", min_fidelity=0.75, deadline_s=0.1), memory_size=4)
            pair = EntangledPair.new("0", "3", bds.werner(fidelity), 0.0, hops=3,
                                     distance_km=60.0)
            env.state.add_pair(pair)
            _, result = env.step(Action(ActionKind.DELIVER, pair_ids=(pair.pair_id,)))
            rewards.append(result.raw_reward)
        assert rewards[1] > rewards[0]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_unknown_purification_protocol_is_rejected(self):
        with pytest.raises(ValueError):
            build_config(purification="wishful_thinking")

    def test_edges_without_a_length_are_rejected(self):
        import networkx as nx

        graph = nx.Graph()
        graph.add_edge(0, 1)
        with pytest.raises(ValueError):
            build_config(graph=graph)

    def test_link_probability_falls_with_span_length(self):
        short = build_config(graph=scenario("chain", num_nodes=2, span_km=10))
        long = build_config(graph=scenario("chain", num_nodes=2, span_km=60))
        assert (list(long.link_probability.values())[0]
                < list(short.link_probability.values())[0])
