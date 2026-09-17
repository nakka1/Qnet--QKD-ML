"""The hybrid allocation layer.

As in phase 3, the tests that matter most are about **what the allocator is
allowed to know**. This environment holds the true QBER, the true drift and the
future of both, and there are three separate places where letting any of them
leak would produce a controller that is excellent in simulation and impossible
to deploy: the observation, the forecaster's training data, and the forecaster's
inputs at inference. Each has a test here.

The second group is about the bookkeeping that is easy to get quietly wrong: a
reconfiguration queue that applies decisions at the wrong time, a demand that is
served twice, key drawn from a pool that did not have it, a deadline that passes
without anyone noticing.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.allocation import (AllocationConfig, ClassicalFirst,
                                       HybridAllocationEnv, PersistenceForecaster,
                                       PredictiveAllocator, QuantumFirst,
                                       ReactiveThreshold, StaticAdmission,
                                       build_demands, evaluate, run_episode,
                                       shortest_paths, visibility_fidelity)
from qnet_autopilot.allocation.forecast import (OracleForecaster, build_windows,
                                                drift_estimate, fit_ridge_forecaster)
from qnet_autopilot.allocation.state import Demand
from qnet_autopilot.compensation.residual import COMPENSATED, ResidualDriftModel
from qnet_autopilot.network import scenario


def small_graph():
    """A four-node chain: short enough to run fast, long enough to have paths."""
    return scenario("chain", num_nodes=4, span_km=40.0)


def quick_config(**overrides) -> AllocationConfig:
    parameters = dict(control_period_s=5.0, duration_s=200.0,
                      reconfiguration_latency_s=0.0,
                      classical_power_full_w=3.0e-4)
    parameters.update(overrides)
    return AllocationConfig(**parameters)


def one_demand(service: str, path, volume: float, deadline_s: float,
               arrival_s: float = 0.0, min_fidelity: float = float("nan")) -> Demand:
    return Demand(demand_id=0, service=service, source="a", destination="b",
                  path=tuple(path), volume=volume, arrival_s=arrival_s,
                  deadline_s=deadline_s, priority=2, min_fidelity=min_fidelity)


class TestConfig:

    @pytest.mark.parametrize("bad", [
        dict(control_period_s=0.0),
        dict(duration_s=1.0),
        dict(estimation_fraction=0.0),
        dict(estimation_fraction=1.0),
        dict(admission_levels=()),
        dict(admission_levels=(0.0, 1.5)),
        dict(degraded_fraction=1.5),
        dict(reconfiguration_latency_s=-5.0),
    ])
    def test_rejects_impossible_settings(self, bad):
        with pytest.raises(ValueError):
            quick_config(**bad)

    def test_latency_must_be_a_whole_number_of_control_periods(self):
        """Otherwise a decision lands mid-period and the queue silently rounds.

        The same failure phase 3 hit with its control loop, where it looked like
        the predictor simply being worse at short latencies.
        """
        with pytest.raises(ValueError, match="whole number of control periods"):
            quick_config(control_period_s=5.0, reconfiguration_latency_s=7.0)
        assert quick_config(control_period_s=5.0,
                            reconfiguration_latency_s=15.0).latency_steps == 3


class TestResidualDrift:

    def test_rejects_impossible_parameters(self):
        for bad in (dict(rms_angle_rad=-0.1), dict(correlation_time_s=0.0),
                    dict(recovery_time_s=-1.0), dict(jump_rate_per_s=-1.0)):
            with pytest.raises(ValueError):
                ResidualDriftModel(**bad)

    def test_it_mean_reverts_rather_than_diffusing(self):
        """The property that distinguishes a compensated link from a bare one.

        A random walk's variance grows without bound; a closed loop's residual
        does not. Comparing the spread of the first and last thirds of a long
        run separates the two.
        """
        model = ResidualDriftModel(jump_rate_per_s=0.0)
        trace = model.simulate(20_000.0, 5.0, rng=0)
        third = len(trace) // 3
        early = float(np.std(trace[:third]))
        late = float(np.std(trace[-third:]))
        assert late == pytest.approx(early, rel=0.35)

    def test_the_qber_distribution_is_skewed_right(self):
        """Two squared Gaussian components, so the mean sits above the median."""
        trace = ResidualDriftModel(jump_rate_per_s=0.0).simulate(50_000.0, 5.0, rng=1)
        assert float(np.mean(trace)) > float(np.median(trace))
        assert float(np.percentile(trace, 99)) > 4 * float(np.median(trace))

    def test_the_exact_discretisation_gets_the_autocorrelation_right(self):
        """An Euler step would be close, and close would change forecastability.

        For the OU components the lag-one correlation is ``exp(-dt/tau)``; the
        QBER is their squared norm, so its correlation is the square of that.
        """
        tau, dt = 150.0, 5.0
        model = ResidualDriftModel(correlation_time_s=tau, jump_rate_per_s=0.0)
        trace = model.simulate(200_000.0, dt, rng=2)
        centred = trace - trace.mean()
        measured = float(np.corrcoef(centred[:-1], centred[1:])[0, 1])
        assert measured == pytest.approx(np.exp(-dt / tau) ** 2, rel=0.05)

    def test_a_bigger_residual_means_a_worse_link(self):
        quiet = ResidualDriftModel(rms_angle_rad=0.08).simulate(20_000.0, 5.0, rng=3)
        loud = ResidualDriftModel(rms_angle_rad=0.20).simulate(20_000.0, 5.0, rng=3)
        assert float(np.mean(loud)) > float(np.mean(quiet))

    def test_jumps_decay_back(self):
        """A cable jump is a transient, not a new operating point."""
        model = ResidualDriftModel(rms_angle_rad=1e-6, jump_rate_per_s=1.0 / 200.0,
                                   jump_scale_rad=0.5, recovery_time_s=20.0)
        trace = model.simulate(4000.0, 5.0, rng=4)
        peak = int(np.argmax(trace))
        assert trace[peak] > 10 * float(np.median(trace))
        settled = trace[min(peak + 30, len(trace) - 1)]
        assert settled < 0.2 * trace[peak]

    def test_the_calibration_matches_what_phase_three_measured(self):
        """COMPENSATED is meant to be phase 3's residual, not a free parameter."""
        trace = np.concatenate([COMPENSATED.simulate(1800.0, 5.0, rng=seed)
                                for seed in range(30)])
        assert float(np.mean(trace)) == pytest.approx(0.0077, abs=0.003)
        assert float(np.percentile(trace, 95)) == pytest.approx(0.0211, abs=0.008)


class TestEnvironment:

    def test_link_profiles_follow_the_graph(self):
        graph = small_graph()
        env = HybridAllocationEnv(graph, [], quick_config())
        assert env.num_links == graph.number_of_edges()
        assert all(link.length_km == 40.0 for link in env.links)
        # A longer span is slower and fainter, both of which matter to Raman.
        far = HybridAllocationEnv(scenario("chain", num_nodes=2, span_km=120.0), [],
                                  quick_config())
        assert far.links[0].sifted_rate_hz < env.links[0].sifted_rate_hz
        assert far.links[0].signal_counts_per_gate < env.links[0].signal_counts_per_gate

    def test_the_same_seed_gives_the_same_plant(self):
        """Every comparison in phase 4 is paired, which requires exactly this."""
        env = HybridAllocationEnv(small_graph(), [], quick_config())
        env.reset(seed=3)
        first = env._polarisation.copy(), env.degraded_links.copy()
        env.reset(seed=3)
        assert np.array_equal(env._polarisation, first[0])
        assert np.array_equal(env.degraded_links, first[1])

        env.reset(seed=4)
        assert not np.array_equal(env._polarisation, first[0])

    def test_links_drift_independently(self):
        """Shared drift would let a policy predict one link from another free."""
        env = HybridAllocationEnv(scenario("ring", num_nodes=6, span_km=40.0), [],
                                  quick_config(duration_s=4000.0,
                                               degraded_fraction=0.0))
        env.reset(seed=5)
        correlations = np.corrcoef(env._polarisation.T)
        off_diagonal = correlations[~np.eye(len(correlations), dtype=bool)]
        assert float(np.max(np.abs(off_diagonal))) < 0.4

    def test_admission_must_be_a_valid_vector(self):
        env = HybridAllocationEnv(small_graph(), [], quick_config())
        env.reset(seed=0)
        with pytest.raises(ValueError, match="shape"):
            env.step(np.ones(env.num_links + 1))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            env.step(np.full(env.num_links, 1.5))

    def test_classical_traffic_raises_the_qber_and_costs_key(self):
        """The coupling, in one step: throughput is bought out of the key.

        A classical demand has to exist for anything to be carried —
        ``classical_bits`` counts bits actually delivered, not capacity that was
        merely admitted, and an admitted link with nothing to send carries
        nothing.
        """
        graph = small_graph()
        flow = one_demand("classical", tuple(range(3)), volume=1e18,
                          deadline_s=190.0)
        env = HybridAllocationEnv(graph, [flow], quick_config())

        env.reset(seed=0)
        quiet = env.step(np.zeros(env.num_links))
        env.reset(seed=0)
        loud = env.step(np.ones(env.num_links))

        assert float(loud.key_bits.sum()) < float(quiet.key_bits.sum())
        assert float(quiet.classical_bits.sum()) == 0.0
        assert float(loud.classical_bits.sum()) > 0.0

    def test_an_idle_plant_still_shows_the_optics_error(self):
        env = HybridAllocationEnv(small_graph(), [],
                                  quick_config(degraded_fraction=0.0))
        env.reset(seed=0)
        env.step(np.zeros(env.num_links))
        assert np.all(env.state.qber > 0.0)

    def test_key_accumulates_in_a_pool(self):
        """Key is a stock, which is what makes banking a good window worth it."""
        env = HybridAllocationEnv(small_graph(), [], quick_config())
        env.reset(seed=0)
        pools = []
        for _ in range(5):
            env.step(np.zeros(env.num_links))
            pools.append(env.state.key_pool_bits.copy())
        assert np.all(np.diff(np.array(pools), axis=0) >= 0)


class TestReconfigurationQueue:

    def test_a_decision_takes_effect_after_the_latency_and_not_before(self):
        env = HybridAllocationEnv(small_graph(), [],
                                  quick_config(reconfiguration_latency_s=15.0))
        env.reset(seed=0)
        assert env.config.latency_steps == 3

        for step in range(3):
            result = env.step(np.ones(env.num_links))
            assert np.all(result.observation.admission == 0.0), \
                f"step {step} should still be running the initial silence"
        result = env.step(np.ones(env.num_links))
        assert np.all(result.observation.admission == 1.0)

    def test_zero_latency_applies_the_decision_immediately(self):
        env = HybridAllocationEnv(small_graph(), [],
                                  quick_config(reconfiguration_latency_s=0.0))
        env.reset(seed=0)
        result = env.step(np.ones(env.num_links))
        assert np.all(result.observation.admission == 1.0)

    def test_the_pending_queue_is_visible_and_correctly_sized(self):
        env = HybridAllocationEnv(small_graph(), [],
                                  quick_config(reconfiguration_latency_s=10.0))
        observation = env.reset(seed=0)
        assert observation.pending_admission.shape == (2, env.num_links)
        result = env.step(np.full(env.num_links, 0.5))
        assert np.all(result.observation.pending_admission[-1] == 0.5)

    def test_latency_delays_a_reaction_and_therefore_costs_key(self):
        """The reason the latency is in the model at all."""
        graph = small_graph()
        demands = build_demands(graph, duration_s=200.0, seed=0)
        served = []
        for latency in (0.0, 60.0):
            env = HybridAllocationEnv(graph, demands,
                                      quick_config(reconfiguration_latency_s=latency))
            served.append(run_episode(env, ReactiveThreshold(), seed=0).served["qkd"])
        assert served[1] <= served[0] + 1e-9


class TestDemandService:

    def test_a_qkd_demand_draws_from_every_hop(self):
        """Trusted-node relay: an end-to-end bit costs a bit on each link."""
        graph = small_graph()
        demand = one_demand("qkd", (0, 1, 2), volume=1e4, deadline_s=150.0)
        env = HybridAllocationEnv(graph, [demand], quick_config())
        env.reset(seed=0)

        before = env.state.key_pool_bits.copy()
        result = env.step(np.zeros(env.num_links))
        drawn = before + result.key_bits - env.state.key_pool_bits
        assert result.qkd_delivered_bits > 0
        assert np.allclose(drawn[[0, 1, 2]], result.qkd_delivered_bits)

    def test_no_demand_is_served_beyond_its_volume(self):
        graph = small_graph()
        demand = one_demand("qkd", (0,), volume=500.0, deadline_s=190.0)
        env = HybridAllocationEnv(graph, [demand], quick_config())
        run_episode(env, StaticAdmission(0.0), seed=0)
        assert env.state.progress[0].delivered <= 500.0 + 1e-6
        assert env.state.progress[0].done

    def test_a_demand_that_misses_its_deadline_expires_once(self):
        graph = small_graph()
        demand = one_demand("qkd", (0,), volume=1e12, deadline_s=50.0)
        env = HybridAllocationEnv(graph, [demand], quick_config())
        env.reset(seed=0)

        expirations = 0
        done = False
        while not done:
            result = env.step(np.zeros(env.num_links))
            expirations += len(result.expired)
            done = result.done
        assert expirations == 1
        assert env.state.progress[0].expired

    def test_an_expired_demand_stops_consuming_capacity(self):
        graph = small_graph()
        demand = one_demand("qkd", (0,), volume=1e12, deadline_s=20.0)
        env = HybridAllocationEnv(graph, [demand], quick_config())
        env.reset(seed=0)
        for _ in range(8):
            result = env.step(np.zeros(env.num_links))
        assert result.qkd_delivered_bits == 0.0
        assert env.state.key_pool_bits[0] > 0

    def test_a_demand_that_has_not_arrived_is_not_served(self):
        graph = small_graph()
        demand = one_demand("qkd", (0,), volume=1e3, deadline_s=190.0,
                            arrival_s=100.0)
        env = HybridAllocationEnv(graph, [demand], quick_config())
        env.reset(seed=0)
        result = env.step(np.zeros(env.num_links))
        assert result.qkd_delivered_bits == 0.0

    def test_entanglement_below_its_fidelity_threshold_is_not_served(self):
        """A pair under the requested fidelity is not worth distributing."""
        graph = small_graph()
        impossible = one_demand("entanglement", (0, 1), volume=5.0,
                                deadline_s=190.0, min_fidelity=0.999)
        env = HybridAllocationEnv(graph, [impossible], quick_config())
        result = run_episode(env, StaticAdmission(0.0), seed=0)
        assert result.pairs_delivered == 0.0

        easy = one_demand("entanglement", (0, 1), volume=5.0, deadline_s=190.0,
                          min_fidelity=0.70)
        env = HybridAllocationEnv(graph, [easy], quick_config())
        assert run_episode(env, StaticAdmission(0.0), seed=0).pairs_delivered > 0

    def test_classical_flows_are_capped_by_the_admitted_capacity(self):
        graph = small_graph()
        demand = one_demand("classical", (0,), volume=1e18, deadline_s=190.0)
        env = HybridAllocationEnv(graph, [demand], quick_config())
        env.reset(seed=0)
        result = env.step(np.full(env.num_links, 0.5))
        expected = 0.5 * env.links[0].classical_capacity_bps * 5.0
        assert result.classical_delivered_bits == pytest.approx(expected, rel=1e-9)

    def test_visibility_fidelity_spans_the_right_range(self):
        assert float(visibility_fidelity(0.0)) == pytest.approx(1.0)
        assert float(visibility_fidelity(0.5)) == pytest.approx(0.25)
        assert float(visibility_fidelity(0.25)) == pytest.approx(0.625)
        qber = np.linspace(0.0, 0.5, 20)
        assert np.all(np.diff(visibility_fidelity(qber)) < 0)


class TestObservationModel:

    def test_the_policy_never_sees_the_true_qber(self):
        """A binomial estimate, not the number the environment scores with."""
        env = HybridAllocationEnv(small_graph(), [], quick_config())
        env.reset(seed=0)
        result = env.step(np.zeros(env.num_links))
        assert not np.allclose(result.observation.measured_qber, env.state.qber)

    def test_the_estimate_is_unbiased_and_tightens_with_the_sample(self):
        env = HybridAllocationEnv(scenario("chain", num_nodes=2, span_km=40.0), [],
                                  quick_config(degraded_fraction=0.0))
        env.reset(seed=0)
        truth = env._true_qber(np.zeros(1), 0)
        draws = np.array([env._measure(truth)[0] for _ in range(400)])
        assert float(np.mean(draws)) == pytest.approx(float(truth[0]), rel=0.05)

        noisy = HybridAllocationEnv(
            scenario("chain", num_nodes=2, span_km=40.0), [],
            quick_config(estimation_fraction=1e-4, degraded_fraction=0.0))
        noisy.reset(seed=0)
        loose = np.array([noisy._measure(truth)[0] for _ in range(400)])
        assert float(np.std(loose)) > float(np.std(draws))

    def test_a_long_span_measures_itself_worse_than_a_short_one(self):
        """Fewer sifted bits, so a noisier estimate exactly where it hurts."""
        spreads = []
        for length in (30.0, 130.0):
            env = HybridAllocationEnv(scenario("chain", num_nodes=2, span_km=length),
                                      [], quick_config(degraded_fraction=0.0))
            env.reset(seed=0)
            truth = env._true_qber(np.zeros(1), 0)
            spreads.append(float(np.std([env._measure(truth)[0] for _ in range(300)])))
        assert spreads[1] > spreads[0]

    def test_the_measurement_includes_the_policys_own_interference(self):
        """Which is why a forecaster has to subtract it before fitting."""
        env = HybridAllocationEnv(small_graph(), [],
                                  quick_config(estimation_fraction=0.5))
        env.reset(seed=0)
        quiet = env.step(np.zeros(env.num_links)).observation.measured_qber.mean()
        env.reset(seed=0)
        loud = env.step(np.ones(env.num_links)).observation.measured_qber.mean()
        assert loud > quiet

    def test_the_reported_raman_matches_what_admission_actually_added(self):
        env = HybridAllocationEnv(small_graph(), [], quick_config())
        env.reset(seed=0)
        assert np.allclose(env.raman_qber(np.zeros(env.num_links)), 0.0, atol=1e-12)
        assert np.all(env.raman_qber(np.ones(env.num_links)) > 0)

    def test_drift_estimate_removes_the_controllers_own_contribution(self):
        """Feed the raw measurement to a model and it predicts its own actions."""
        env = HybridAllocationEnv(small_graph(), [],
                                  quick_config(estimation_fraction=0.9,
                                               degraded_fraction=0.0))
        env.reset(seed=0)
        admission = np.full(env.num_links, 1.0)
        env.step(admission)
        observation = env.step(admission).observation

        raw = observation.measured_qber
        stripped = drift_estimate(raw, observation.raman_qber_estimate,
                                  env.coexistence.optical_error_rate)
        assert np.all(stripped < raw)
        assert np.all(stripped >= 0.0)


class TestForecasters:

    def test_persistence_repeats_the_present(self):
        forecaster = PersistenceForecaster(horizon=4)
        drift = np.array([0.01, 0.02])
        prediction = forecaster.predict(drift, 0)
        assert prediction.shape == (2, 4)
        assert np.allclose(prediction, drift[:, None])

    def test_the_oracle_returns_the_actual_future(self):
        env = HybridAllocationEnv(small_graph(), [], quick_config())
        env.reset(seed=0)
        oracle = OracleForecaster(horizon=3)
        oracle.reset(env)
        prediction = oracle.predict(np.zeros(env.num_links), 5)
        assert np.allclose(prediction, env._polarisation[6:9].T)

    def test_the_oracle_refuses_to_predict_before_it_is_bound(self):
        with pytest.raises(RuntimeError, match="reset"):
            OracleForecaster(horizon=3).predict(np.zeros(2), 0)

    def test_windows_line_up_features_with_their_own_future(self):
        trace = np.arange(40.0).reshape(20, 2)
        features, targets = build_windows([trace], window=3, horizon=2)
        assert features.shape[1] == 3 and targets.shape[1] == 2
        # Row 0 is link 0's first window: values 0, 2, 4, then 6, 8 ahead.
        assert np.allclose(features[0], [0.0, 2.0, 4.0])
        assert np.allclose(targets[0], [6.0, 8.0])

    def test_windows_refuse_a_trace_that_is_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            build_windows([np.zeros((4, 2))], window=3, horizon=5)

    def test_the_fitted_model_beats_persistence_on_a_mean_reverting_process(self):
        """The premise of the whole predictive policy, checked directly.

        On an OU process persistence is biased towards the current value while
        the truth decays towards the mean, so a fitted linear model must win —
        and by more at longer horizons, which is the shape that says the model
        learned mean reversion rather than a constant offset.
        """
        model = ResidualDriftModel(correlation_time_s=100.0, jump_rate_per_s=0.0)
        traces = [np.stack([model.simulate(3000.0, 5.0, rng=seed * 10 + link)
                            for link in range(3)], axis=1) for seed in range(6)]
        features, targets = build_windows(traces[:4], window=4, horizon=6)
        forecaster = fit_ridge_forecaster(features, targets, window=4, horizon=6)

        v_features, v_targets = build_windows(traces[4:], window=4, horizon=6)
        design = np.hstack([v_features, np.ones((len(v_features), 1))])
        fitted = np.mean((np.maximum(design @ forecaster.coefficients, 0.0)
                          - v_targets) ** 2, axis=0)
        persistence = np.mean((v_features[:, -1:] - v_targets) ** 2, axis=0)

        assert np.all(fitted <= persistence * 1.02)
        assert fitted[-1] / persistence[-1] < fitted[0] / persistence[0]

    def test_the_penalty_is_scale_free(self):
        """An absolute penalty against QBER-sized features shrinks the fit away.

        Features here are of order 0.005, so an unscaled ``alpha`` of 1 would
        outweigh the data term entirely and leave a model worse than the
        persistence it is supposed to beat — silently.
        """
        model = ResidualDriftModel(correlation_time_s=100.0, jump_rate_per_s=0.0)
        traces = [np.stack([model.simulate(3000.0, 5.0, rng=seed * 10 + link)
                            for link in range(3)], axis=1) for seed in range(4)]
        features, targets = build_windows(traces, window=4, horizon=6)

        def predictions(forecaster, rows):
            design = np.hstack([rows, np.ones((len(rows), 1))])
            return design @ forecaster.coefficients

        probe = features[:200]
        default = fit_ridge_forecaster(features, targets, window=4, horizon=6)
        tiny = fit_ridge_forecaster(features, targets, window=4, horizon=6,
                                    alpha=1e-9)
        # The default penalty shrinks the fit by a fraction of a percent, which
        # is the point of a numerical regulariser; an unscaled one would shrink
        # it to the intercept.
        typical = float(np.mean(np.abs(predictions(tiny, probe))))
        assert np.allclose(predictions(default, probe), predictions(tiny, probe),
                           rtol=0.0, atol=0.05 * typical)

        # The same relationship expressed in different units must fit the same
        # way, which is exactly what an unscaled penalty would break.
        scaled = fit_ridge_forecaster(features * 100.0, targets * 100.0,
                                      window=4, horizon=6)
        assert np.allclose(predictions(scaled, probe * 100.0) / 100.0,
                           predictions(default, probe), rtol=1e-9, atol=1e-12)

    def test_fitting_rejects_shapes_that_disagree_with_the_declared_layout(self):
        features, targets = build_windows([np.zeros((30, 2))], window=3, horizon=4)
        with pytest.raises(ValueError, match="features"):
            fit_ridge_forecaster(features, targets, window=5, horizon=4)
        with pytest.raises(ValueError, match="targets"):
            fit_ridge_forecaster(features, targets, window=3, horizon=9)

    def test_the_ridge_forecaster_falls_back_before_its_window_fills(self):
        features, targets = build_windows([np.zeros((60, 2)) + 0.01],
                                          window=4, horizon=3)
        forecaster = fit_ridge_forecaster(features, targets, window=4, horizon=3)
        forecaster.reset(None)
        drift = np.array([0.03, 0.04])
        first = forecaster.predict(drift, 0)
        assert np.allclose(first, drift[:, None]), "should persist, not guess"


class TestPolicies:

    @pytest.mark.parametrize("policy", [
        ClassicalFirst(), QuantumFirst(), StaticAdmission(0.25),
        ReactiveThreshold(), PredictiveAllocator(),
    ])
    def test_every_policy_returns_a_legal_admission_vector(self, policy):
        graph = small_graph()
        demands = build_demands(graph, duration_s=200.0, seed=0)
        env = HybridAllocationEnv(graph, demands, quick_config())
        observation = env.reset(seed=0)
        policy.reset(env)
        for step in range(6):
            action = policy.act(observation, step)
            assert action.shape == (env.num_links,)
            assert np.all(action >= 0.0) and np.all(action <= 1.0)
            observation = env.step(action).observation

    def test_the_two_extremes_bracket_everything_else(self):
        graph = small_graph()
        demands = build_demands(graph, duration_s=400.0, seed=0)
        env = HybridAllocationEnv(graph, demands, quick_config(duration_s=400.0))

        classical = run_episode(env, ClassicalFirst(), seed=0)
        quantum = run_episode(env, QuantumFirst(), seed=0)
        assert quantum.served["qkd"] >= classical.served["qkd"]
        assert classical.served["classical"] >= quantum.served["classical"]

    def test_static_admission_rejects_a_level_outside_the_range(self):
        with pytest.raises(ValueError):
            StaticAdmission(1.5)

    def test_the_reactive_threshold_needs_an_ordered_dead_band(self):
        with pytest.raises(ValueError, match="low <= high"):
            ReactiveThreshold(low=0.09, high=0.02)

    def test_the_reactive_controller_throttles_a_link_it_sees_failing(self):
        graph = scenario("chain", num_nodes=2, span_km=120.0)
        demand = one_demand("qkd", (0,), volume=1e9, deadline_s=380.0)
        env = HybridAllocationEnv(graph, [demand], quick_config(duration_s=400.0))
        result = run_episode(env, ReactiveThreshold(low=0.02, high=0.04), seed=0)
        assert result.mean_admission < 0.5

    def test_the_predictive_allocator_rejects_a_nonsense_coverage_target(self):
        with pytest.raises(ValueError, match="coverage_target"):
            PredictiveAllocator(coverage_target=0.0)

    def test_coverage_target_moves_the_operating_point_monotonically(self):
        """The knob that traces the frontier has to actually be a knob."""
        graph = small_graph()
        demands = build_demands(graph, duration_s=400.0, seed=0)
        env = HybridAllocationEnv(graph, demands, quick_config(duration_s=400.0))

        classical = [run_episode(env, PredictiveAllocator(coverage_target=target),
                                 seed=0).served["classical"]
                     for target in (0.1, 1.0, 5.0)]
        assert classical[0] >= classical[1] >= classical[2]

    def test_it_leaves_idle_links_alone(self):
        """Throttling a link with no quantum work outstanding is a pure loss."""
        graph = small_graph()
        env = HybridAllocationEnv(graph, [], quick_config())
        observation = env.reset(seed=0)
        policy = PredictiveAllocator()
        policy.reset(env)
        assert np.allclose(policy.act(observation, 0), 1.0)


class TestRollout:

    def test_service_metrics_stay_in_range(self):
        graph = small_graph()
        demands = build_demands(graph, duration_s=400.0, seed=0)
        env = HybridAllocationEnv(graph, demands, quick_config(duration_s=400.0))
        result = run_episode(env, StaticAdmission(0.5), seed=0)
        for service, value in result.served.items():
            assert np.isnan(value) or 0.0 <= value <= 1.0 + 1e-9

    def test_demands_whose_deadline_falls_past_the_episode_are_not_scored(self):
        """They were never given their full chance; counting them is a lottery."""
        graph = small_graph()
        inside = one_demand("qkd", (0,), volume=100.0, deadline_s=100.0)
        outside = Demand(demand_id=1, service="qkd", source="a", destination="b",
                         path=(0,), volume=1e12, arrival_s=0.0, deadline_s=1e9,
                         priority=2, min_fidelity=float("nan"))
        env = HybridAllocationEnv(graph, [inside, outside], quick_config())
        result = run_episode(env, StaticAdmission(0.0), seed=0)
        assert result.served["qkd"] == pytest.approx(1.0)

    def test_evaluation_is_paired_across_policies(self):
        graph = small_graph()
        demands = build_demands(graph, duration_s=200.0, seed=0)
        env = HybridAllocationEnv(graph, demands, quick_config())
        result = evaluate(env, [ClassicalFirst(), QuantumFirst()], seeds=(0, 1, 2))

        assert len(result.frame) == 6
        difference = result.paired_difference("quântico primeiro",
                                              "clássico primeiro", "served_qkd")
        assert difference["wins"] + difference["losses"] + difference["ties"] == 3
        assert difference["mean"] >= 0

    def test_paired_difference_refuses_a_policy_it_never_ran(self):
        graph = small_graph()
        env = HybridAllocationEnv(graph, [], quick_config())
        result = evaluate(env, [ClassicalFirst()], seeds=(0,))
        with pytest.raises(KeyError):
            result.paired_difference("nada", "clássico primeiro")

    def test_a_policy_is_reproducible_at_a_seed(self):
        graph = small_graph()
        demands = build_demands(graph, duration_s=200.0, seed=0)
        env = HybridAllocationEnv(graph, demands, quick_config())
        first = run_episode(env, PredictiveAllocator(), seed=7)
        second = run_episode(env, PredictiveAllocator(), seed=7)
        assert first.served["qkd"] == second.served["qkd"]
        assert np.array_equal(first.admission, second.admission)


class TestScenario:

    def test_paths_are_resolved_to_link_indices(self):
        graph = small_graph()
        paths, index = shortest_paths(graph)
        assert len(index) == graph.number_of_edges()
        route = paths[(0, 3)]
        assert len(route) == 3
        assert set(route) <= set(range(graph.number_of_edges()))

    def test_demands_carry_a_route_and_are_sorted_by_arrival(self):
        demands = build_demands(small_graph(), duration_s=400.0, seed=0)
        assert demands
        arrivals = [demand.arrival_s for demand in demands]
        assert arrivals == sorted(arrivals)
        assert all(demand.path for demand in demands)
        assert all(demand.deadline_s > demand.arrival_s for demand in demands)

    def test_load_scale_multiplies_volume_without_touching_deadlines(self):
        """Otherwise the load sweep also sweeps urgency and means nothing."""
        light = build_demands(small_graph(), duration_s=400.0, seed=0, load_scale=1.0)
        heavy = build_demands(small_graph(), duration_s=400.0, seed=0, load_scale=3.0)
        assert len(light) == len(heavy)
        for left, right in zip(light, heavy):
            assert right.volume == pytest.approx(3.0 * left.volume)
            assert right.deadline_s == pytest.approx(left.deadline_s)
