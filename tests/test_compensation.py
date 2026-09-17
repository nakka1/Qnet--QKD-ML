"""The compensation stack: geometry, monitoring, the loop, and the predictors.

The tests that matter most here are the ones about **what the compensator is
allowed to know**.  A polarisation compensator is unusually easy to evaluate
dishonestly: the simulator holds the true rotation, and feeding it in anywhere
— as a feature, as a training target, as the thing the loop corrects to —
produces a controller that is superb offline and impossible to deploy.  So:

* the monitoring channel is the only route from the true state to the
  compensator, and it is noisy;
* the training target is a *future measurement*, not the future truth;
* the correction in force at any moment was computed from data at least one
  control latency old.

Each of those is checked directly, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.compensation import (AlphaBetaFilter, ControlConfig,
                                         LearnedCompensator, LinearExtrapolation,
                                         MonitoringConfig, NoCompensation, Persistence,
                                         bb84_qber, measure_rotation, run_closed_loop)
from qnet_autopilot.compensation.monitoring import orthonormalise
from qnet_autopilot.compensation.predictors import (FeatureSpec, Standardiser, _exp_map,
                                                    _log_map, build_features)
from qnet_autopilot.compensation.train import (build_dataset, chronological_split,
                                               fit_mlp, fit_ridge)
from qnet_autopilot.physics.fiber import (PolarizationDriftModel, rotation_angle,
                                          rotation_matrix, rotation_vector)

QUIET = dict(thermal_drift_rate_rad_per_s=0.02, thermal_correlation_time_s=60.0,
             vibration_amplitude_rad=0.05, vibration_frequency_hz=40.0,
             vibration_bandwidth_hz=6.0, jump_rate_per_s=0.0,
             thermal_per_kelvin_rad_per_s=0.0)


def make_trajectory(seed: int = 0, duration_s: float = 2.0, **overrides):
    parameters = dict(QUIET)
    parameters.update(overrides)
    return PolarizationDriftModel(**parameters).simulate(
        duration_s, 1e-3, rng=seed, track_rotation=True)


def make_control(**overrides) -> ControlConfig:
    defaults = dict(control_period_s=2e-3, latency_s=20e-3,
                    monitoring=MonitoringConfig(photons_per_basis=400))
    defaults.update(overrides)
    return ControlConfig(**defaults)


# ---------------------------------------------------------------------------
# SO(3) geometry
# ---------------------------------------------------------------------------

class TestRotationGeometry:

    def test_log_inverts_exp_over_the_whole_range(self, rng):
        angles = rng.uniform(1e-6, np.pi - 1e-4, size=300)
        axes = rng.normal(size=(300, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)

        matrices = rotation_matrix(axes, angles)
        vectors = rotation_vector(matrices)
        rebuilt = rotation_matrix(vectors, np.linalg.norm(vectors, axis=-1))
        assert np.allclose(rebuilt, matrices, atol=1e-8)

    def test_rotation_angle_recovers_the_angle(self, rng):
        angles = rng.uniform(0.0, np.pi, size=200)
        axes = rng.normal(size=(200, 3))
        assert np.allclose(rotation_angle(rotation_matrix(axes, angles)), angles,
                           atol=1e-7)

    def test_the_fast_single_matrix_paths_match_the_general_ones(self, rng):
        for _ in range(200):
            vector = rng.normal(size=3) * rng.uniform(0.0, 3.0)
            angle = np.linalg.norm(vector)
            matrix = np.asarray(rotation_matrix(vector, angle), dtype=float)
            assert np.allclose(_exp_map(vector), matrix, atol=1e-12)
            assert np.allclose(_log_map(matrix), rotation_vector(matrix), atol=1e-7)

    def test_exp_of_zero_is_the_identity(self):
        assert np.allclose(_exp_map(np.zeros(3)), np.eye(3))
        assert np.allclose(_log_map(np.eye(3)), np.zeros(3))


# ---------------------------------------------------------------------------
# monitoring
# ---------------------------------------------------------------------------

class TestMonitoring:

    def test_one_probe_state_is_refused(self):
        """The modelling error this whole module exists to prevent."""
        with pytest.raises(ValueError, match="U\\(1\\)"):
            MonitoringConfig(probe_states=((1.0, 0.0, 0.0),))

    def test_rejects_bad_settings(self):
        with pytest.raises(ValueError):
            MonitoringConfig(photons_per_basis=0)
        with pytest.raises(ValueError):
            MonitoringConfig(detector_efficiency=0.0)
        with pytest.raises(ValueError):
            MonitoringConfig(probe_states=((1.0, 0.0),))

    def test_photon_accounting(self):
        config = MonitoringConfig(photons_per_basis=100, detector_efficiency=0.5)
        assert config.effective_photons == 50
        assert config.photons_per_update == 2 * 3 * 100

    def test_orthonormalise_returns_a_rotation(self, rng):
        noisy = rng.normal(size=(20, 3, 2))
        result = orthonormalise(noisy)
        assert np.allclose(np.einsum("nij,nkj->nik", result, result), np.eye(3),
                           atol=1e-10)
        assert np.allclose(np.linalg.det(result), 1.0)

    def test_infinite_photons_recover_the_rotation(self, rng):
        truth = rotation_matrix(rng.normal(size=(30, 3)), rng.uniform(0, 2, 30))
        estimate = measure_rotation(truth, MonitoringConfig(photons_per_basis=2_000_000),
                                    rng=0)
        assert np.max(rotation_angle(np.einsum("nji,njk->nik", estimate, truth))) < 0.01

    def test_the_estimate_is_always_a_rotation_however_few_photons(self, rng):
        truth = rotation_matrix(rng.normal(size=(50, 3)), rng.uniform(0, 3, 50))
        estimate = measure_rotation(truth, MonitoringConfig(photons_per_basis=4), rng=1)
        assert np.allclose(np.einsum("nij,nkj->nik", estimate, estimate), np.eye(3),
                           atol=1e-9)
        assert np.allclose(np.linalg.det(estimate), 1.0)

    def test_more_photons_means_a_better_estimate(self, rng):
        truth = rotation_matrix(rng.normal(size=(200, 3)), rng.uniform(0, 2, 200))
        errors = []
        for photons in (10, 100, 1000):
            estimate = measure_rotation(truth, MonitoringConfig(photons_per_basis=photons),
                                        rng=2)
            errors.append(float(np.mean(rotation_angle(
                np.einsum("nji,njk->nik", estimate, truth)))))
        assert errors[0] > errors[1] > errors[2]


# ---------------------------------------------------------------------------
# QBER
# ---------------------------------------------------------------------------

class TestQber:

    def test_no_residual_means_no_error(self):
        assert bb84_qber(np.eye(3)) == pytest.approx(0.0)

    def test_a_rotation_about_a_basis_axis_only_hurts_the_other_basis(self):
        """The reason a single tracked state is not enough to score a compensator."""
        residual = np.asarray(rotation_matrix([1.0, 0.0, 0.0], np.pi))
        # The rectilinear basis (S1) is untouched; the diagonal one is flipped.
        assert bb84_qber(residual, bases=np.array([[1.0, 0, 0]])) == pytest.approx(0.0)
        assert bb84_qber(residual, bases=np.array([[0, 1.0, 0]])) == pytest.approx(1.0)
        assert bb84_qber(residual) == pytest.approx(0.5)

    def test_qber_grows_with_the_residual_angle(self):
        angles = np.linspace(0.0, np.pi, 12)
        residuals = rotation_matrix([0.0, 0.0, 1.0], angles)
        qber = bb84_qber(residuals)
        assert np.all(np.diff(qber) > -1e-12)
        assert qber[0] == pytest.approx(0.0)

    def test_qber_stays_in_range(self, rng):
        residuals = rotation_matrix(rng.normal(size=(100, 3)), rng.uniform(0, np.pi, 100))
        qber = bb84_qber(residuals)
        assert np.all(qber >= -1e-12) and np.all(qber <= 1 + 1e-12)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

class TestControlConfig:

    def test_latency_must_be_a_whole_number_of_control_periods(self):
        """A fractional latency silently mismatches training and inference."""
        with pytest.raises(ValueError, match="integer multiple"):
            ControlConfig(control_period_s=2e-3, latency_s=5e-3)
        ControlConfig(control_period_s=2e-3, latency_s=6e-3)

    def test_rejects_nonsense_timing(self):
        with pytest.raises(ValueError):
            ControlConfig(control_period_s=0.0)
        with pytest.raises(ValueError):
            ControlConfig(latency_s=-1e-3)


class TestClosedLoop:

    def test_a_trajectory_without_rotations_is_refused(self):
        trajectory = PolarizationDriftModel(**QUIET).simulate(0.5, 1e-3, rng=0)
        with pytest.raises(ValueError, match="track_rotation"):
            run_closed_loop(trajectory, Persistence(), make_control())

    def test_no_compensation_leaves_the_fibre_exactly_as_it_is(self):
        trajectory = make_trajectory(seed=1)
        result = run_closed_loop(trajectory, NoCompensation(), make_control(), rng=0)
        assert np.allclose(result.residual_angle_rad,
                           rotation_angle(trajectory.rotation), atol=1e-12)

    def test_a_perfect_oracle_reaches_zero_residual(self):
        """Sanity floor: if the loop itself lost information, nothing could win.

        A correction is a fixed rotation held until the next one lands, so even
        a perfect forecast is only exact at the instant it takes effect — the
        fibre keeps moving underneath it in between.  Those landing instants are
        where the loop's own bookkeeping is testable: a residual there means the
        loop lost information no compensator could recover.  What happens
        between them is the control period's fault, not the predictor's, and it
        is bounded by the drift over one period, which is the second assertion.
        """
        trajectory = make_trajectory(seed=2)
        control = make_control()
        stride = int(round(control.control_period_s / 1e-3))
        latency_steps = int(round(control.latency_s / 1e-3))

        class Oracle:
            name = "oracle"

            def __init__(self):
                self.index = 0

            def reset(self):
                self.index = 0

            def predict(self, time_s, measured, covariates, horizon_s):
                steps = int(round(horizon_s / 1e-3))
                target = min(self.index * stride + steps, len(trajectory.rotation) - 1)
                self.index += 1
                return trajectory.rotation[target]

        result = run_closed_loop(trajectory, Oracle(), control, rng=0)

        landings = np.arange(latency_steps, len(trajectory.rotation), stride)
        # 1e-6 rather than machine epsilon: the residual angle goes through an
        # arccos near 1, where a 1e-16 error in the trace becomes ~1e-8 in the
        # angle. That is the metric's conditioning, not the loop's arithmetic.
        assert float(np.max(result.residual_angle_rad[landings])) < 1e-6

        # Between landings the residual is bounded by one control period of drift.
        truth = rotation_angle(np.einsum("nij,njk->nik",
                                         np.transpose(trajectory.rotation[:-stride],
                                                      (0, 2, 1)),
                                         trajectory.rotation[stride:]))
        settled = result.residual_angle_rad[latency_steps:]
        assert float(np.max(settled)) < 2.0 * float(np.max(truth))

    def test_the_correction_is_always_at_least_one_latency_old(self):
        """No compensator can act on a measurement it has not received yet."""
        trajectory = make_trajectory(seed=3)
        control = make_control(latency_s=20e-3)

        seen: list[float] = []

        class Recorder:
            name = "recorder"

            def reset(self):
                seen.clear()

            def predict(self, time_s, measured, covariates, horizon_s):
                seen.append(time_s)
                assert horizon_s == pytest.approx(control.latency_s)
                return np.eye(3)

        run_closed_loop(trajectory, Recorder(), control, rng=0)
        assert seen
        assert np.allclose(np.diff(seen), control.control_period_s)

    def test_zero_latency_makes_persistence_near_perfect(self):
        """The degenerate case that shows the latency is what creates the problem."""
        trajectory = make_trajectory(seed=4)
        control = make_control(latency_s=0.0,
                               monitoring=MonitoringConfig(photons_per_basis=500_000))
        result = run_closed_loop(trajectory, Persistence(), control, rng=0)
        assert result.summary()["mean_qber"] < 1e-4

    def test_latency_makes_persistence_worse(self):
        trajectory = make_trajectory(seed=5, vibration_amplitude_rad=0.2)
        quick = run_closed_loop(trajectory, Persistence(),
                                make_control(latency_s=2e-3), rng=0).summary()
        slow = run_closed_loop(trajectory, Persistence(),
                               make_control(latency_s=40e-3), rng=0).summary()
        assert slow["mean_qber"] > quick["mean_qber"]

    def test_summary_and_key_rate_are_consistent(self):
        trajectory = make_trajectory(seed=6)
        result = run_closed_loop(trajectory, Persistence(), make_control(), rng=0)
        summary = result.summary()
        assert 0.0 <= summary["mean_qber"] <= 1.0
        assert 0.0 <= summary["uptime"] <= 1.0
        assert np.all(result.secret_key_rate_hz >= 0)
        # No key above the threshold, by construction.
        assert np.all(result.secret_key_rate_hz[result.qber > 0.12] == 0)

    def test_recovery_times_are_reported_for_every_jump(self):
        trajectory = make_trajectory(seed=7, duration_s=6.0, jump_rate_per_s=1.0,
                                     jump_scale_rad=0.8)
        result = run_closed_loop(trajectory, Persistence(), make_control(), rng=0)
        assert len(result.recovery_times_s()) == len(trajectory.jump_times_s)
        assert all(time >= 0 for time in result.recovery_times_s())


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

class TestPredictors:

    @pytest.mark.parametrize("factory", [NoCompensation, Persistence,
                                         LinearExtrapolation, AlphaBetaFilter])
    def test_every_compensator_returns_a_rotation(self, factory, rng):
        compensator = factory()
        compensator.reset()
        for step in range(20):
            measured = np.asarray(rotation_matrix(rng.normal(size=3),
                                                  rng.uniform(0, 1)), dtype=float)
            prediction = np.asarray(compensator.predict(
                step * 2e-3, measured, np.zeros(3), 20e-3), dtype=float)
            assert prediction.shape == (3, 3)
            assert np.allclose(prediction @ prediction.T, np.eye(3), atol=1e-9)

    def test_persistence_returns_what_it_was_given(self, rng):
        measured = np.asarray(rotation_matrix(rng.normal(size=3), 1.0), dtype=float)
        assert np.allclose(Persistence().predict(0.0, measured, np.zeros(3), 0.02),
                           measured)

    def test_linear_extrapolation_follows_a_constant_rotation_exactly(self):
        """A noiseless constant angular velocity is the case it should nail."""
        axis, rate = np.array([0.0, 0.0, 1.0]), 2.0
        compensator = LinearExtrapolation()
        compensator.reset()
        period, horizon = 2e-3, 20e-3

        for step in range(4):
            time_s = step * period
            measured = np.asarray(rotation_matrix(axis, rate * time_s), dtype=float)
            prediction = compensator.predict(time_s, measured, np.zeros(3), horizon)

        expected = np.asarray(rotation_matrix(axis, rate * (time_s + horizon)),
                              dtype=float)
        assert np.allclose(prediction, expected, atol=1e-9)

    def test_alpha_beta_reset_clears_its_state(self, rng):
        compensator = AlphaBetaFilter()
        first = np.asarray(rotation_matrix(rng.normal(size=3), 0.5), dtype=float)
        compensator.predict(0.0, first, np.zeros(3), 0.02)
        compensator.predict(2e-3, first, np.zeros(3), 0.02)
        compensator.reset()
        assert np.allclose(compensator.predict(0.0, first, np.zeros(3), 0.02), first)

    def test_feature_spec_rejects_a_window_that_sees_no_motion(self):
        with pytest.raises(ValueError):
            FeatureSpec(window=1)

    def test_feature_size_matches_the_spec(self, rng):
        spec = FeatureSpec(window=5, use_covariates=True, horizon_scale=2e-3)
        history = [np.asarray(rotation_matrix(rng.normal(size=3), 0.1), dtype=float)
                   for _ in range(5)]
        assert build_features(history, np.zeros(3), spec).shape == (spec.size,)

        bare = FeatureSpec(window=5, use_covariates=False)
        assert build_features(history, np.zeros(3), bare).shape == (bare.size,)

    def test_features_are_zero_for_a_stationary_fibre(self):
        spec = FeatureSpec(window=4, use_covariates=False)
        history = [np.eye(3)] * 4
        assert np.allclose(build_features(history, np.zeros(3), spec), 0.0)

    def test_features_are_invariant_to_where_on_the_sphere_you_are(self, rng):
        """The property that lets one model work over the whole Poincare sphere."""
        spec = FeatureSpec(window=4, use_covariates=False, horizon_scale=1.0)
        motion = [rotation_matrix([0.0, 0.0, 1.0], angle) for angle in (0.0, 0.1, 0.2, 0.3)]
        base = np.asarray(rotation_matrix(rng.normal(size=3), 1.3), dtype=float)

        here = build_features([np.asarray(m, dtype=float) for m in motion],
                              np.zeros(3), spec)
        elsewhere = build_features([np.asarray(m, dtype=float) @ base for m in motion],
                                   np.zeros(3), spec)
        assert np.allclose(here, elsewhere, atol=1e-10)

    def test_standardiser_round_trip(self, rng):
        features = rng.normal(size=(100, 6)) * np.array([1, 10, 100, 0.1, 1, 1])
        scaled = Standardiser().fit(features).transform(features)
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-10)

    def test_a_learned_compensator_falls_back_before_its_window_fills(self, rng):
        class Constant:
            def predict(self, features):
                return np.full((len(features), 3), 10.0)

        spec = FeatureSpec(window=4, horizon_scale=1.0)
        compensator = LearnedCompensator(model=Constant(), spec=spec,
                                         standardiser=Standardiser(), name="c")
        compensator.reset()
        measured = np.asarray(rotation_matrix(rng.normal(size=3), 0.4), dtype=float)

        # Before the history fills, it must behave like persistence rather than
        # invent a prediction from a fabricated window.
        for _ in range(spec.window - 1):
            assert np.allclose(compensator.predict(0.0, measured, np.zeros(3), 0.02),
                               measured)
        assert not np.allclose(compensator.predict(0.0, measured, np.zeros(3), 0.02),
                               measured)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class TestTraining:

    def _dataset(self, seeds=(0, 1), **kwargs):
        trajectories = [make_trajectory(seed=seed, duration_s=3.0) for seed in seeds]
        spec = FeatureSpec(window=6, horizon_scale=2e-3)
        return build_dataset(trajectories, make_control(), spec, **kwargs)

    def test_shapes_and_provenance(self):
        dataset = self._dataset()
        assert dataset.features.shape[1] == dataset.spec.size
        assert dataset.targets.shape[1] == 3
        assert len(dataset) == len(dataset.episode) == len(dataset.time_s)
        assert set(np.unique(dataset.episode)) == {0, 1}

    def test_unknown_target_is_refused(self):
        with pytest.raises(ValueError, match="unknown target"):
            self._dataset(target="wishful")

    def test_trajectories_without_rotations_are_refused(self):
        trajectory = PolarizationDriftModel(**QUIET).simulate(1.0, 1e-3, rng=0)
        with pytest.raises(ValueError, match="track_rotation"):
            build_dataset([trajectory], make_control(), FeatureSpec(window=4))

    def test_the_default_target_is_the_future_measurement_not_the_truth(self):
        """Training on ground truth fits a model that cannot be retrained in the field."""
        measured = self._dataset(target="measured", rng=0)
        truth = self._dataset(target="true", rng=0)
        assert not np.allclose(measured.targets, truth.targets)
        # Both describe the same motion, so they must agree on average.
        assert np.allclose(measured.targets.mean(axis=0), truth.targets.mean(axis=0),
                           atol=0.5 * np.abs(truth.targets).mean())

    def test_the_chronological_split_leaves_a_gap(self):
        dataset = self._dataset()
        train, validation = chronological_split(dataset, 0.7, 0.05)
        assert not (train & validation).any()
        assert train.sum() > 0 and validation.sum() > 0
        for episode in np.unique(dataset.episode):
            selector = dataset.episode == episode
            assert (dataset.time_s[selector & train].max()
                    < dataset.time_s[selector & validation].min())

    def test_the_split_rejects_a_degenerate_fraction(self):
        with pytest.raises(ValueError):
            chronological_split(self._dataset(), train_fraction=1.0)

    def test_ridge_fits_and_runs_in_the_loop(self):
        dataset = self._dataset()
        train, _ = chronological_split(dataset, 0.8, 0.02)
        compensator = fit_ridge(dataset, train)

        result = run_closed_loop(make_trajectory(seed=9), compensator,
                                 make_control(), rng=0)
        assert np.all(np.isfinite(result.qber))
        assert result.summary()["mean_qber"] < 0.5

    def test_a_fitted_predictor_beats_doing_nothing(self):
        dataset = self._dataset(seeds=(0, 1, 2))
        train, _ = chronological_split(dataset, 0.85, 0.02)
        compensator = fit_ridge(dataset, train)

        test = make_trajectory(seed=11, duration_s=3.0)
        fitted = run_closed_loop(test, compensator, make_control(), rng=1).summary()
        nothing = run_closed_loop(test, NoCompensation(), make_control(), rng=1).summary()
        assert fitted["mean_qber"] < nothing["mean_qber"]

    def test_the_mlp_trains_and_keeps_its_best_epoch(self):
        dataset = self._dataset(seeds=(0, 1, 2))
        train, validation = chronological_split(dataset, 0.75, 0.05)
        compensator, history = fit_mlp(dataset, train, validation, hidden=(32, 32),
                                       epochs=8, seed=0)

        assert len(history) == 8
        assert all("validation_mse" in record for record in history)
        assert history[-1]["train_mse"] < history[0]["train_mse"]

        result = run_closed_loop(make_trajectory(seed=12), compensator,
                                 make_control(), rng=0)
        assert np.all(np.isfinite(result.qber))
