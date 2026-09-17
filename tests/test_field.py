"""Field calibration: the measured constants, and the fit that consumes real data.

Two things are under test here and they are different in kind.

The first is arithmetic: converting a published polarisation linewidth into this
model's drift rate, superposing relaxation times, fitting a spectral exponent.
Those have right answers and the tests check them against closed forms.

The second is an estimator, and estimators are tested by round trip: simulate a
trajectory from known parameters, hand the estimator nothing but the state of
polarisation, and see whether it comes back with the parameters. That is the
only test that means anything, because the estimator will be run on data whose
true parameters nobody knows.

The round-trip tests also pin the estimator's **limits**, which matter as much
as its accuracy. A recording sampled below a mechanical band cannot report that
band, and one dominated by it cannot report the drift underneath — the fit is
required to say so rather than to return a number anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.datagen.measured import (StokesRecording, fit_drift_model,
                                             load_stokes)
from qnet_autopilot.physics.field_calibration import (
    AERIAL_OPGW, BURIED_CZEGLEDI, CALIBRATED_DRIFT_RATE_RAD_PER_S,
    FLICKER_EXPONENT, MEASUREMENTS, POLARISATION_LINEWIDTH_HZ,
    RAMAN_COEFFICIENT_RANGE_PER_KM_NM, SPECTRAL_FIT_BAND_HZ,
    flicker_components, linewidth_to_drift_rate, spectral_exponent)
from qnet_autopilot.physics.fiber import PolarizationDriftModel


def _drift(**overrides) -> PolarizationDriftModel:
    parameters = dict(thermal_correlation_time_s=60.0,
                      thermal_drift_rate_rad_per_s=0.02,
                      thermal_per_kelvin_rad_per_s=0.0,
                      vibration_amplitude_rad=0.0,
                      jump_rate_per_s=0.0)
    parameters.update(overrides)
    return PolarizationDriftModel(**parameters)


def _record(model, duration_s, dt_s, seed):
    trajectory = model.simulate(duration_s, dt_s, rng=seed)
    return StokesRecording(time_s=trajectory.time_s, stokes=trajectory.stokes,
                           source=f"synthetic seed {seed}")


class TestProvenance:
    """The constants have to be traceable, or they are just numbers."""

    def test_every_measurement_names_a_source_and_a_url(self):
        for measurement in MEASUREMENTS:
            assert measurement.source.strip(), measurement.quantity
            assert measurement.url.startswith("https://"), measurement.quantity
            assert measurement.calibrates.strip(), measurement.quantity

    def test_the_measured_raman_coefficient_is_below_the_package_default(self):
        """Which direction the package errs in is worth knowing, and testing.

        The default is 4.0e-9 /km/nm and the measured range tops out at 3.1e-9,
        so every coexistence figure in phase 4 is pessimistic rather than
        optimistic. If someone lowers the default below the measured range this
        test should stop them and make them say why.
        """
        from qnet_autopilot.physics.coexistence import CoexistenceParameters

        default = CoexistenceParameters().raman_coefficient_per_km_nm
        low, high = RAMAN_COEFFICIENT_RANGE_PER_KM_NM
        assert low < high
        assert default >= high, (
            "the default Raman coefficient has dropped into or below the "
            "measured range; phase 4 is no longer conservative")


class TestLinewidthConversion:
    """Czegledi's random walk and this model's angular velocity, reconciled."""

    @pytest.mark.parametrize("linewidth_hz", [60e-9, 80e-9, 1e-6])
    @pytest.mark.parametrize("dt_s", [1e-3, 1.0, 60.0])
    def test_it_matches_the_closed_form(self, linewidth_hz, dt_s):
        expected = np.sqrt(2.0 * np.pi * linewidth_hz / dt_s)
        assert linewidth_to_drift_rate(linewidth_hz, dt_s) == pytest.approx(expected)

    def test_a_shorter_interval_reports_a_faster_rate(self):
        """Not a defect: a random walk has no characteristic velocity."""
        fast = linewidth_to_drift_rate(60e-9, 1e-3)
        slow = linewidth_to_drift_rate(60e-9, 1.0)
        assert fast > slow
        assert fast / slow == pytest.approx(np.sqrt(1000.0), rel=1e-9)

    def test_it_reproduces_the_measured_step_variance(self):
        """The rate held for T must give a step of the published variance."""
        linewidth, dt_s = 60e-9, 1.0
        rate = linewidth_to_drift_rate(linewidth, dt_s)
        assert (rate * dt_s) ** 2 == pytest.approx(2 * np.pi * linewidth * dt_s)

    @pytest.mark.parametrize("linewidth_hz,dt_s", [(0.0, 1.0), (-1.0, 1.0),
                                                   (1e-9, 0.0), (1e-9, -1.0)])
    def test_it_rejects_impossible_arguments(self, linewidth_hz, dt_s):
        with pytest.raises(ValueError):
            linewidth_to_drift_rate(linewidth_hz, dt_s)


class TestFlickerComponents:
    """The superposition that was tried, kept, and not needed."""

    def test_it_preserves_the_total_variance(self):
        components = flicker_components(0.05, 1e-4, 1.0, 5)
        total = np.sqrt(sum(rate ** 2 for _, rate in components))
        assert total == pytest.approx(0.05)

    def test_the_corners_span_the_requested_band(self):
        components = flicker_components(0.05, 1e-4, 1.0, 5)
        corners = sorted(1.0 / (2 * np.pi * tau) for tau, _ in components)
        assert corners[0] == pytest.approx(1e-4, rel=1e-9)
        assert corners[-1] == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize("kwargs", [
        dict(low_frequency_hz=1.0, high_frequency_hz=1e-4),
        dict(low_frequency_hz=0.0),
        dict(num_components=1),
    ])
    def test_it_rejects_a_degenerate_request(self, kwargs):
        with pytest.raises(ValueError):
            flicker_components(0.05, **kwargs)

    def test_the_components_reach_the_model(self):
        """Two identical components must add in variance, not in amplitude.

        Measured on the angular *increment* rather than on the accumulated
        angle: the angle between the launched and received states is bounded by
        pi, so at any interesting drift rate both models saturate and the
        comparison would be between two ceilings.
        """
        one = _drift(thermal_components=((60.0, 0.02),))
        two = _drift(thermal_components=((60.0, 0.02), (60.0, 0.02)))
        speeds = []
        for model in (one, two):
            stokes = model.simulate(400.0, 0.5, rng=5).stokes
            step = np.linalg.norm(np.diff(stokes, axis=0), axis=1)
            speeds.append(float(np.sqrt(np.mean(step ** 2))))
        assert speeds[1] == pytest.approx(speeds[0] * np.sqrt(2.0), rel=0.15)


class TestSpectralExponent:
    """The one number that says whether the drift has the right shape."""

    def test_white_noise_is_flat(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(size=200_000)
        assert spectral_exponent(signal, 1.0, 1e-3, 0.4) == pytest.approx(0.0, abs=0.15)

    def test_a_random_walk_falls_as_f_squared(self):
        rng = np.random.default_rng(1)
        signal = np.cumsum(rng.normal(size=200_000))
        assert spectral_exponent(signal, 1.0, 1e-3, 0.4) == pytest.approx(2.0, abs=0.25)

    def test_it_refuses_a_band_with_too_few_points(self):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="spectral points"):
            spectral_exponent(rng.normal(size=64), 1.0, 0.4, 0.41)


class TestTheCalibratedOperatingPoint:
    """What the two field measurements actually pin down."""

    def test_the_aerial_preset_carries_the_power_line_harmonics(self):
        """Phase 3 used 40 Hz; the only aerial measurement says 60/120/180."""
        assert AERIAL_OPGW.vibration_frequency_hz == 60.0
        assert len(AERIAL_OPGW.vibration_harmonics) == 3
        assert AERIAL_OPGW.vibration_harmonics[0] > AERIAL_OPGW.vibration_harmonics[-1]

    def test_the_harmonics_put_power_where_they_claim_to(self):
        model = _drift(vibration_frequency_hz=60.0, vibration_bandwidth_hz=2.0,
                       vibration_amplitude_rad=0.1,
                       vibration_harmonics=(1.0, 0.5, 0.25),
                       thermal_drift_rate_rad_per_s=0.0)
        trajectory = model.simulate(20.0, 1e-3, rng=4)
        signal = trajectory.stokes[:, 1] - trajectory.stokes[:, 1].mean()
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        frequency = np.fft.rfftfreq(signal.size, 1e-3)

        def power_near(centre, width=4.0):
            inside = np.abs(frequency - centre) < width
            return float(spectrum[inside].sum())

        for harmonic in (60.0, 120.0, 180.0):
            assert power_near(harmonic) > 20.0 * power_near(300.0), (
                f"no power at {harmonic} Hz")
        assert power_near(60.0) > power_near(180.0)

    def test_harmonics_redistribute_the_jitter_rather_than_adding_any(self):
        """The amplitude parameter has to keep meaning what it says."""
        plain = _drift(vibration_frequency_hz=60.0, vibration_amplitude_rad=0.1,
                       thermal_drift_rate_rad_per_s=0.0)
        rich = _drift(vibration_frequency_hz=60.0, vibration_amplitude_rad=0.1,
                      vibration_harmonics=(1.0, 0.5, 0.25),
                      thermal_drift_rate_rad_per_s=0.0)
        spreads = [float(np.std(model.simulate(20.0, 1e-3, rng=6).angle_rad))
                   for model in (plain, rich)]
        assert spreads[1] == pytest.approx(spreads[0], rel=0.20)

    def test_the_default_harmonics_leave_the_original_model_untouched(self):
        """A new parameter that changes an old result is a broken parameter."""
        model = _drift(vibration_frequency_hz=40.0, vibration_amplitude_rad=0.3)
        first = model.simulate(2.0, 1e-3, rng=9).stokes
        second = PolarizationDriftModel(
            thermal_correlation_time_s=60.0, thermal_drift_rate_rad_per_s=0.02,
            thermal_per_kelvin_rad_per_s=0.0, vibration_frequency_hz=40.0,
            vibration_amplitude_rad=0.3, jump_rate_per_s=0.0,
        ).simulate(2.0, 1e-3, rng=9).stokes
        assert np.array_equal(first, second)

    def test_the_calibrated_rate_reproduces_the_measured_exponent(self):
        """The rate is not chosen for taste: the spectrum pins it.

        ORNL measured beta = 1.0 +- 0.2 over 0.3-10 mHz. In this model beta is a
        sharp, monotone function of the drift rate, because the state of
        polarisation wraps around the sphere and drifting faster whitens the
        spectrum. This checks that the constant recorded as calibrated actually
        lands inside the measured uncertainty, and — the part that makes it a
        calibration rather than a coincidence — that the rate phase 3 used does
        not.
        """
        centre, tolerance = FLICKER_EXPONENT
        duration, dt_s = 40_000.0, 1.0

        calibrated = _drift(thermal_drift_rate_rad_per_s=CALIBRATED_DRIFT_RATE_RAD_PER_S)
        exponents = [spectral_exponent(
            calibrated.simulate(duration, dt_s, rng=seed).stokes[:, 0],
            dt_s, *SPECTRAL_FIT_BAND_HZ) for seed in range(2)]
        assert float(np.mean(exponents)) == pytest.approx(centre, abs=tolerance)

        phase_three = _drift(thermal_drift_rate_rad_per_s=0.05)
        too_fast = spectral_exponent(
            phase_three.simulate(duration, dt_s, rng=0).stokes[:, 0],
            dt_s, *SPECTRAL_FIT_BAND_HZ)
        assert too_fast < centre - tolerance, (
            "the phase-3 rate reproduces the measured spectrum after all, "
            "which would make the recalibration pointless")

    def test_the_buried_preset_is_quiet(self):
        """Czegledi's fibre barely moves; that is the point of quoting it."""
        assert BURIED_CZEGLEDI.vibration_amplitude_rad == 0.0
        assert BURIED_CZEGLEDI.jump_rate_per_s == 0.0
        expected = linewidth_to_drift_rate(float(np.mean(POLARISATION_LINEWIDTH_HZ)), 1e-3)
        assert BURIED_CZEGLEDI.thermal_drift_rate_rad_per_s == pytest.approx(expected)


class TestLoadingARecording:
    """Reading somebody else's file is where the failures actually happen."""

    def _write(self, tmp_path, rows, header="time_s,s1,s2,s3"):
        path = tmp_path / "sop.csv"
        path.write_text(header + "\n" + "\n".join(rows) + "\n")
        return path

    def test_it_normalises_to_the_poincare_sphere(self, tmp_path):
        rows = [f"{index * 0.1},{2.0},{0.0},{0.0}" for index in range(20)]
        recording = load_stokes(self._write(tmp_path, rows))
        assert np.allclose(np.linalg.norm(recording.stokes, axis=1), 1.0)
        assert recording.dt_s == pytest.approx(0.1)
        assert recording.source == "sop.csv"

    def test_it_sorts_by_time_and_rebases_to_zero(self, tmp_path):
        rows = [f"{100.0 - index * 0.5},1,0,0" for index in range(20)]
        recording = load_stokes(self._write(tmp_path, rows))
        assert recording.time_s[0] == 0.0
        assert np.all(np.diff(recording.time_s) > 0)

    def test_it_names_the_column_it_could_not_find(self, tmp_path):
        rows = [f"{index},1,0,0" for index in range(20)]
        path = self._write(tmp_path, rows, header="t,s1,s2,s3")
        with pytest.raises(ValueError, match="time_s"):
            load_stokes(path)

    def test_it_refuses_a_lost_signal_rather_than_inventing_a_direction(self, tmp_path):
        rows = [f"{index * 0.1},1,0,0" for index in range(19)] + ["1.9,0,0,0"]
        with pytest.raises(ValueError, match="zero-length"):
            load_stokes(self._write(tmp_path, rows))

    def test_it_refuses_a_recording_too_short_to_say_anything(self, tmp_path):
        rows = [f"{index * 0.1},1,0,0" for index in range(4)]
        with pytest.raises(ValueError, match="usable rows"):
            load_stokes(self._write(tmp_path, rows))


class TestFittingARecording:
    """Round trip: known parameters in, state of polarisation out, fit back."""

    def test_it_recovers_a_slow_drift_rate(self):
        """The regime every downloadable recording is in: slow, no band."""
        truth = 0.02
        model = _drift(thermal_drift_rate_rad_per_s=truth)
        rates = [fit_drift_model(_record(model, 4000.0, 0.2, seed)).drift_rate_rad_per_s
                 for seed in (11, 12, 13)]
        assert float(np.mean(rates)) == pytest.approx(truth, rel=0.20)

    def test_it_recovers_a_much_slower_drift_rate(self):
        """A subsea link at 5 Hz is three orders of magnitude quieter."""
        truth = 6e-4
        model = _drift(thermal_correlation_time_s=600.0,
                       thermal_drift_rate_rad_per_s=truth)
        fit = fit_drift_model(_record(model, 40_000.0, 0.2, 11))
        assert fit.drift_rate_rad_per_s == pytest.approx(truth, rel=0.30)
        assert "drift rate" in fit.resolved

    def test_it_recovers_the_jump_statistics(self):
        model = _drift(jump_rate_per_s=0.02, jump_scale_rad=0.35)
        fit = fit_drift_model(_record(model, 4000.0, 0.2, 11))
        assert fit.jump_rate_per_s == pytest.approx(0.02, rel=0.5)
        assert fit.jump_scale_rad == pytest.approx(0.35, rel=0.5)

    def test_an_undersampled_band_is_never_reported_as_measured(self):
        """A 5 Hz recording of a link with a 60 Hz tone must not claim one.

        This is the case every public dataset is in. The power is really there —
        folded down by aliasing — so the detector may well find a peak, and no
        analysis of this recording can undo the fold. What the fit must not do
        is present it as a measurement: either it finds nothing, or it says the
        peak may be an alias, and either way ``to_model`` leaves it out.
        """
        model = _drift(vibration_frequency_hz=60.0, vibration_amplitude_rad=0.1)
        for seed in (11, 12):
            fit = fit_drift_model(_record(model, 2000.0, 0.2, seed))
            if fit.vibration_frequency_hz is None:
                assert any("vibration band" in note for note in fit.unresolved)
            else:
                assert any("aliasing" in note for note in fit.unresolved)
                assert fit.vibration_frequency_hz < 2.5, (
                    "a peak above Nyquist would mean the FFT is wrong")
            assert fit.to_model().vibration_amplitude_rad == 0.0

    def test_it_finds_a_band_the_sampling_does_resolve(self):
        model = _drift(vibration_frequency_hz=60.0, vibration_bandwidth_hz=2.0,
                       vibration_amplitude_rad=0.05)
        fit = fit_drift_model(_record(model, 300.0, 1e-3, 13))
        assert fit.vibration_frequency_hz == pytest.approx(60.0, abs=3.0)
        assert any("amplitude" in note for note in fit.unresolved), (
            "the amplitude is a lower bound and the fit has to say so")

    def test_it_calls_the_drift_rate_contaminated_when_a_band_dominates(self):
        """An upper bound reported as an estimate is worse than no number."""
        model = _drift(vibration_frequency_hz=60.0, vibration_bandwidth_hz=2.0,
                       vibration_amplitude_rad=0.05)
        fit = fit_drift_model(_record(model, 300.0, 1e-3, 13))
        assert any("drift rate" in note for note in fit.unresolved)
        assert "drift rate" not in fit.resolved

    def test_a_clean_recording_is_not_called_contaminated(self):
        model = _drift(thermal_drift_rate_rad_per_s=0.02)
        fit = fit_drift_model(_record(model, 4000.0, 0.2, 11))
        assert "drift rate" in fit.resolved
        assert not any("drift rate" in note for note in fit.unresolved)

    def test_the_fitted_model_leaves_out_what_was_never_measured(self):
        undersampled = _drift(vibration_frequency_hz=60.0,
                              vibration_amplitude_rad=0.1)
        fitted = fit_drift_model(_record(undersampled, 2000.0, 0.2, 11)).to_model()
        assert fitted.vibration_amplitude_rad == 0.0, (
            "a band the recording could not resolve must not appear in the model")

        with_jumps = _drift(jump_rate_per_s=0.02, jump_scale_rad=0.35)
        fitted = fit_drift_model(_record(with_jumps, 2000.0, 0.2, 11)).to_model()
        assert fitted.jump_rate_per_s > 0.0

    def test_a_loud_band_hides_the_jumps_underneath_it(self):
        """And the fit reports nothing rather than a rate of zero it believes.

        A jump is defined against the typical increment. Put a 0.1 rad band on
        the same fibre and a 0.35 rad jump is no longer in anybody's tail — so
        the jump rate collapses to zero. That is a true statement about what the
        recording can show, and it is only safe because the drift rate is
        flagged contaminated in the same breath.
        """
        loud = _drift(vibration_frequency_hz=60.0, vibration_bandwidth_hz=2.0,
                      vibration_amplitude_rad=0.1, jump_rate_per_s=0.05,
                      jump_scale_rad=0.35)
        fit = fit_drift_model(_record(loud, 300.0, 1e-3, 13))
        assert fit.jump_rate_per_s < 0.05
        assert any("drift rate" in note for note in fit.unresolved)

    def test_the_diagnostics_carry_the_interval_the_rate_belongs_to(self):
        """A drift rate quoted without its interval is not a number."""
        fit = fit_drift_model(_record(_drift(), 2000.0, 0.2, 11))
        assert fit.diagnostics["dt_s"] == pytest.approx(0.2)
        assert fit.diagnostics["drift_lag_s"] >= fit.diagnostics["dt_s"]
        assert fit.diagnostics["nyquist_hz"] == pytest.approx(2.5)

    def test_it_refuses_a_recording_too_short_to_fit(self):
        recording = StokesRecording(time_s=np.arange(8) * 0.1,
                                    stokes=np.tile([1.0, 0.0, 0.0], (8, 1)))
        with pytest.raises(ValueError, match="at least 32"):
            fit_drift_model(recording)
