"""The Raman coexistence model, checked against its own limits.

There is no reference implementation to compare against here the way phase 1
compared Bell-diagonal operations against SeQUeNCe, so the tests are written
against the properties the physics must have: linearity in launched power, the
right growth with length in each direction, the ceiling at a coin flip, and an
inversion that actually inverts. Where a number can be pinned to the published
coexistence experiments, it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.physics.coexistence import (CoexistenceParameters,
                                                coexistence_qber,
                                                max_classical_power,
                                                raman_noise_counts,
                                                raman_noise_power)
from qnet_autopilot.physics.fiber import transmittance
from qnet_autopilot.physics.link import bb84_secret_key_rate


def signal_counts(length_km: float, mean_photon_number: float = 0.5,
                  parameters: CoexistenceParameters | None = None) -> float:
    """Detected signal counts per gate for a weak-coherent-pulse link."""
    parameters = parameters or CoexistenceParameters()
    return (mean_photon_number * float(transmittance(length_km))
            * parameters.detector_efficiency)


class TestParameters:

    @pytest.mark.parametrize("bad", [
        dict(raman_coefficient_per_km_nm=-1.0),
        dict(filter_bandwidth_nm=0.0),
        dict(detection_window_s=-1e-9),
        dict(detector_efficiency=0.0),
        dict(detector_efficiency=1.5),
        dict(optical_error_rate=0.5),
        dict(optical_error_rate=-0.01),
    ])
    def test_rejects_impossible_hardware(self, bad):
        with pytest.raises(ValueError):
            CoexistenceParameters(**bad)

    def test_photon_energy_is_right_for_the_c_band(self):
        """1550 nm photons carry about 0.8 eV."""
        energy_ev = CoexistenceParameters().photon_energy_j / 1.602176634e-19
        assert energy_ev == pytest.approx(0.80, abs=0.01)

    def test_attenuation_converts_db_to_napierian(self):
        """0.2 dB/km is 0.04605 per km; a 1/e length of about 21.7 km."""
        parameters = CoexistenceParameters(attenuation_db_per_km=0.2)
        assert parameters.attenuation_per_km == pytest.approx(0.046052, rel=1e-4)
        assert 1.0 / parameters.attenuation_per_km == pytest.approx(21.71, rel=1e-3)


class TestRamanPower:

    def test_scattering_is_linear_in_launched_power(self):
        """Spontaneous scattering has no threshold and no saturation.

        This is the property the allocator relies on: classical admission is a
        continuous, proportional dial on the noise it causes.
        """
        powers = np.array([0.0, 1e-6, 1e-4, 1e-3, 1e-2])
        for direction in ("forward", "backward"):
            noise = raman_noise_power(powers, 40.0, direction=direction)
            assert noise[0] == 0.0
            ratios = noise[1:] / powers[1:]
            assert np.allclose(ratios, ratios[0], rtol=1e-12)

    def test_idle_fibre_scatters_nothing(self):
        assert float(raman_noise_power(0.0, 50.0)) == 0.0

    def test_forward_scattering_peaks_and_then_falls(self):
        """Generated along the span, then attenuated with what is left of it.

        ``L exp(-alpha L)`` peaks at ``L = 1/alpha``, about 21.7 km at 0.2
        dB/km. Past that, extra fibre attenuates the accumulated noise faster
        than it generates more.
        """
        parameters = CoexistenceParameters()
        lengths = np.linspace(1.0, 200.0, 400)
        noise = raman_noise_power(1e-3, lengths, parameters, "forward")
        peak = lengths[int(np.argmax(noise))]
        assert peak == pytest.approx(1.0 / parameters.attenuation_per_km, rel=0.02)
        assert noise[-1] < noise[int(np.argmax(noise))]

    def test_backward_scattering_saturates(self):
        """Light scattered far down the span never makes it back.

        The integral converges to ``P rho dlambda / (2 alpha)``, so beyond a
        few attenuation lengths more fibre adds no more noise.
        """
        parameters = CoexistenceParameters()
        limit = (1e-3 * parameters.raman_coefficient_per_km_nm
                 * parameters.filter_bandwidth_nm / (2.0 * parameters.attenuation_per_km))
        assert float(raman_noise_power(1e-3, 1000.0, parameters, "backward")) == \
            pytest.approx(limit, rel=1e-6)

        lengths = np.array([50.0, 100.0, 200.0, 400.0])
        noise = raman_noise_power(1e-3, lengths, parameters, "backward")
        assert np.all(np.diff(noise) > 0), "monotone in length"
        assert noise[-1] / noise[-2] < 1.01, "and saturating"

    def test_short_spans_barely_distinguish_the_two_directions(self):
        """Both integrals reduce to ``P rho dlambda L`` when ``alpha L << 1``."""
        forward = float(raman_noise_power(1e-3, 0.5, direction="forward"))
        backward = float(raman_noise_power(1e-3, 0.5, direction="backward"))
        assert forward == pytest.approx(backward, rel=0.03)

    def test_long_spans_make_counter_propagation_much_worse(self):
        forward = float(raman_noise_power(1e-3, 150.0, direction="forward"))
        backward = float(raman_noise_power(1e-3, 150.0, direction="backward"))
        assert backward > 50 * forward

    def test_rejects_nonsense_inputs(self):
        with pytest.raises(ValueError):
            raman_noise_power(-1e-3, 10.0)
        with pytest.raises(ValueError):
            raman_noise_power(1e-3, -10.0)
        with pytest.raises(ValueError, match="unknown direction"):
            raman_noise_power(1e-3, 10.0, direction="sideways")

    def test_counts_scale_with_the_gate_and_the_detector(self):
        base = CoexistenceParameters()
        wider = CoexistenceParameters(detection_window_s=base.detection_window_s * 3)
        blinder = CoexistenceParameters(detector_efficiency=base.detector_efficiency / 2)

        reference = float(raman_noise_counts(1e-3, 40.0, base))
        assert float(raman_noise_counts(1e-3, 40.0, wider)) == \
            pytest.approx(3 * reference, rel=1e-12)
        assert float(raman_noise_counts(1e-3, 40.0, blinder)) == \
            pytest.approx(reference / 2, rel=1e-12)

    def test_a_narrower_filter_buys_proportionally_less_noise(self):
        wide = CoexistenceParameters(filter_bandwidth_nm=1.0)
        narrow = CoexistenceParameters(filter_bandwidth_nm=0.1)
        assert float(raman_noise_counts(1e-3, 40.0, wide)) == \
            pytest.approx(10 * float(raman_noise_counts(1e-3, 40.0, narrow)), rel=1e-12)


class TestCoexistenceQber:

    def test_an_idle_fibre_shows_only_alignment_error(self):
        """With no classical traffic and negligible dark counts, e = e_optics."""
        parameters = CoexistenceParameters(dark_count_rate_hz=0.0)
        qber = float(coexistence_qber(signal_counts(25.0), 0.0, 25.0, 0.0, parameters))
        assert qber == pytest.approx(parameters.optical_error_rate, rel=1e-9)

    def test_drift_adds_to_the_optics_error(self):
        parameters = CoexistenceParameters(dark_count_rate_hz=0.0)
        qber = float(coexistence_qber(signal_counts(25.0), 0.0, 25.0, 0.03, parameters))
        assert qber == pytest.approx(0.03 + parameters.optical_error_rate, rel=1e-9)

    def test_qber_rises_with_classical_power(self):
        powers = np.array([0.0, 1e-5, 1e-4, 1e-3, 1e-2])
        qber = coexistence_qber(signal_counts(50.0), powers, 50.0)
        assert np.all(np.diff(qber) > 0)

    def test_background_saturates_at_a_coin_flip(self):
        """Unpolarised noise is wrong half the time, and never more than that.

        The ceiling is what keeps an allocator sane: QBER does not grow without
        bound as classical power rises, so a link that is already dead cannot be
        made *more* dead, and there is nothing to buy by throttling further.
        """
        signal = signal_counts(50.0)
        powers = np.array([1e0, 1e2, 1e4, 1e6])
        qber = coexistence_qber(signal, powers, 50.0)

        assert np.all(qber <= 0.5)
        assert np.all(np.diff(qber) > 0), "approaches the ceiling from below"
        assert float(qber[-1]) == pytest.approx(0.5, abs=1e-8)

    def test_a_stronger_signal_dilutes_the_same_background(self):
        """Error sources share a denominator; they do not simply add."""
        weak = float(coexistence_qber(1e-3, 1e-4, 50.0))
        strong = float(coexistence_qber(1e-1, 1e-4, 50.0))
        assert strong < weak

    def test_naive_addition_would_overstate_the_total(self):
        """The combination is sub-additive, which is the reason it is not a sum."""
        parameters = CoexistenceParameters()
        signal = signal_counts(50.0, parameters=parameters)
        drift = 0.04
        power = 5e-4

        total = float(coexistence_qber(signal, power, 50.0, drift, parameters))
        raman_alone = float(coexistence_qber(signal, power, 50.0, 0.0, parameters))
        drift_alone = float(coexistence_qber(signal, 0.0, 50.0, drift, parameters))
        assert total < raman_alone + drift_alone

    def test_dark_counts_alone_hurt_a_faint_signal(self):
        loud = CoexistenceParameters(dark_count_rate_hz=1.0e5)
        quiet = CoexistenceParameters(dark_count_rate_hz=0.0)
        signal = signal_counts(120.0)
        assert float(coexistence_qber(signal, 0.0, 120.0, 0.0, loud)) > \
            float(coexistence_qber(signal, 0.0, 120.0, 0.0, quiet))

    def test_a_dead_signal_is_pure_noise(self):
        assert float(coexistence_qber(0.0, 0.0, 50.0,
                                      parameters=CoexistenceParameters(
                                          dark_count_rate_hz=0.0))) == 0.5

    def test_rejects_a_negative_signal(self):
        with pytest.raises(ValueError):
            coexistence_qber(-1.0, 0.0, 50.0)

    def test_the_operating_point_matches_the_published_experiments(self):
        """Coexistence works near -10 dBm and fails near 0 dBm over metro spans.

        Chapuran et al. (2009), Peters et al. (2009) and Eraerds et al. (2010)
        all report QKD surviving alongside classical channels at launch powers
        of roughly -10 to -20 dBm over tens of kilometres, and failing as the
        classical channel approaches the 0 dBm a normal telecom link runs at.
        This pins the model to that regime — if a parameter change moved it by
        an order of magnitude, every conclusion in phase 4 would move with it.
        """
        length_km = 50.0
        signal = signal_counts(length_km)

        surviving = float(bb84_secret_key_rate(
            1e5, coexistence_qber(signal, 1e-4, length_km)))       # -10 dBm
        dead = float(bb84_secret_key_rate(
            1e5, coexistence_qber(signal, 1e-3, length_km)))       # 0 dBm

        assert surviving > 0.5e5, "at -10 dBm the link should still make good key"
        assert dead == 0.0, "at 0 dBm a 50 km link should produce none"


class TestInversion:

    @pytest.mark.parametrize("length_km", [10.0, 50.0, 90.0])
    @pytest.mark.parametrize("drift", [0.0, 0.02, 0.05])
    def test_the_returned_power_exactly_exhausts_the_budget(self, length_km, drift):
        budget = 0.08
        signal = signal_counts(length_km)
        power = float(max_classical_power(signal, length_km, budget, drift))
        assert power > 0
        assert float(coexistence_qber(signal, power, length_km, drift)) == \
            pytest.approx(budget, rel=1e-9)

    def test_headroom_shrinks_as_the_fibre_drifts(self):
        """The allocator's whole lever: drift and classical traffic share a budget."""
        signal = signal_counts(50.0)
        powers = [float(max_classical_power(signal, 50.0, 0.08, drift))
                  for drift in (0.0, 0.02, 0.04, 0.06)]
        assert np.all(np.diff(powers) < 0)

    def test_a_spent_budget_returns_zero_rather_than_a_negative_power(self):
        """Drift past the budget cannot be bought back by throttling classical."""
        signal = signal_counts(50.0)
        assert float(max_classical_power(signal, 50.0, 0.05, 0.20)) == 0.0

    def test_a_budget_at_the_coin_flip_admits_anything(self):
        assert np.isinf(float(max_classical_power(signal_counts(50.0), 50.0, 0.5, 0.0)))

    def test_longer_spans_admit_less(self):
        lengths = np.array([10.0, 30.0, 60.0, 100.0])
        powers = np.array([float(max_classical_power(signal_counts(length),
                                                     length, 0.08, 0.01))
                           for length in lengths])
        assert np.all(np.diff(powers) < 0)

    def test_it_inverts_the_backward_direction_too(self):
        signal = signal_counts(70.0)
        power = float(max_classical_power(signal, 70.0, 0.07, 0.01,
                                          direction="backward"))
        assert float(coexistence_qber(signal, power, 70.0, 0.01,
                                      direction="backward")) == \
            pytest.approx(0.07, rel=1e-9)
        assert power < float(max_classical_power(signal, 70.0, 0.07, 0.01,
                                                 direction="forward"))
