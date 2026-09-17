"""Property tests for the physics layer.

:mod:`test_bds_matches_sequence` proves the Bell-diagonal functions agree with
SeQUeNCe.  That is necessary but not sufficient: agreement with a reference
does not catch a shared misunderstanding, and the fibre, memory and link models
have no SeQUeNCe counterpart to compare against at all.

So these tests check the properties the physics must have regardless of
implementation — closed forms at the limits, monotonicity, conservation,
statistics of the stochastic parts.  They are the tests that would catch a sign
error that happened to be copied faithfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.physics import bell_diagonal as bds
from qnet_autopilot.physics import constants as C
from qnet_autopilot.physics import fiber, link, memory


# ---------------------------------------------------------------------------
# Bell-diagonal algebra
# ---------------------------------------------------------------------------

class TestBellDiagonal:

    @pytest.mark.parametrize("F", [0.25, 0.5, 0.73, 1.0])
    def test_werner_is_normalised_and_has_the_stated_fidelity(self, F):
        state = bds.werner(F)
        assert bds.fidelity(state) == pytest.approx(F)
        assert state.sum() == pytest.approx(1.0)
        assert bool(bds.is_valid(state))

    def test_werner_broadcasts(self):
        state = bds.werner([0.5, 0.9])
        assert state.shape == (2, 4)
        assert np.allclose(bds.fidelity(state), [0.5, 0.9])

    def test_twirl_preserves_fidelity_and_flattens_the_rest(self):
        state = np.array([0.7, 0.2, 0.05, 0.05])
        twirled = bds.twirl(state)
        assert bds.fidelity(twirled) == pytest.approx(0.7)
        assert np.allclose(twirled[1:], 0.1)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            bds.fidelity([0.5, 0.5, 0.0])

    @pytest.mark.parametrize("F1", [0.6, 0.8, 0.95])
    @pytest.mark.parametrize("F2", [0.6, 0.8, 0.95])
    def test_perfect_swap_matches_the_textbook_werner_formula(self, F1, F2):
        """With ideal hardware, swapping two Werner states gives
        ``F = F1 F2 + (1-F1)(1-F2)/3`` — the standard result."""
        result = bds.swap(bds.werner(F1), bds.werner(F2), gate_fid=1.0, meas_fid=1.0)
        expected = F1 * F2 + (1 - F1) * (1 - F2) / 3
        assert bds.fidelity(result) == pytest.approx(expected)

    def test_swapping_never_improves_fidelity(self):
        for F1 in np.linspace(0.5, 1.0, 11):
            for F2 in np.linspace(0.5, 1.0, 11):
                result = bds.fidelity(bds.swap(bds.werner(F1), bds.werner(F2)))
                assert result <= min(F1, F2) + 1e-12

    def test_swap_output_is_a_valid_state(self, rng):
        left = rng.dirichlet([6, 2, 1, 1], size=20)
        right = rng.dirichlet([5, 1, 3, 1], size=20)
        result = bds.swap(left, right, gate_fid=0.97, meas_fid=0.98)
        assert np.all(bds.is_valid(result))

    @pytest.mark.parametrize("F", [0.55, 0.7, 0.9])
    def test_purification_improves_fidelity_above_one_half(self, F):
        """Both protocols are recurrence schemes: above F = 1/2 they gain, and
        that is the entire reason to spend two pairs on one."""
        state = bds.werner(F)
        _, purified = bds.bbpssw(state, state, gate_fid=1.0, meas_fid=1.0)
        assert bds.fidelity(purified) > F

        _, purified = bds.dejmps(state, state, gate_fid=1.0, meas_fid=1.0)
        assert bds.fidelity(purified) > F

    def test_purification_does_not_improve_at_the_half_fixed_point(self):
        state = bds.werner(0.5)
        _, purified = bds.bbpssw(state, state, gate_fid=1.0, meas_fid=1.0)
        assert bds.fidelity(purified) == pytest.approx(0.5, abs=1e-9)

    def test_bbpssw_success_probability_matches_the_closed_form(self):
        """For ideal hardware, ``p = F^2 + 2F(1-F)/3 + 5((1-F)/3)^2``."""
        for F in [0.6, 0.75, 0.9, 1.0]:
            state = bds.werner(F)
            p_s_w, _ = bds.bbpssw(state, state, gate_fid=1.0, meas_fid=1.0)
            expected = F ** 2 + 2 * F * (1 - F) / 3 + 5 * ((1 - F) / 3) ** 2
            assert float(p_s_w) == pytest.approx(expected)

    def test_dejmps_beats_bbpssw_on_a_dephased_input(self):
        """A pair whose error is concentrated in one Bell component is exactly
        where twirling throws away usable structure."""
        state = np.array([0.8, 0.18, 0.01, 0.01])
        _, from_bbpssw = bds.bbpssw(state, state, gate_fid=1.0, meas_fid=1.0)
        _, from_dejmps = bds.dejmps(state, state, gate_fid=1.0, meas_fid=1.0)
        assert bds.fidelity(from_dejmps) > bds.fidelity(from_bbpssw)

    def test_bbpssw_rejects_zero_gate_fidelity(self):
        with pytest.raises(ValueError):
            bds.bbpssw(bds.werner(0.9), bds.werner(0.9), gate_fid=0.0)

    def test_decoherence_decays_to_the_maximally_mixed_floor(self):
        """Every Pauli rate positive means the state ends at 1/4 each."""
        state = bds.werner(1.0)
        late = bds.decohere(state, 1.0, 1.0, 1.0, time=1e4)
        assert np.allclose(late, 0.25, atol=1e-9)

    def test_decoherence_is_the_identity_at_zero_time(self):
        state = np.array([0.7, 0.1, 0.15, 0.05])
        assert np.allclose(bds.decohere(state, 1.0, 2.0, 3.0, time=0.0), state)

    def test_decoherence_preserves_normalisation(self, rng):
        state = rng.dirichlet([5, 1, 1, 1], size=50)
        for t in [1e-3, 0.1, 1.0, 10.0]:
            decohered = bds.decohere(state, 0.3, 0.3, 0.4, time=t)
            assert np.allclose(decohered.sum(axis=-1), 1.0)

    def test_decoherence_is_monotone_in_time(self):
        state = bds.werner(0.99)
        times = np.linspace(0.0, 5.0, 50)
        fidelities = bds.fidelity(bds.decohere(state, 0.2, 0.3, 0.5, times[:, None]))
        assert np.all(np.diff(fidelities) <= 1e-12)

    def test_both_qubits_decoheres_exactly_twice(self):
        state = bds.werner(0.95)
        once = bds.decohere(state, 0.2, 0.3, 0.5, 0.1, both_qubits=False)
        twice = bds.decohere(state, 0.2, 0.3, 0.5, 0.1, both_qubits=True)
        assert np.allclose(bds.decohere(once, 0.2, 0.3, 0.5, 0.1, both_qubits=False), twice)


# ---------------------------------------------------------------------------
# fibre
# ---------------------------------------------------------------------------

class TestFiber:

    def test_transmittance_is_one_at_zero_length(self):
        assert fiber.transmittance(0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("length_km", [1.0, 25.0, 100.0])
    def test_transmittance_follows_the_db_definition(self, length_km):
        alpha = 0.2
        assert fiber.transmittance(length_km, alpha) == pytest.approx(
            10 ** (-alpha * length_km / 10))

    def test_one_attenuation_length_loses_a_factor_of_e(self):
        assert fiber.transmittance(C.DEFAULT_ATTENUATION_LENGTH_KM) == pytest.approx(
            1 / np.e, rel=1e-9)

    def test_delay_is_about_five_microseconds_per_kilometre(self):
        assert fiber.propagation_delay_s(1.0) == pytest.approx(4.9e-6, rel=0.02)

    def test_rotation_matrices_are_orthogonal_with_unit_determinant(self, rng):
        axes = rng.normal(size=(30, 3))
        angles = rng.uniform(0, 2 * np.pi, size=30)
        matrices = fiber.rotation_matrix(axes, angles)
        identity = np.einsum("...ij,...kj->...ik", matrices, matrices)
        assert np.allclose(identity, np.eye(3), atol=1e-12)
        assert np.allclose(np.linalg.det(matrices), 1.0)

    def test_rotation_about_a_vector_leaves_it_fixed(self, rng):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotated = fiber.rotation_matrix(axis, 1.234) @ axis
        assert np.allclose(rotated, axis)

    def test_zero_angle_gives_the_identity(self):
        assert np.allclose(fiber.rotation_matrix([0, 0, 1], 0.0), np.eye(3))

    @pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, np.pi])
    def test_qber_is_sin_squared_half_the_misalignment(self, theta):
        """The defining relation between Poincare-sphere angle and bit error."""
        reference = np.array([1.0, 0.0, 0.0])
        received = fiber.rotation_matrix([0, 0, 1], theta) @ reference
        assert fiber.stokes_qber(received, reference) == pytest.approx(np.sin(theta / 2) ** 2)

    def test_fidelity_and_qber_are_complementary(self, rng):
        received = rng.normal(size=(20, 3))
        received /= np.linalg.norm(received, axis=-1, keepdims=True)
        reference = np.array([0.0, 1.0, 0.0])
        assert np.allclose(fiber.stokes_fidelity(received, reference)
                           + fiber.stokes_qber(received, reference), 1.0)

    def test_polarization_fidelity_matches_sequence_parameter_range(self):
        assert fiber.polarization_fidelity_from_angle(0.0) == pytest.approx(1.0)
        assert fiber.polarization_fidelity_from_angle(np.pi) == pytest.approx(0.0, abs=1e-12)


class TestPolarizationDrift:

    def test_same_seed_gives_the_same_trajectory(self):
        model = fiber.PolarizationDriftModel()
        first = model.simulate(60.0, 0.05, rng=7)
        second = model.simulate(60.0, 0.05, rng=7)
        assert np.allclose(first.angle_rad, second.angle_rad)

    def test_different_seeds_diverge(self):
        model = fiber.PolarizationDriftModel()
        first = model.simulate(60.0, 0.05, rng=1)
        second = model.simulate(60.0, 0.05, rng=2)
        assert not np.allclose(first.angle_rad, second.angle_rad)

    def test_stokes_vectors_stay_on_the_sphere(self):
        trajectory = fiber.PolarizationDriftModel().simulate(120.0, 0.1, rng=3)
        assert np.allclose(np.linalg.norm(trajectory.stokes, axis=-1), 1.0, atol=1e-10)

    def test_a_still_fibre_does_not_drift(self):
        """Kill every source of motion and the polarisation must not move."""
        model = fiber.PolarizationDriftModel(
            thermal_drift_rate_rad_per_s=0.0,
            thermal_per_kelvin_rad_per_s=0.0,
            vibration_amplitude_rad=0.0,
            jump_rate_per_s=0.0,
        )
        trajectory = model.simulate(100.0, 0.5, rng=4)
        assert np.allclose(trajectory.angle_rad, 0.0, atol=1e-12)
        assert np.allclose(trajectory.qber, 0.0, atol=1e-12)

    def test_vibration_only_gives_bounded_jitter_not_drift(self):
        """Vibration must not integrate into a random walk — if it does, the
        model will overstate how much a slow compensator loses."""
        amplitude = 5e-3
        model = fiber.PolarizationDriftModel(
            thermal_drift_rate_rad_per_s=0.0,
            thermal_per_kelvin_rad_per_s=0.0,
            vibration_amplitude_rad=amplitude,
            jump_rate_per_s=0.0,
        )
        trajectory = model.simulate(60.0, 1e-3, rng=5)
        first_half = trajectory.angle_rad[: len(trajectory) // 2]
        second_half = trajectory.angle_rad[len(trajectory) // 2:]
        # Same statistics in both halves means stationary, not drifting.
        assert second_half.mean() == pytest.approx(first_half.mean(), rel=0.25)
        assert trajectory.angle_rad.max() < 20 * amplitude

    def test_vibration_rms_amplitude_is_what_was_asked_for(self):
        amplitude = 4e-3
        model = fiber.PolarizationDriftModel(
            thermal_drift_rate_rad_per_s=0.0,
            thermal_per_kelvin_rad_per_s=0.0,
            vibration_amplitude_rad=amplitude,
            jump_rate_per_s=0.0,
            vibration_num_tones=16,
        )
        trajectory = model.simulate(120.0, 1e-3, rng=6)
        # Three independent axes each with RMS `amplitude`.
        expected = amplitude * np.sqrt(3)
        assert trajectory.vibration_rms.std() > 0
        assert np.sqrt((trajectory.vibration_rms ** 2).mean()) == pytest.approx(
            expected, rel=0.15)

    def test_thermal_drift_accumulates(self):
        model = fiber.PolarizationDriftModel(
            thermal_drift_rate_rad_per_s=1e-2,
            thermal_per_kelvin_rad_per_s=0.0,
            vibration_amplitude_rad=0.0,
            jump_rate_per_s=0.0,
        )
        short = model.simulate(30.0, 0.1, rng=8).angle_rad[-1]
        long = model.simulate(600.0, 0.1, rng=8).angle_rad[-1]
        assert long > short

    def test_jumps_are_recorded_and_roughly_poisson(self):
        rate = 0.5
        duration = 400.0
        model = fiber.PolarizationDriftModel(jump_rate_per_s=rate)
        counts = [len(model.simulate(duration, 1.0, rng=seed).jump_times_s)
                  for seed in range(30)]
        assert np.mean(counts) == pytest.approx(rate * duration, rel=0.25)

    def test_temperature_series_must_match_the_run_length(self):
        model = fiber.PolarizationDriftModel()
        with pytest.raises(ValueError):
            model.simulate(10.0, 1.0, temperature_c=np.zeros(3))

    def test_supplied_temperature_is_returned_untouched(self):
        model = fiber.PolarizationDriftModel()
        temperature = 15.0 + np.arange(50) * 0.01
        trajectory = model.simulate(50.0, 1.0, temperature_c=temperature, rng=9)
        assert np.allclose(trajectory.temperature_c, temperature)

    def test_temperature_coupling_actually_couples(self):
        """A model with dT/dt coupling must respond to a temperature ramp."""
        steady = np.full(400, 20.0)
        ramp = 20.0 + np.linspace(0, 8.0, 400)
        model = fiber.PolarizationDriftModel(
            thermal_drift_rate_rad_per_s=0.0,
            thermal_per_kelvin_rad_per_s=0.2,
            vibration_amplitude_rad=0.0,
            jump_rate_per_s=0.0,
        )
        quiet = model.simulate(400.0, 1.0, temperature_c=steady, rng=10).angle_rad[-1]
        driven = model.simulate(400.0, 1.0, temperature_c=ramp, rng=10).angle_rad[-1]
        assert driven > quiet

    def test_rejects_bad_step(self):
        model = fiber.PolarizationDriftModel()
        with pytest.raises(ValueError):
            model.simulate(10.0, 0.0)
        with pytest.raises(ValueError):
            model.simulate(0.0, 0.1)


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

class TestMemory:

    def test_pauli_rates_sum_to_the_decoherence_rate(self):
        rates = memory.pauli_rates(0.5, [0.2, 0.3, 0.5])
        assert sum(rates) == pytest.approx(1 / 0.5)

    def test_rejects_error_distribution_that_does_not_sum_to_one(self):
        with pytest.raises(ValueError):
            memory.MemoryParameters(decoherence_errors=(0.5, 0.5, 0.5))

    def test_rejects_non_positive_coherence_time(self):
        with pytest.raises(ValueError):
            memory.MemoryParameters(coherence_time_s=0.0)

    def test_fidelity_decays_monotonically(self):
        parameters = memory.MemoryParameters(coherence_time_s=1.0, raw_fidelity=0.95)
        times = np.linspace(0, 3, 40)
        fidelities = memory.fidelity_after(parameters.fresh_state(), times, parameters)
        assert np.all(np.diff(fidelities) <= 1e-12)
        assert fidelities[0] == pytest.approx(0.95)

    def test_time_to_fidelity_inverts_fidelity_after(self):
        parameters = memory.MemoryParameters(coherence_time_s=2.0, raw_fidelity=0.98)
        target = 0.8
        elapsed = memory.time_to_fidelity(target, parameters=parameters)
        assert memory.fidelity_after(parameters.fresh_state(), elapsed,
                                     parameters) == pytest.approx(target, abs=1e-6)

    def test_time_to_fidelity_is_zero_when_already_below(self):
        parameters = memory.MemoryParameters(raw_fidelity=0.6)
        assert memory.time_to_fidelity(0.9, parameters=parameters) == 0.0

    def test_time_to_fidelity_is_infinite_below_the_depolarised_floor(self):
        parameters = memory.MemoryParameters(raw_fidelity=0.9)
        assert memory.time_to_fidelity(0.1, parameters=parameters) == float("inf")

    def test_longer_coherence_buys_proportionally_more_time(self):
        short = memory.time_to_fidelity(
            0.8, parameters=memory.MemoryParameters(coherence_time_s=1.0))
        long = memory.time_to_fidelity(
            0.8, parameters=memory.MemoryParameters(coherence_time_s=4.0))
        assert long == pytest.approx(4 * short, rel=1e-6)

    def test_cutoff_time_follows_the_ratio(self):
        parameters = memory.MemoryParameters(coherence_time_s=2.0, cutoff_ratio=0.5)
        assert parameters.cutoff_time_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# link budget
# ---------------------------------------------------------------------------

class TestLink:

    def test_arm_efficiency_is_bounded(self):
        for length in [0.0, 10.0, 200.0]:
            eta = link.arm_efficiency(length)
            assert 0.0 <= eta <= 1.0

    def test_efficiency_falls_exponentially_with_distance(self):
        near = link.arm_efficiency(10.0)
        far = link.arm_efficiency(10.0 + C.DEFAULT_ATTENUATION_LENGTH_KM)
        assert far == pytest.approx(near / np.e, rel=1e-9)

    def test_barrett_kok_is_the_square_of_single_heralded(self):
        single = link.attempt_success_probability(50.0, "single_heralded")
        double = link.attempt_success_probability(50.0, "barrett_kok")
        assert double == pytest.approx(single ** 2)

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError):
            link.attempt_success_probability(10.0, "teleportation_by_hope")

    def test_attempt_rate_is_capped_by_the_round_trip(self):
        """Past a few tens of km the herald, not the hardware, sets the rate."""
        assert link.max_attempt_rate_hz(0.1) == pytest.approx(C.DEFAULT_ATTEMPT_RATE_HZ)
        long_span = link.max_attempt_rate_hz(200.0)
        assert long_span < C.DEFAULT_ATTEMPT_RATE_HZ
        assert long_span == pytest.approx(1.0 / fiber.propagation_delay_s(200.0))

    def test_rate_decreases_with_distance(self):
        lengths = np.array([10.0, 30.0, 60.0, 120.0])
        rates = link.entanglement_rate_hz(lengths)
        assert np.all(np.diff(rates) < 0)

    def test_expected_time_is_the_reciprocal_of_the_rate(self):
        rate = link.entanglement_rate_hz(40.0)
        assert link.expected_generation_time_s(40.0) == pytest.approx(1.0 / rate)

    def test_link_budget_is_internally_consistent(self):
        budget = link.LinkBudget.compute(75.0)
        assert budget.rate_hz == pytest.approx(
            budget.attempt_rate_hz * budget.attempt_probability)
        assert budget.expected_time_s == pytest.approx(1.0 / budget.rate_hz)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_binary_entropy_at_the_special_points(self, p):
        expected = {0.0: 0.0, 0.5: 1.0, 1.0: 0.0}[p]
        assert link.binary_entropy(p) == pytest.approx(expected)

    def test_secret_key_rate_vanishes_at_the_bb84_threshold(self):
        assert link.bb84_secret_key_rate(1e6, C.BB84_QBER_THRESHOLD) == pytest.approx(
            0.0, abs=2e4)
        assert link.bb84_secret_key_rate(1e6, 0.15) == 0.0

    def test_secret_key_rate_decreases_with_qber(self):
        rates = link.bb84_secret_key_rate(1e6, np.array([0.0, 0.01, 0.03, 0.06, 0.09]))
        assert np.all(np.diff(rates) < 0)

    def test_secret_key_rate_is_the_sifted_rate_at_zero_error(self):
        assert link.bb84_secret_key_rate(1234.0, 0.0) == pytest.approx(1234.0)
