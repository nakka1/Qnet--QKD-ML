"""Finite-key accounting and a polarisation controller that can refuse.

Both of these close gaps the phase reports named in their own limits sections,
and both are the kind of model where a plausible-looking implementation can be
quietly wrong in the favourable direction. So the tests are mostly about
limits and impossibilities:

* a finite-key bound must never exceed the asymptotic rate it sits under, must
  grow with block size, and must converge to that rate — the first version here
  failed all three because a diverging phase-error bound was clipped at 1
  instead of at 0.5, and binary entropy is symmetric about 0.5, so a *worse*
  bound produced *more* key;
* an endless polarisation controller must be able to unwind an element by a
  full turn and land somewhere optically identical. That is true of retarder
  angles and false of rotation-vector components, and building it the second
  way gives a reset that silently moves the fibre.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.compensation.actuator import (ActuatorConfig, IdealActuator,
                                                  PolarisationActuator,
                                                  retarder_angles,
                                                  retarder_rotation)
from qnet_autopilot.compensation.predictors import _exp_map
from qnet_autopilot.physics.link import (bb84_finite_key_length,
                                         bb84_finite_key_rate,
                                         bb84_secret_key_rate)


class TestFiniteKey:

    def test_it_never_exceeds_the_asymptotic_rate(self):
        """The property the first implementation violated, at every block size."""
        sizes = np.logspace(3, 13, 80)
        for qber in (0.0, 0.005, 0.02, 0.05, 0.08):
            asymptotic = float(bb84_secret_key_rate(1.0, qber))
            per_bit = bb84_finite_key_length(sizes, qber) / sizes
            assert np.all(per_bit <= asymptotic + 1e-12), f"qber {qber}"

    def test_it_grows_with_the_block(self):
        sizes = np.logspace(3, 13, 120)
        per_bit = bb84_finite_key_length(sizes, 0.02) / sizes
        assert np.all(np.diff(per_bit) > -1e-12)

    def test_it_converges_to_the_asymptotic_rate(self):
        """With the test fraction optimised, as a real protocol would."""
        for qber in (0.01, 0.02, 0.05):
            asymptotic = float(bb84_secret_key_rate(1.0, qber))
            huge = float(bb84_finite_key_length(1e13, qber)) / 1e13
            assert huge == pytest.approx(asymptotic, rel=0.01)

    def test_a_fixed_test_fraction_does_not_converge(self):
        """Which is why the fraction is searched rather than set.

        Holding it at 10% caps the bound at 90% of the asymptotic rate forever,
        because the asymptotic formula assumes error estimation is free — true
        only if the fraction shrinks as the block grows.
        """
        asymptotic = float(bb84_secret_key_rate(1.0, 0.02))
        fixed = float(bb84_finite_key_length(1e13, 0.02, test_fraction=0.1)) / 1e13
        assert fixed == pytest.approx(0.9 * asymptotic, rel=0.02)
        assert fixed < asymptotic

    def test_a_short_block_is_worth_nothing(self):
        """Even at a QBER the asymptotic formula is happy with."""
        assert float(bb84_secret_key_rate(1.0, 0.02)) > 0.6
        assert float(bb84_finite_key_length(500, 0.02)) == 0.0
        assert float(bb84_finite_key_length(2_000, 0.02)) == 0.0
        assert float(bb84_finite_key_length(100_000, 0.02)) > 0.0

    def test_it_is_zero_above_the_threshold(self):
        for qber in (0.11, 0.15, 0.3, 0.5):
            assert float(bb84_finite_key_length(1e12, qber)) == 0.0

    def test_it_decreases_with_the_error_rate(self):
        rates = bb84_finite_key_length(1e9, np.linspace(0.0, 0.09, 25))
        assert np.all(np.diff(rates) < 0)

    def test_tighter_security_costs_key(self):
        loose = float(bb84_finite_key_length(1e6, 0.02, epsilon_secrecy=1e-6))
        tight = float(bb84_finite_key_length(1e6, 0.02, epsilon_secrecy=1e-15))
        assert tight < loose

    def test_the_rate_form_is_the_length_over_the_duration(self):
        length = float(bb84_finite_key_length(2e5, 0.02))
        rate = float(bb84_finite_key_rate(1e5, 0.02, 2.0))
        assert rate == pytest.approx(length / 2.0, rel=1e-12)

    def test_longer_blocks_yield_more_at_the_same_rate(self):
        rates = [float(bb84_finite_key_rate(1e5, 0.02, duration))
                 for duration in (0.01, 0.1, 1.0, 10.0, 100.0)]
        assert np.all(np.diff(rates) > 0)

    def test_it_rejects_impossible_parameters(self):
        for bad in (dict(epsilon_secrecy=0.0), dict(epsilon_secrecy=1.0),
                    dict(epsilon_correctness=-1.0), dict(test_fraction=0.0),
                    dict(test_fraction=1.0)):
            with pytest.raises(ValueError):
                bb84_finite_key_length(1e6, 0.02, **bad)


class TestRetarderParametrisation:

    def test_decomposition_round_trips(self):
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(500):
            angles = rng.uniform(-np.pi, np.pi, size=3)
            rotation = retarder_rotation(angles)
            worst = max(worst, float(np.max(np.abs(
                retarder_rotation(retarder_angles(rotation)) - rotation))))
        assert worst < 1e-10

    def test_a_full_turn_on_one_element_is_the_identity(self):
        """What makes an endless reset possible, and correct."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            angles = rng.uniform(-np.pi, np.pi, size=3)
            rotation = retarder_rotation(angles)
            for axis in range(3):
                turned = angles.copy()
                turned[axis] += 2.0 * np.pi
                assert np.allclose(retarder_rotation(turned), rotation, atol=1e-12)

    def test_a_rotation_vector_would_get_this_wrong(self):
        """The reason the state is retarder angles and not a rotation vector.

        This is not a style choice: writing the reset in the wrong
        parametrisation moves the fibre by a third of a radian while claiming
        to be optically neutral.
        """
        vector = np.array([0.4, 0.3, 0.2])
        turned = vector + np.array([2.0 * np.pi, 0.0, 0.0])
        assert not np.allclose(_exp_map(turned), _exp_map(vector), atol=1e-3)

    def test_it_returns_the_solution_nearest_the_current_setting(self):
        rng = np.random.default_rng(2)
        angles = rng.uniform(-np.pi, np.pi, size=3)
        rotation = retarder_rotation(angles)
        far = angles + np.array([4.0 * np.pi, 0.0, -2.0 * np.pi])
        near = retarder_angles(rotation, near=far)
        assert np.allclose(retarder_rotation(near), rotation, atol=1e-10)
        assert np.sum(np.abs(near - far)) < np.sum(np.abs(angles - far))

    def test_the_degenerate_case_does_not_move_the_element_it_need_not(self):
        """Gimbal lock: only alpha + gamma is determined at beta = 0."""
        rotation = retarder_rotation([0.7, 0.0, 0.0])
        current = np.array([0.0, 0.0, 1.3])
        angles = retarder_angles(rotation, near=current)
        assert np.allclose(retarder_rotation(angles), rotation, atol=1e-10)
        assert angles[2] == pytest.approx(1.3)


class TestActuator:

    def test_it_rejects_impossible_hardware(self):
        for bad in (dict(range_rad=0.0), dict(slew_rate_rad_per_s=0.0),
                    dict(reset_slew_multiplier=-1.0), dict(range_rad=np.pi)):
            with pytest.raises(ValueError):
                ActuatorConfig(**bad)

    def test_unbounded_and_infinitely_fast_is_the_ideal_actuator(self):
        rng = np.random.default_rng(3)
        actuator = PolarisationActuator(
            ActuatorConfig(range_rad=np.inf, slew_rate_rad_per_s=1e9))
        for _ in range(200):
            rotation = retarder_rotation(rng.uniform(-1.0, 1.0, size=3))
            assert np.allclose(actuator.apply(rotation, 1e-3), rotation, atol=1e-9)
        assert actuator.resets == 0

    def test_the_slew_rate_bounds_how_far_it_moves(self):
        actuator = PolarisationActuator(
            ActuatorConfig(slew_rate_rad_per_s=1.0, range_rad=np.inf))
        target = retarder_rotation([1.0, 0.0, 0.0])
        actuator.apply(target, 0.1)
        assert actuator.angles[0] == pytest.approx(0.1, abs=1e-9)
        assert not np.allclose(retarder_rotation(actuator.angles), target)

    def test_it_arrives_eventually(self):
        actuator = PolarisationActuator(
            ActuatorConfig(slew_rate_rad_per_s=1.0, range_rad=np.inf))
        target = retarder_rotation([0.5, 0.3, -0.2])
        for _ in range(400):
            actuator.apply(target, 0.01)
        assert np.allclose(retarder_rotation(actuator.angles), target, atol=1e-6)

    def test_it_resets_when_an_element_runs_out_of_travel(self):
        """Driven steadily in one direction, it must unwind rather than stall."""
        actuator = PolarisationActuator(
            ActuatorConfig(range_rad=2.0 * np.pi, slew_rate_rad_per_s=50.0))
        angle = 0.0
        for _ in range(4000):
            angle += 0.01
            actuator.apply(retarder_rotation([angle, 0.2, 0.0]), 1e-3)
        assert actuator.resets > 0
        assert abs(actuator.angles[0]) <= 2.0 * np.pi + 1e-6

    def test_a_reset_ends_somewhere_optically_equivalent(self):
        """The whole justification for unwinding: a full turn changes nothing."""
        config = ActuatorConfig(range_rad=2.0 * np.pi, slew_rate_rad_per_s=50.0)
        actuator = PolarisationActuator(config)
        angle, before, after = 0.0, None, None
        for _ in range(6000):
            angle += 0.01
            target = retarder_rotation([angle, 0.2, 0.0])
            was_resetting = actuator._resetting.any()
            applied = actuator.apply(target, 1e-3)
            if not was_resetting and actuator._resetting.any():
                before = applied
            if was_resetting and not actuator._resetting.any() and after is None:
                after = applied
                break
        assert before is not None and after is not None, "no reset completed"
        # The element unwound a full turn while the fibre kept drifting, so the
        # two are close rather than equal — the point is that the *device* did
        # not jump, which a wrong parametrisation would show as a large step.
        assert np.max(np.abs(after - before)) < 0.35

    def test_travel_accumulates_and_wandering_commands_cost_more(self):
        rng = np.random.default_rng(4)
        steady = PolarisationActuator(ActuatorConfig(range_rad=np.inf,
                                                     slew_rate_rad_per_s=1e6))
        jittery = PolarisationActuator(ActuatorConfig(range_rad=np.inf,
                                                      slew_rate_rad_per_s=1e6))
        for step in range(300):
            steady.apply(retarder_rotation([0.001 * step, 0.1, 0.0]), 1e-3)
            jittery.apply(retarder_rotation(
                [0.001 * step + rng.normal(0.0, 0.2), 0.1, 0.0]), 1e-3)
        assert jittery.travel_rad > 3 * steady.travel_rad

    def test_the_ideal_actuator_reports_no_wear(self):
        actuator = IdealActuator()
        rotation = retarder_rotation([0.3, 0.2, 0.1])
        assert np.array_equal(actuator.apply(rotation, 1e-3), rotation)
        assert actuator.resets == 0
        assert actuator.travel_rad == 0.0


class TestClosedLoopWithActuator:

    def trajectory(self):
        from qnet_autopilot.compensation.residual import ResidualDriftModel  # noqa
        from qnet_autopilot.physics.fiber import PolarizationDriftModel

        return PolarizationDriftModel(
            thermal_drift_rate_rad_per_s=0.05, thermal_correlation_time_s=60.0,
            vibration_amplitude_rad=0.30, vibration_frequency_hz=40.0,
            vibration_bandwidth_hz=6.0, jump_rate_per_s=0.0,
            thermal_per_kelvin_rad_per_s=0.0,
        ).simulate(1.0, 1e-3, rng=0, track_rotation=True)

    def control(self):
        from qnet_autopilot.compensation import ControlConfig, MonitoringConfig

        return ControlConfig(control_period_s=2e-3, latency_s=20e-3,
                             monitoring=MonitoringConfig(photons_per_basis=400))

    def test_an_ideal_actuator_changes_nothing(self):
        """So the bounded case is compared against the same code path."""
        from qnet_autopilot.compensation import Persistence, run_closed_loop

        trajectory, control = self.trajectory(), self.control()
        without = run_closed_loop(trajectory, Persistence(), control, rng=0)
        with_ideal = run_closed_loop(trajectory, Persistence(), control, rng=0,
                                     actuator=IdealActuator())
        assert np.array_equal(without.qber, with_ideal.qber)

    def test_an_unbounded_fast_device_reproduces_the_ideal_one(self):
        from qnet_autopilot.compensation import Persistence, run_closed_loop

        trajectory, control = self.trajectory(), self.control()
        ideal = run_closed_loop(trajectory, Persistence(), control, rng=0)
        generous = run_closed_loop(
            trajectory, Persistence(), control, rng=0,
            actuator=PolarisationActuator(
                ActuatorConfig(range_rad=np.inf, slew_rate_rad_per_s=1e9)))
        assert np.allclose(ideal.qber, generous.qber, atol=1e-9)

    def test_a_slow_device_hurts_a_compensator_worth_applying(self):
        from qnet_autopilot.compensation import AlphaBetaFilter, run_closed_loop

        trajectory, control = self.trajectory(), self.control()
        rates = [run_closed_loop(
            trajectory, AlphaBetaFilter(alpha=0.05, beta=0.005), control, rng=0,
            actuator=PolarisationActuator(
                ActuatorConfig(slew_rate_rad_per_s=slew))).summary()["mean_qber"]
            for slew in (20.0, 100.0, 5000.0)]
        assert rates[0] > rates[1] > rates[2]

    def test_a_slow_device_can_rescue_a_compensator_that_is_making_it_worse(self):
        """Not a bug, and worth pinning down: throttling a harmful correction helps.

        Phase 3 measured that at 40 ms of loop latency, persistence is *worse*
        than leaving the fibre alone — it applies a correction the vibration has
        already invalidated. A rate-limited actuator low-pass filters that
        correction, so it does less damage. The general claim "a slower device
        is worse" is therefore false; the true one is that a slower device
        delivers less of whatever the compensator asked for, which is a loss
        only when the request was good.
        """
        from qnet_autopilot.compensation import Persistence, run_closed_loop

        trajectory, control = self.trajectory(), self.control()
        unbounded = run_closed_loop(trajectory, Persistence(), control,
                                    rng=0).summary()["mean_qber"]
        throttled = run_closed_loop(
            trajectory, Persistence(), control, rng=0,
            actuator=PolarisationActuator(
                ActuatorConfig(slew_rate_rad_per_s=100.0))).summary()["mean_qber"]
        assert throttled < unbounded

    def test_the_finite_key_rate_never_beats_the_asymptotic_one(self):
        from qnet_autopilot.compensation import Persistence, run_closed_loop

        result = run_closed_loop(self.trajectory(), Persistence(),
                                 self.control(), rng=0)
        asymptotic = result.summary()["mean_key_rate_hz"]
        finite = result.finite_key_rate_hz(1.0e5, block_duration_s=1.0)
        assert 0.0 <= finite <= asymptotic + 1e-9
