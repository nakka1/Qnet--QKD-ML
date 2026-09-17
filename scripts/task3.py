"""The phase-3 benchmark: an aerial fibre with a slow polarisation actuator.

One place for the operating point, so training and evaluation cannot drift apart.

Why these numbers
-----------------
At any realistic control latency the **thermal** drift is trivially trackable:
0.05 rad/s over 40 ms is 2 milliradians, far below the measurement noise. What
actually challenges a compensator is the **acoustic band** — mechanical
vibration moves the state of polarisation by a large fraction of a radian
within one loop delay, and no amount of smoothing recovers a signal the
actuator is 40 ms behind.

So the operating point is chosen where compensation is genuinely hard and
genuinely necessary: aerial fibre under mechanical disturbance, driven by an
actuator slow enough that prediction is the only thing that helps. Both the
vibration amplitude and the latency are swept in the report, because neither is
a number anyone should take on faith.
"""
from __future__ import annotations

import numpy as np

from qnet_autopilot.compensation import ControlConfig, MonitoringConfig
from qnet_autopilot.compensation.predictors import FeatureSpec
from qnet_autopilot.physics.fiber import PolarizationDriftModel

#: Aerial fibre under mechanical disturbance.
DRIFT = dict(
    thermal_drift_rate_rad_per_s=0.05,     # aerial, an order above buried metro
    thermal_correlation_time_s=60.0,
    vibration_amplitude_rad=0.30,          # strong acoustic-band disturbance
    vibration_frequency_hz=40.0,
    vibration_bandwidth_hz=6.0,
    vibration_num_tones=8,
    jump_rate_per_s=0.05,                  # a cable disturbed every ~20 s
    jump_scale_rad=0.35,
    thermal_per_kelvin_rad_per_s=0.2,
)

SAMPLE_INTERVAL_S = 1.0e-3
CONTROL_PERIOD_S = 2.0e-3
LATENCY_S = 40.0e-3
DURATION_S = 20.0

#: 16 control steps is 32 ms, longer than one period of the vibration band —
#: the window has to span a period or the oscillation is invisible to the model.
FEATURE_SPEC = FeatureSpec(window=16, use_covariates=True,
                           horizon_scale=CONTROL_PERIOD_S)

#: Tuned on validation trajectories; see scripts/tune_alpha_beta.py.
ALPHA_BETA = dict(alpha=0.1, beta=0.0)

TRAIN_SEEDS = tuple(range(0, 12))
VALIDATION_SEEDS = tuple(range(100, 104))
TEST_SEEDS = tuple(range(500, 512))


def make_model(**overrides) -> PolarizationDriftModel:
    parameters = dict(DRIFT)
    parameters.update(overrides)
    return PolarizationDriftModel(**parameters)


def make_control(latency_s: float = LATENCY_S, photons_per_basis: int = 400
                 ) -> ControlConfig:
    return ControlConfig(control_period_s=CONTROL_PERIOD_S, latency_s=latency_s,
                         monitoring=MonitoringConfig(photons_per_basis=photons_per_basis),
                         sifted_rate_hz=1.0e5)


def make_trajectories(seeds, duration_s: float = DURATION_S, **overrides):
    model = make_model(**overrides)
    return [model.simulate(duration_s, SAMPLE_INTERVAL_S, rng=int(seed),
                           track_rotation=True) for seed in seeds]
