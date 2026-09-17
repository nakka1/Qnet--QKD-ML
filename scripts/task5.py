"""The phase-5 benchmark: a decoy-state link, watched and driven.

One place for the operating point, so the dataset generator, the detector, the
key-rate agent and the synchroniser cannot drift apart.

Why these numbers
-----------------
**The span.**  50 km of buried metropolitan fibre with a decibel of excess loss
is the case where all four problems are live at once: the gain is high enough
that the detectors are within sight of saturation, low enough that the decoy
bound is noisy on a one-second block, and the pulse has spread far enough that
the clock has a ceiling.  Shorter and the link is saturation-limited; longer and
it is dark-count-limited and every question answers itself.

**The settling period.**  One second.  It is what a deployed system does, and it
is what makes the single-photon bound noisy: a gigahertz clock gives 10\\ :sup:`9`
pulses, of which the weak decoy gets a fifth, of which a few per thousand
produce a click.  The bound differences two exponentially amplified gains, so
that is not nearly as many counts as it sounds.

**The compensator underneath.**  ``residual_fraction`` 0.2 — a deliberately
pessimistic reading of phase 3.  The attack classes have to be told apart from
polarisation drift, and drift that has been compensated away is not a
discriminator problem at all.  The report sweeps it.

**The seeds.**  Training, validation and test ranges are disjoint everywhere, and
the detector splits by *episode* rather than by row: rows inside an episode share
a baseline, a span length and a drift realisation.
"""
from __future__ import annotations

import numpy as np

from qnet_autopilot.physics.decoy import DecoySettings
from qnet_autopilot.qkd.detection import FeatureWindow
from qnet_autopilot.qkd.keyrate import (KeyRateEnv, KeyRateEnvConfig,
                                        commissioned_settings)
from qnet_autopilot.qkd.session import LinkConditions, SessionConfig
from qnet_autopilot.qkd.timing import ClockModel, GateConfig, TimingBudget

#: The span, as commissioned.
LINK = LinkConditions(
    length_km=50.0,
    excess_loss_db=1.0,
    coupling_efficiency=0.85,
    detector_efficiency=0.7,
    dark_count_rate_hz=10.0,
    misalignment_error=0.015,
)

#: The transmitter, before anyone optimises it.
SETTINGS = DecoySettings(signal_intensity=0.5, decoy_intensity=0.1,
                         signal_probability=0.7, vacuum_probability=0.1,
                         clock_rate_hz=1.0e9)

BUDGET = TimingBudget()

#: Detector dead time and afterpulsing — what makes the clock a trade.
DEAD_TIME_S = 50.0e-9
AFTERPULSE_PROBABILITY = 0.01

# --- detection ---------------------------------------------------------------

PERIOD_S = 1.0
SESSION_DURATION_S = 600.0
BASELINE_PERIODS = 60
MIN_ACTIVE_PERIODS = 60
RESIDUAL_FRACTION = 0.2
LENGTHS_KM = (20.0, 35.0, 50.0, 70.0)

FEATURE_WINDOW = FeatureWindow(window=30, baseline_periods=BASELINE_PERIODS)

#: Beam splitting leaves no signature at all, by construction.  It is generated
#: and scored, and it is *not* trained on: a class whose features are identical
#: to another class's teaches a classifier only to split noise, and it costs the
#: ``normal`` class precision to do it.
UNTRAINABLE_LABELS = ("beam_splitting",)

DATASET_SEED = 0
DETECTOR_SEED = 0

# --- key rate ----------------------------------------------------------------

KEYRATE_STEPS = 240
TRAIN_SEEDS = tuple(range(0, 10_000))
VALIDATION_SEED = 700_000
TEST_SEEDS = tuple(range(900_000, 900_048))

# --- timing ------------------------------------------------------------------

SYNC_DURATION_S = 3600.0
SYNC_PERIOD_S = 1.0
SYNC_LATENCY_S = 2.0
SYNC_TRAIN_SEEDS = tuple(range(0, 8))
SYNC_TEST_SEEDS = tuple(range(500, 512))


def make_session_config(length_km: float = LINK.length_km,
                        residual_fraction: float = RESIDUAL_FRACTION,
                        **overrides) -> SessionConfig:
    """The session used for the attack datasets."""
    from dataclasses import replace

    parameters = dict(link=replace(LINK, length_km=length_km), settings=SETTINGS,
                      period_s=PERIOD_S, duration_s=SESSION_DURATION_S,
                      baseline_periods=BASELINE_PERIODS,
                      min_active_periods=MIN_ACTIVE_PERIODS,
                      residual_fraction=residual_fraction)
    parameters.update(overrides)
    return SessionConfig(**parameters)


def make_commissioned(length_km: float = LINK.length_km) -> DecoySettings:
    """The static operating point an engineer would leave the link at."""
    from dataclasses import replace

    return commissioned_settings(replace(LINK, length_km=length_km), SETTINGS,
                                 BUDGET, DEAD_TIME_S, AFTERPULSE_PROBABILITY)


def make_keyrate_config(length_km: float = LINK.length_km,
                        initial: DecoySettings | None = None,
                        **overrides) -> KeyRateEnvConfig:
    """The key-rate environment's configuration."""
    from dataclasses import replace

    parameters = dict(link=replace(LINK, length_km=length_km),
                      initial=initial or make_commissioned(length_km),
                      budget=BUDGET, period_s=PERIOD_S, steps=KEYRATE_STEPS,
                      dead_time_s=DEAD_TIME_S,
                      afterpulse_probability=AFTERPULSE_PROBABILITY)
    parameters.update(overrides)
    return KeyRateEnvConfig(**parameters)


def make_keyrate_env(length_km: float = LINK.length_km, rng=0, **overrides) -> KeyRateEnv:
    """A fresh key-rate environment at the benchmark operating point."""
    return KeyRateEnv(make_keyrate_config(length_km, **overrides), rng=rng)


def make_clock_model(length_km: float = LINK.length_km, **overrides) -> ClockModel:
    """The delay-drift model for the synchronisation study."""
    parameters = dict(length_km=length_km)
    parameters.update(overrides)
    return ClockModel(**parameters)


def make_gate(width_s: float = LINK.gate_width_s, photons: int = 2000) -> GateConfig:
    """The gate and its control loop."""
    from qnet_autopilot.qkd.timing import MeasurementConfig

    return GateConfig(width_s=width_s, update_period_s=SYNC_PERIOD_S,
                      latency_s=SYNC_LATENCY_S,
                      measurement=MeasurementConfig(photons=photons),
                      budget=BUDGET)


def make_timing_trajectories(seeds, length_km: float = LINK.length_km,
                             duration_s: float = SYNC_DURATION_S, **overrides):
    """One delay-drift history per seed."""
    model = make_clock_model(length_km, **overrides)
    return [model.simulate(duration_s, SYNC_PERIOD_S, rng=int(seed)) for seed in seeds]
