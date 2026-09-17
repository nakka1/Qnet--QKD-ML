"""Decoy-state QKD: attacks, their detection, key-rate control, and timing.

Phase 5, and the first part of this package that is about a prepare-and-measure
link rather than an entangled one.  Four pieces, in dependency order:

:mod:`.attacks`
    Eavesdropping expressed as a modification of the channel's per-photon-number
    yields, so that what Bob observes is derived rather than asserted.

:mod:`.session`
    A link running over time — drifting fibre, finite counting statistics, and
    an event that starts when nobody is watching.  The source of every dataset
    here.

:mod:`.detection`
    Telling an attack from the weather: supervised classifiers on the published
    observables, against the QBER threshold a deployed system uses today.

:mod:`.keyrate`
    Choosing the intensities and the clock that maximise secret key under
    conditions that move, by reinforcement learning, against the static optimum.

:mod:`.timing`
    The thermal drift of the propagation delay, and keeping the detector gate on
    top of the pulse without it.

The physics they all stand on is :mod:`qnet_autopilot.physics.decoy`.
"""

from __future__ import annotations

from . import attacks, detection, keyrate, session, timing
from .attacks import (Attack, BeamSplitting, DetectorBlinding, InterceptResend,
                      NoAttack, PhotonNumberSplitting, attack_labels)
from .detection import (AttackDetector, DetectionReport, FeatureWindow,
                        QberThreshold, build_detection_features, evaluate_detector,
                        fit_forest, fit_svm)
from .keyrate import (KeyRateEnv, KeyRateEnvConfig, KeyRatePolicy, StaticPolicy,
                      run_episode, train_key_rate_policy)
from .session import (EVENTS, Event, LinkConditions, SessionConfig, event_labels,
                      simulate_session)
from .timing import (ClockModel, GateConfig, SyncResult, TimingTrajectory,
                     gate_efficiency, run_sync_loop)

__all__ = [
    "attacks", "detection", "keyrate", "session", "timing",
    "Attack", "NoAttack", "InterceptResend", "PhotonNumberSplitting",
    "DetectorBlinding", "BeamSplitting", "attack_labels",
    "LinkConditions", "Event", "SessionConfig", "EVENTS", "event_labels",
    "simulate_session",
    "FeatureWindow", "build_detection_features", "QberThreshold", "AttackDetector",
    "fit_forest", "fit_svm", "evaluate_detector", "DetectionReport",
    "KeyRateEnvConfig", "KeyRateEnv", "KeyRatePolicy", "StaticPolicy",
    "run_episode", "train_key_rate_policy",
    "ClockModel", "GateConfig", "TimingTrajectory", "SyncResult",
    "gate_efficiency", "run_sync_loop",
]
