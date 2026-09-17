"""Real-time polarisation compensation: monitoring, control loop, predictors."""

from __future__ import annotations

from . import (actuator, closed_loop, heteroscedastic, monitoring, predictors, report,
               residual, sequence)
from .actuator import (ActuatorConfig, IdealActuator, PolarisationActuator,
                       retarder_angles, retarder_rotation)
from .closed_loop import (BB84_BASES, Compensator, ControlConfig, LoopResult, bb84_qber,
                          run_closed_loop)
from .monitoring import MonitoringConfig, measure_rotation
from .predictors import (AlphaBetaFilter, FeatureSpec, LearnedCompensator,
                         LinearExtrapolation, NoCompensation, Persistence,
                         Standardiser, build_features)
from .report import render_phase3_report
from .residual import COMPENSATED, DEGRADED, ResidualDriftModel
from .sequence import (CausalConv1d, SequenceRegressor, SequenceSpec, TemporalConvNet,
                       fit_sequence)

__all__ = [
    "actuator", "closed_loop", "heteroscedastic", "monitoring", "predictors",
    "report", "residual", "sequence",
    "CausalConv1d", "SequenceSpec", "TemporalConvNet", "SequenceRegressor",
    "fit_sequence",
    "ActuatorConfig", "IdealActuator", "PolarisationActuator",
    "retarder_angles", "retarder_rotation",
    "ResidualDriftModel", "COMPENSATED", "DEGRADED",
    "render_phase3_report",
    "MonitoringConfig", "measure_rotation",
    "ControlConfig", "Compensator", "LoopResult", "run_closed_loop", "bb84_qber",
    "BB84_BASES",
    "NoCompensation", "Persistence", "LinearExtrapolation", "AlphaBetaFilter",
    "LearnedCompensator", "FeatureSpec", "Standardiser", "build_features",
]
