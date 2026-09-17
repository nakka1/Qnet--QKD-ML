"""Hybrid classical/quantum allocation: one dial per link, and what it costs."""

from __future__ import annotations

from . import env, forecast, policies, report, rollout, scenario, state
from .env import (AllocationConfig, HybridAllocationEnv, Observation, StepResult,
                  visibility_fidelity)
from .forecast import (Forecaster, OracleForecaster, PersistenceForecaster,
                       RidgeForecaster, build_windows, drift_estimate,
                       fit_ridge_forecaster)
from .policies import (AllocationPolicy, ClassicalFirst, PredictiveAllocator,
                       QuantumFirst, ReactiveThreshold, StaticAdmission)
from .report import render_phase4_report
from .rollout import EpisodeResult, EvaluationResult, evaluate, run_episode
from .scenario import build_demands, offered_load, shortest_paths
from .state import SERVICES, AllocationState, Demand, DemandProgress, LinkProfile

__all__ = [
    "env", "forecast", "policies", "report", "rollout", "scenario", "state",
    "render_phase4_report",
    "AllocationConfig", "HybridAllocationEnv", "Observation", "StepResult",
    "visibility_fidelity",
    "build_demands", "shortest_paths", "offered_load",
    "LinkProfile", "Demand", "DemandProgress", "AllocationState", "SERVICES",
    "Forecaster", "PersistenceForecaster", "RidgeForecaster", "OracleForecaster",
    "drift_estimate", "build_windows", "fit_ridge_forecaster",
    "AllocationPolicy", "ClassicalFirst", "QuantumFirst", "StaticAdmission",
    "ReactiveThreshold", "PredictiveAllocator",
    "EpisodeResult", "EvaluationResult", "run_episode", "evaluate",
]
