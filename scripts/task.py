"""The phase-2 benchmark task, in one place so training and evaluation agree."""
from __future__ import annotations
import numpy as np
from qnet_autopilot.network import scenario
from qnet_autopilot.env import EnvironmentConfig, QuantumNetworkEnv, Request
from qnet_autopilot.physics.memory import MemoryParameters

GRAPH = scenario("chain", num_nodes=4, span_km=20)
MEMORY_SIZE = 6

def make_config(shaping_weight: float = 0.0,
                heralding_round_trips: float = 1.0) -> EnvironmentConfig:
    """The benchmark configuration.

    Args:
        shaping_weight: potential-based shaping, for training only.
        heralding_round_trips: span traversals charged per heralding attempt.
            One is what phase 2 was published with — the photon flight alone.
            ``MEASURED_HERALDING_ROUND_TRIPS`` is what SeQUeNCe's protocol
            actually costs, measured against it; see
            ``scripts/evaluate_sequence.py``.
    """
    return EnvironmentConfig(
        graph=GRAPH,
        heralding_round_trips=heralding_round_trips,
        memory=MemoryParameters(coherence_time_s=1.0, raw_fidelity=0.90, cutoff_ratio=0.1),
        gate_fid=0.999, meas_fid=0.999,
        coupling_efficiency=0.7, detector_efficiency=0.85,
        generate_budget_s=0.004, wait_slot_s=0.002,
        max_steps=120, max_pairs_per_link=3,
        shaping_weight=shaping_weight, gamma=0.995)

def make_env(shaping_weight: float = 0.0, rng=0,
             heralding_round_trips: float = 1.0) -> QuantumNetworkEnv:
    return QuantumNetworkEnv(make_config(shaping_weight, heralding_round_trips),
                             rng=rng)

def make_request(rng: np.random.Generator, deadline_scale: float = 1.0) -> Request:
    """Threshold and deadline both vary — that is what a fixed rule cannot track.

    Args:
        rng: draws the threshold and the deadline.
        deadline_scale: multiplies the deadline. The published benchmark uses
            one. Charging the measured heralding cost makes every attempt four
            times more expensive, and a deadline that was generous at the old
            cost is not at the new one; scaling it is the control that
            separates "the agent's advantage was an artefact of the cheap cost
            model" from "the advantage lives in a regime the corrected cost
            moves further out".
    """
    return Request(source="0", destination="3",
                   min_fidelity=float(rng.choice([0.70, 0.74, 0.78])),
                   deadline_s=float(rng.uniform(0.040, 0.090)) * deadline_scale)
