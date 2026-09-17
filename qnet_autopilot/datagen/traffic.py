"""Hybrid classical/quantum traffic, and the time-varying capacity it competes for.

The allocation problem
----------------------
Classical data and quantum signals share the same fibre plant.  Classical
traffic is elastic: it can be delayed, re-routed, or served at a lower rate,
and it degrades gracefully.  Quantum traffic is not.  A QKD session on a link
whose QBER has drifted past ~11% produces *no* key at all — not less key —
and an entanglement request whose pairs decohere past the fidelity threshold
delivers nothing.  So the scheduler is not balancing two similar things; it is
protecting a brittle service from an elastic one.

What makes it a learning problem rather than a linear program is that link
capacity for the quantum service is **not constant**.  It moves with the
polarisation state of the fibre, on the timescales
:mod:`qnet_autopilot.physics.fiber` models: a link that yields 40 kbit/s of
secret key now may yield zero in four minutes and recover ten minutes later.
A static priority rule wastes the good windows; a predictive one uses them.

So this generator produces two things in step:

* a **demand stream** — classical flows and quantum sessions arriving,
  each with a size, a priority and a deadline;
* a **capacity trace** — per-link secret-key capacity over time, derived from
  an actual polarisation drift simulation rather than an i.i.d. random number,
  so it has the autocorrelation and the jump structure a predictor could
  exploit.

Nothing here decides anything.  It emits the state a scheduler observes and the
demands it must serve; the policy is the next phase's problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ..physics import constants as C
from ..physics.fiber import PolarizationDriftModel
from ..physics.link import bb84_secret_key_rate, entanglement_rate_hz
from .schema import Dataset, DatasetMetadata, describe_columns

__all__ = [
    "TrafficProfile",
    "generate_capacity_trace",
    "generate_demand_stream",
    "CAPACITY_COLUMNS",
    "DEMAND_COLUMNS",
]

CAPACITY_COLUMNS = {
    "time_s": "sample time",
    "link": "edge identifier, 'u|v'",
    "length_km": "span length",
    "qber": "current QBER on the link, from polarisation drift plus a detector floor",
    "sifted_rate_hz": "sifted key rate the link can sustain right now",
    "secret_key_rate_hz": "asymptotic BB84 secret key rate — zero above the QBER threshold",
    "entanglement_rate_hz": "heralded pair rate for the entanglement service",
    "usable": "whether the link is currently producing any secret key at all",
    "classical_load": "fraction of the link's classical capacity currently in use",
}

DEMAND_COLUMNS = {
    "demand_id": "unique identifier",
    "arrival_s": "time the demand arrives",
    "service": "'classical', 'qkd' or 'entanglement'",
    "source": "origin node",
    "destination": "target node",
    "volume": "bits for classical and qkd; number of pairs for entanglement",
    "min_rate_hz": "rate below which the demand is not worth serving",
    "min_fidelity": "fidelity threshold — entanglement only, NaN otherwise",
    "deadline_s": "absolute time by which the demand must be met",
    "priority": "1 (best effort) to 3 (must not be dropped)",
}


@dataclass
class TrafficProfile:
    """Arrival statistics for the three services.

    Attributes:
        classical_arrivals_per_s: Poisson rate of classical flows.
        qkd_arrivals_per_s: Poisson rate of QKD sessions.  Much rarer than
            classical traffic and much longer lived — a key-establishment
            session runs for minutes.
        entanglement_arrivals_per_s: Poisson rate of entanglement requests
            (distributed computing, sensing, teleportation).
        diurnal_amplitude: fraction by which arrival rates swing over a day.
            Real networks are not stationary, and a policy trained on
            stationary traffic learns the wrong thing about headroom.
        classical_volume_mean_bits: mean flow size; exponentially distributed.
        qkd_volume_mean_bits: mean key material requested per session.
        entanglement_pairs_mean: mean pairs requested.
        min_fidelity_choices: fidelity thresholds entanglement requests draw
            from.  A spread matters: a request at 0.99 and one at 0.8 are
            different problems on the same link.
        deadline_slack: deadline as a multiple of the time the demand would
            take at its minimum rate.  Small values make the problem tight.
    """

    classical_arrivals_per_s: float = 2.0
    qkd_arrivals_per_s: float = 0.05
    entanglement_arrivals_per_s: float = 0.1
    diurnal_amplitude: float = 0.4
    classical_volume_mean_bits: float = 5.0e8
    qkd_volume_mean_bits: float = 1.0e6
    entanglement_pairs_mean: float = 50.0
    min_fidelity_choices: Sequence[float] = (0.80, 0.90, 0.95, 0.99)
    deadline_slack: float = 3.0

    def as_dict(self) -> dict:
        data = asdict(self)
        data["min_fidelity_choices"] = list(self.min_fidelity_choices)
        return data


def _link_key(u, v) -> str:
    return f"{u}|{v}"


def generate_capacity_trace(graph: nx.Graph,
                            duration_s: float = 1800.0,
                            dt_s: float = 5.0,
                            sifted_rate_hz: float = 1.0e5,
                            detector_qber_floor: float = 0.005,
                            drift_model: PolarizationDriftModel | None = None,
                            classical_load_mean: float = 0.5,
                            seed: int = 0) -> Dataset:
    """Per-link capacity over time, driven by simulated polarisation drift.

    Each edge gets its **own** drift realisation — links in different ducts do
    not share a thermal environment, and a trace that made them move together
    would let a policy cheat by predicting one link from another.

    Args:
        graph: topology with ``length`` on every edge.
        duration_s: length of the trace.
        dt_s: sampling interval.  Coarser than the compensator's update period:
            an allocator reacts on seconds, not milliseconds.
        sifted_rate_hz: sifted key rate at zero loss, before the QBER penalty.
            Scaled per link by the span's transmittance.
        detector_qber_floor: QBER contributed by dark counts and imperfect
            optics, independent of polarisation.  Without it a perfectly
            aligned link shows zero error, which no real link does.
        drift_model: polarisation model; defaults to
            :class:`~qnet_autopilot.physics.fiber.PolarizationDriftModel`.
        classical_load_mean: mean utilisation of the classical capacity, as a
            fraction.  Modelled as a slowly varying process, not white noise.
        seed: base seed; each link gets its own derived seed.

    Returns:
        Dataset: one row per link per sample.
    """
    drift_model = drift_model or PolarizationDriftModel()

    # An allocator samples every few seconds; the vibration band is hundreds of
    # hertz. Point-sampling that band at this cadence does not represent it —
    # it aliases it, and worse, the aliased beats are similar enough across
    # links to look like shared structure that a policy could learn from and
    # that does not exist. What a real allocator sees over a 5-second window is
    # the *average* over hundreds of vibration cycles, so the jitter enters as
    # a small constant bias, not as a sampled signal.
    #
    # For small angles the QBER contribution is E[sin^2(theta/2)] ~ E[theta^2]/4,
    # and only the two components perpendicular to the launch state displace it,
    # so E[theta^2] = 2a^2 for per-axis RMS amplitude a.
    vibration_amplitude = drift_model.vibration_amplitude_rad
    vibration_qber_bias = 0.5 * vibration_amplitude ** 2
    slow_model = replace(drift_model, vibration_amplitude_rad=0.0)

    frames = []

    for index, (u, v, data) in enumerate(graph.edges(data=True)):
        rng = np.random.default_rng(seed + 1000 * index)
        trajectory = slow_model.simulate(duration_s, dt_s, rng=rng)

        length_km = float(data["length"])
        qber = np.clip(trajectory.qber + detector_qber_floor + vibration_qber_bias,
                       0.0, 1.0)

        # Sifted rate scales with how many photons survive the span.
        from ..physics.fiber import transmittance
        link_sifted = sifted_rate_hz * float(transmittance(length_km))
        secret = bb84_secret_key_rate(link_sifted, qber)
        pair_rate = float(entanglement_rate_hz(length_km))

        # Classical load: a slow random walk in (0, 1), not white noise —
        # utilisation is autocorrelated over minutes in any real network.
        steps = rng.normal(0.0, 0.05, size=len(trajectory))
        walk = np.cumsum(steps)
        walk -= walk.mean()
        load = np.clip(classical_load_mean + 0.25 * np.tanh(walk), 0.0, 1.0)

        frames.append(pd.DataFrame({
            "time_s": trajectory.time_s,
            "link": _link_key(u, v),
            "length_km": length_km,
            "qber": qber,
            "sifted_rate_hz": link_sifted,
            "secret_key_rate_hz": secret,
            "entanglement_rate_hz": pair_rate,
            "usable": secret > 0.0,
            "classical_load": load,
        }))

    frame = pd.concat(frames, ignore_index=True)

    metadata = DatasetMetadata(
        name="hybrid_capacity",
        description=("Per-link, per-timestep capacity for the quantum and classical "
                     "services, driven by an independent polarisation drift "
                     "simulation on every edge."),
        parameters={
            "graph": graph.graph.get("name", "unnamed"),
            "num_links": graph.number_of_edges(),
            "duration_s": duration_s,
            "dt_s": dt_s,
            "sifted_rate_hz": sifted_rate_hz,
            "detector_qber_floor": detector_qber_floor,
            "classical_load_mean": classical_load_mean,
            "vibration_qber_bias": vibration_qber_bias,
            "drift_model": asdict(drift_model),
        },
        seed=seed,
        columns=describe_columns(CAPACITY_COLUMNS, frame),
    )
    return Dataset(frame=frame, metadata=metadata)


def generate_demand_stream(graph: nx.Graph,
                           duration_s: float = 1800.0,
                           profile: TrafficProfile | None = None,
                           seed: int = 0) -> Dataset:
    """Arrivals of classical flows, QKD sessions and entanglement requests.

    Arrival times are Poisson with a diurnal rate envelope, sampled by
    thinning, so the process is genuinely non-homogeneous rather than a
    piecewise-constant approximation of one.

    Args:
        graph: topology; source and destination are drawn from its nodes.
        duration_s: window to generate over.
        profile: arrival statistics; defaults to :class:`TrafficProfile`.
        seed: RNG seed.

    Returns:
        Dataset: one row per demand, sorted by arrival time.

    Raises:
        ValueError: if the graph has fewer than two nodes.
    """
    if graph.number_of_nodes() < 2:
        raise ValueError("a demand needs a source and a distinct destination")

    profile = profile or TrafficProfile()
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())

    records = []
    demand_id = 0

    services = (
        ("classical", profile.classical_arrivals_per_s),
        ("qkd", profile.qkd_arrivals_per_s),
        ("entanglement", profile.entanglement_arrivals_per_s),
    )

    for service, base_rate in services:
        peak_rate = base_rate * (1.0 + profile.diurnal_amplitude)
        arrivals = _thinned_poisson(peak_rate, base_rate, profile.diurnal_amplitude,
                                    duration_s, rng)

        for arrival in arrivals:
            source, destination = rng.choice(len(nodes), size=2, replace=False)
            source, destination = nodes[source], nodes[destination]

            if service == "classical":
                volume = rng.exponential(profile.classical_volume_mean_bits)
                min_rate = 1.0e7
                min_fidelity = np.nan
                priority = int(rng.choice([1, 2], p=[0.75, 0.25]))
            elif service == "qkd":
                volume = rng.exponential(profile.qkd_volume_mean_bits)
                min_rate = 1.0e3
                min_fidelity = np.nan
                # QKD is the service that cannot degrade gracefully, so it
                # skews high priority.
                priority = int(rng.choice([2, 3], p=[0.4, 0.6]))
            else:
                volume = max(1.0, rng.exponential(profile.entanglement_pairs_mean))
                min_rate = 1.0
                min_fidelity = float(rng.choice(list(profile.min_fidelity_choices)))
                priority = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))

            service_time = volume / min_rate
            records.append({
                "demand_id": demand_id,
                "arrival_s": float(arrival),
                "service": service,
                "source": source,
                "destination": destination,
                "volume": float(volume),
                "min_rate_hz": float(min_rate),
                "min_fidelity": min_fidelity,
                "deadline_s": float(arrival + profile.deadline_slack * service_time),
                "priority": priority,
            })
            demand_id += 1

    frame = (pd.DataFrame(records, columns=list(DEMAND_COLUMNS))
             .sort_values("arrival_s", ignore_index=True))
    frame["demand_id"] = np.arange(len(frame))

    metadata = DatasetMetadata(
        name="hybrid_demands",
        description=("One row per arriving demand across the classical, QKD and "
                     "entanglement services, with non-homogeneous Poisson arrivals."),
        parameters={
            "graph": graph.graph.get("name", "unnamed"),
            "duration_s": duration_s,
            "profile": profile.as_dict(),
        },
        seed=seed,
        columns=describe_columns(DEMAND_COLUMNS, frame),
    )
    return Dataset(frame=frame, metadata=metadata)


def _thinned_poisson(peak_rate: float, base_rate: float, amplitude: float,
                     duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Non-homogeneous Poisson arrivals by thinning a homogeneous process.

    Generate at the envelope's peak rate, then keep each point with
    probability ``rate(t) / peak_rate``.  Exact, unlike binning time and
    treating the rate as constant within each bin.
    """
    if peak_rate <= 0 or duration_s <= 0:
        return np.empty(0)

    count = rng.poisson(peak_rate * duration_s)
    if count == 0:
        return np.empty(0)

    candidates = np.sort(rng.uniform(0.0, duration_s, size=count))
    rate = base_rate * (1.0 + amplitude * np.sin(2 * np.pi * candidates / 86400.0))
    keep = rng.random(count) < np.clip(rate / peak_rate, 0.0, 1.0)
    return candidates[keep]
