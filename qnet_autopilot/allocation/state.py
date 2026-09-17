"""The things a hybrid allocator holds: links, demands, and banked key.

Two modelling choices here carry most of the weight.

**Key is a stock, not a flow.** A QKD link does not deliver key to a session on
demand; it fills a key pool continuously, and sessions draw from that pool.
This is how deployed QKD actually works, and it is what makes prediction
valuable at all: if the pool were not there, a link that is good now and bad in
two minutes would be worth exactly as much as one that is bad now and good in
two minutes. With a pool, the good window can be banked *before* it closes.

**Routing is fixed, on purpose.** Every demand takes the shortest path by fibre
length, computed once. Choosing paths is phase 2's problem and it is already
answered; leaving it free here would mean a policy could win by routing better
rather than by scheduling better, and the comparison would say nothing about
either. The one decision under test is how much classical traffic each link
carries, which is the decision that couples the two services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["LinkProfile", "Demand", "DemandProgress", "AllocationState",
           "SERVICES"]

#: The three services that share the plant, in the order metrics report them.
SERVICES = ("qkd", "entanglement", "classical")


@dataclass(frozen=True)
class LinkProfile:
    """Everything about a link that does not change during an episode.

    Attributes:
        index: position in the environment's link arrays.
        name: ``"u|v"``, matching the phase-1 capacity trace.
        source, target: endpoint node names.
        length_km: fibre length.
        sifted_rate_hz: sifted bits per second the quantum channel produces,
            before the QBER penalty. Scaled by the span's transmittance, so a
            long link is slow as well as noisy.
        signal_counts_per_gate: detected signal counts per detection gate,
            which is what the Raman model needs to know how much background it
            takes to swamp the signal.
        classical_capacity_bps: classical throughput at full admission.
        classical_power_full_w: classical launch power at full admission. The
            exchange rate between classical throughput and QBER; swept in the
            report, because it is the one number that decides how hard the
            problem is.
        entanglement_rate_hz: heralded pair rate over the span.
    """

    index: int
    name: str
    source: str
    target: str
    length_km: float
    sifted_rate_hz: float
    signal_counts_per_gate: float
    classical_capacity_bps: float
    classical_power_full_w: float
    entanglement_rate_hz: float


@dataclass(frozen=True)
class Demand:
    """One arriving request, with the path it will take already resolved.

    Attributes:
        demand_id: identifier, stable across policies so runs can be paired.
        service: ``"classical"``, ``"qkd"`` or ``"entanglement"``.
        source, destination: endpoints.
        path: link indices, in order, from the fixed shortest-path route.
        volume: bits for classical and QKD; pairs for entanglement.
        arrival_s, deadline_s: absolute times.
        priority: 1 (best effort) to 3 (must not be dropped), from the phase-1
            demand stream.
        min_fidelity: entanglement only; NaN otherwise.
    """

    demand_id: int
    service: str
    source: str
    destination: str
    path: tuple[int, ...]
    volume: float
    arrival_s: float
    deadline_s: float
    priority: int
    min_fidelity: float = float("nan")

    @property
    def hops(self) -> int:
        return len(self.path)


@dataclass
class DemandProgress:
    """How far along one demand is.

    Attributes:
        delivered: bits (or pairs) served so far.
        started_s: when it first received anything, or NaN.
        completed_s: when it was fully served, or NaN.
        expired: whether its deadline passed with volume outstanding.
    """

    delivered: float = 0.0
    started_s: float = float("nan")
    completed_s: float = float("nan")
    expired: bool = False

    def remaining(self, volume: float) -> float:
        return max(volume - self.delivered, 0.0)

    @property
    def done(self) -> bool:
        return not np.isnan(self.completed_s)

    @property
    def active(self) -> bool:
        """Still worth spending capacity on."""
        return not self.done and not self.expired


@dataclass
class AllocationState:
    """The mutable world: the clock, the key pools, and demand progress.

    Attributes:
        time_s: current time.
        key_pool_bits: banked secret key per link. The stock that makes a good
            window worth something after it closes.
        qber: true QBER per link this step — polarisation drift plus the Raman
            the allocator's own classical admission caused. **Not** observable;
            the environment holds it to score with, and hands the policy an
            estimate.
        measured_qber: what the allocator actually sees, from a finite sample
            of sifted bits.
        admission: the classical admission fraction currently in force per link.
        progress: per-demand progress, keyed by ``demand_id``.
    """

    time_s: float = 0.0
    key_pool_bits: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    qber: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    measured_qber: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    admission: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    progress: dict[int, DemandProgress] = field(default_factory=dict)

    def active_demands(self, demands: Sequence[Demand], now: float) -> list[Demand]:
        """Demands that have arrived, are not finished, and have not expired."""
        return [demand for demand in demands
                if demand.arrival_s <= now
                and self.progress[demand.demand_id].active]
