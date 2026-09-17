"""The hybrid allocation problem: one dial per link, and what it costs.

The decision
------------
Every control step the allocator sets, for each link, how much classical
traffic to admit. That is the whole action space, and it is small on purpose.
Classical admission is the one control that couples the two services: the
launched power scatters into the quantum channel
(:mod:`qnet_autopilot.physics.coexistence`), raising its QBER, and BB84's
secret key rate falls off a cliff at 11%. Throughput on one service is bought
directly out of the other's error budget.

What is held fixed, and why
---------------------------
Routing is shortest-path, resolved once per demand. Scheduling *within* the
quantum service is earliest-deadline-first, and so is scheduling within the
classical service. Both are the same for every policy compared here.

That is deliberate experimental design rather than laziness. Phase 2 already
answers the routing question, and a policy allowed to route *and* throttle
could win either way, leaving the result uninterpretable. Holding everything
but the throttle fixed means a difference between two policies is a difference
in one decision.

What the allocator can see
--------------------------
Not the true QBER. It sees an estimate from the sifted bits sacrificed for
parameter estimation that step — a binomial sample, precise on a fast link and
noisy on a long one. And that estimate is *contaminated by its own action*:
it includes the Raman noise the allocator itself just caused by admitting
classical traffic. A policy that wants to forecast where the fibre is drifting
has to subtract the interference it knows it created, which is a small piece of
causal bookkeeping that is easy to get wrong and produces a forecaster that
chases its own control signal if you do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics import constants as C
from ..physics.bell_diagonal import fidelity, swap, werner
from ..physics.coexistence import (CoexistenceParameters, coexistence_qber,
                                   raman_noise_counts)
from ..compensation.residual import COMPENSATED, DEGRADED, ResidualDriftModel
from ..physics.fiber import transmittance
from ..physics.link import (bb84_finite_key_rate, bb84_secret_key_rate,
                            entanglement_rate_hz)
from .state import AllocationState, Demand, DemandProgress, LinkProfile

__all__ = ["AllocationConfig", "HybridAllocationEnv", "Observation", "StepResult",
           "visibility_fidelity"]


def visibility_fidelity(qber: ArrayLike) -> NDArray[np.float64]:
    """Werner fidelity of a polarisation-entangled pair at a given QBER.

    A polarisation-encoded link's QBER and its two-photon visibility are the
    same measurement read two ways: :math:`V = 1 - 2e`. Depolarising noise
    takes the pair to a Werner state, whose fidelity is :math:`(3V + 1)/4`.

    So the same background that kills BB84 key degrades entangled pairs, on the
    same schedule. That is not a modelling convenience — it is why one control
    dial serves both services, and why an allocator that protects the QBER
    budget protects both at once.

    Args:
        qber: error rate in ``[0, 0.5]``.

    Returns:
        Fidelity in ``[0.25, 1]``. A fully mixed pair is 0.25, not 0.
    """
    visibility = np.clip(1.0 - 2.0 * np.asarray(qber, dtype=float), 0.0, 1.0)
    return (3.0 * visibility + 1.0) / 4.0


@dataclass(frozen=True)
class AllocationConfig:
    """Timing, hardware and traffic for one allocation episode.

    Attributes:
        control_period_s: how often admission is re-decided. Seconds, not
            milliseconds — this is a traffic controller, not the polarisation
            loop of phase 3.
        reconfiguration_latency_s: delay between deciding an admission level and
            that level taking effect. Changing how much classical traffic a link
            carries is not instantaneous: flows have to be re-routed, transponder
            power adjusted, and the change signalled across a control plane.
            Must be a whole number of control periods.

            This is the parameter that decides whether forecasting is worth
            anything. At zero delay a controller can simply react to what it
            measures, and reacting to a slowly mean-reverting process is nearly
            as good as predicting it; the further ahead the decision has to be
            committed, the more the forecast is worth. Phase 3 found the same
            structure in the polarisation loop, and for the same reason. It is
            swept rather than asserted.
        duration_s: episode length.
        sifted_rate_hz: sifted rate at zero loss, scaled per link by the span's
            transmittance.
        mean_photon_number: weak-coherent-pulse intensity, for the signal
            counts the Raman model needs.
        classical_capacity_bps: classical throughput per link at full admission.
        classical_power_full_w: classical launch power at full admission. The
            exchange rate between the two services; 1 mW (0 dBm) is what a
            normal telecom channel runs at.
        estimation_fraction: share of sifted bits spent estimating the QBER
            rather than becoming key. Real protocols sacrifice a few percent.
        qber_threshold: where BB84 stops producing key.
        admission_levels: the discrete admission fractions a policy may choose.
            Continuous control is not obviously better here and discrete levels
            make every policy directly comparable.
        compensated: residual drift of a link whose polarisation loop is
            working — this environment sits **on top of** phase 3, so what it
            sees is what that loop leaves, not a bare fibre.
        degraded: residual drift of a link whose loop is struggling.
        degraded_fraction: share of links drawn as degraded each episode.
            Drawn per episode rather than fixed to particular links on purpose:
            tying poor compensation to the longest spans would make span length
            and drift quality collinear, and no result could then say which of
            the two a policy was reacting to.
        coexistence: Raman and detector parameters.
        finite_key: charge the finite-block cost of key extraction instead of
            quoting the asymptotic rate. A control period's worth of sifted bits
            on a 145 km span is a few thousand, which is squarely in the regime
            where the two differ by more than a rounding error — and where a
            short block can be worth nothing at a QBER the asymptotic formula
            is happy with. Off by default so the published phase-4 numbers stay
            reproducible; the report gives both.
        seed_stride: spacing between per-link drift seeds, so links in
            different ducts genuinely do not share a thermal environment.
    """

    control_period_s: float = 5.0
    reconfiguration_latency_s: float = 30.0
    duration_s: float = 1800.0
    sifted_rate_hz: float = 1.0e6
    mean_photon_number: float = 0.5
    classical_capacity_bps: float = 1.0e11
    classical_power_full_w: float = 1.0e-3
    estimation_fraction: float = 0.05
    qber_threshold: float = C.BB84_QBER_THRESHOLD
    admission_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    compensated: ResidualDriftModel = COMPENSATED
    degraded: ResidualDriftModel = DEGRADED
    degraded_fraction: float = 0.35
    coexistence: CoexistenceParameters | None = None
    finite_key: bool = False
    seed_stride: int = 1000

    def __post_init__(self) -> None:
        if self.control_period_s <= 0:
            raise ValueError("control_period_s must be positive")
        if self.duration_s <= self.control_period_s:
            raise ValueError("duration_s must cover at least one control period")
        if not 0 < self.estimation_fraction < 1:
            raise ValueError("estimation_fraction must be in (0, 1)")
        if not self.admission_levels:
            raise ValueError("admission_levels cannot be empty")
        if min(self.admission_levels) < 0 or max(self.admission_levels) > 1:
            raise ValueError("admission_levels must lie in [0, 1]")
        if not 0.0 <= self.degraded_fraction <= 1.0:
            raise ValueError("degraded_fraction must lie in [0, 1]")
        if self.reconfiguration_latency_s < 0:
            raise ValueError("reconfiguration_latency_s cannot be negative")
        periods = self.reconfiguration_latency_s / self.control_period_s
        if abs(periods - round(periods)) > 1e-9:
            raise ValueError(
                f"reconfiguration_latency_s ({self.reconfiguration_latency_s:g} s) "
                f"must be a whole number of control periods "
                f"({self.control_period_s:g} s); got {periods:.3f}")

    @property
    def steps(self) -> int:
        return int(round(self.duration_s / self.control_period_s))

    @property
    def latency_steps(self) -> int:
        return int(round(self.reconfiguration_latency_s / self.control_period_s))


@dataclass
class Observation:
    """What a policy is handed each step.

    Attributes:
        time_s: now.
        measured_qber: per-link QBER estimate from this step's sifted sample —
            includes the Raman the policy itself caused.
        admission: the admission fractions currently **in force**, which are the
            ones decided ``reconfiguration_latency_s`` ago, not the ones the
            policy asked for last step.
        pending_admission: admissions already decided and waiting to take
            effect, oldest first, shape ``(latency_steps, links)``. A policy
            that ignores these is bidding against its own queued commitments.
        raman_qber_estimate: the policy's own interference, as the environment
            would compute it from the admission it applied. Given explicitly
            because a deployed controller knows its own transmit power; it is
            not privileged information about the fibre.
        key_pool_bits: banked key per link.
        key_rate_hz: secret key rate each link produced this step.
        backlog_bits: outstanding QKD volume routed over each link.
        urgency_s: time to the nearest deadline among QKD demands on each link;
            ``inf`` where there are none.
        classical_backlog_bits: outstanding classical volume over each link.
    """

    time_s: float
    measured_qber: NDArray[np.float64]
    admission: NDArray[np.float64]
    pending_admission: NDArray[np.float64]
    raman_qber_estimate: NDArray[np.float64]
    key_pool_bits: NDArray[np.float64]
    key_rate_hz: NDArray[np.float64]
    backlog_bits: NDArray[np.float64]
    urgency_s: NDArray[np.float64]
    classical_backlog_bits: NDArray[np.float64]


@dataclass
class StepResult:
    """One step's outcome.

    Attributes:
        observation: what the policy sees next.
        key_bits: secret key produced per link this step.
        classical_bits: classical bits carried per link this step.
        qkd_delivered_bits, classical_delivered_bits, pairs_delivered: volume
            actually handed to demands, which is less than what was produced
            when there is nothing to serve.
        expired: demand ids that missed their deadline this step.
        completed: demand ids fully served this step.
        done: whether the episode is over.
    """

    observation: Observation
    key_bits: NDArray[np.float64]
    classical_bits: NDArray[np.float64]
    qkd_delivered_bits: float
    classical_delivered_bits: float
    pairs_delivered: float
    expired: list[int]
    completed: list[int]
    done: bool


class HybridAllocationEnv:
    """Classical and quantum traffic on one plant, stepped in control periods.

    Args:
        graph: topology with ``length`` on every edge.
        demands: the arriving stream. Built by
            :func:`~.scenario.build_demands` from the phase-1 generator so the
            traffic is the same one phase 1 validated.
        config: timing, hardware and traffic parameters.

    Note:
        The environment is deterministic given its seed, including the
        measurement noise, so two policies run at the same seed face exactly
        the same fibre and the same arrivals. Every comparison in phase 4 is
        paired on that basis.
    """

    def __init__(self, graph: nx.Graph, demands: Sequence[Demand],
                 config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()
        self.graph = graph
        self.demands = list(demands)
        self.coexistence = self.config.coexistence or CoexistenceParameters()

        self.links: list[LinkProfile] = []
        for index, (u, v, data) in enumerate(graph.edges(data=True)):
            length_km = float(data["length"])
            span = float(transmittance(length_km))
            self.links.append(LinkProfile(
                index=index,
                name=f"{u}|{v}",
                source=str(u),
                target=str(v),
                length_km=length_km,
                sifted_rate_hz=self.config.sifted_rate_hz * span,
                signal_counts_per_gate=(self.config.mean_photon_number * span
                                        * self.coexistence.detector_efficiency),
                classical_capacity_bps=self.config.classical_capacity_bps,
                classical_power_full_w=self.config.classical_power_full_w,
                entanglement_rate_hz=float(entanglement_rate_hz(length_km)),
            ))

        self.num_links = len(self.links)
        self._length_km = np.array([link.length_km for link in self.links])
        self._signal = np.array([link.signal_counts_per_gate for link in self.links])
        self._sifted = np.array([link.sifted_rate_hz for link in self.links])
        self._classical_capacity = np.array(
            [link.classical_capacity_bps for link in self.links])
        self._full_power = np.array(
            [link.classical_power_full_w for link in self.links])
        self._pair_rate = np.array([link.entanglement_rate_hz for link in self.links])

        self.state = AllocationState()
        self._pending: NDArray[np.float64] = np.zeros(
            (self.config.latency_steps, self.num_links))
        self._polarisation: NDArray[np.float64] = np.zeros((0, 0))
        self._rng = np.random.default_rng(0)
        self._step_index = 0

    # -- setup --------------------------------------------------------------

    def reset(self, seed: int = 0) -> Observation:
        """Draw a fresh fibre, empty the pools, and return the first view.

        Each link gets its own drift realisation from a derived seed: links in
        different ducts do not share a thermal environment, and a trace that
        made them move together would let a policy predict one link from
        another for free.

        Which links are degraded is also drawn here, from the episode seed, so
        the same seed always produces the same plant and two policies compared
        at that seed face exactly the same fibre.
        """
        config = self.config
        self._rng = np.random.default_rng(seed)
        self._step_index = 0

        chooser = np.random.default_rng(seed + 7_000_003)
        self.degraded_links = (chooser.random(self.num_links)
                               < config.degraded_fraction)

        steps = config.steps + 1
        polarisation = np.empty((steps, self.num_links))
        for index in range(self.num_links):
            model = (config.degraded if self.degraded_links[index]
                     else config.compensated)
            trace = model.simulate(config.duration_s + config.control_period_s,
                                   config.control_period_s,
                                   rng=seed + config.seed_stride * (index + 1))
            polarisation[:, index] = trace[:steps]
        self._polarisation = polarisation

        self.state = AllocationState(
            time_s=0.0,
            key_pool_bits=np.zeros(self.num_links),
            qber=np.zeros(self.num_links),
            measured_qber=np.zeros(self.num_links),
            admission=np.zeros(self.num_links),
            progress={demand.demand_id: DemandProgress() for demand in self.demands},
        )

        # Nothing has been decided yet, so the plant starts silent and the
        # queue of pending reconfigurations is empty rather than optimistic.
        self._pending = np.zeros((config.latency_steps, self.num_links))

        # A first look at an idle plant, so a policy's opening move is informed
        # by a measurement rather than by nothing.
        self.state.qber = self._true_qber(np.zeros(self.num_links), 0)
        self.state.measured_qber = self._measure(self.state.qber)
        return self._observe(np.zeros(self.num_links))

    # -- physics ------------------------------------------------------------

    def _true_qber(self, admission: NDArray[np.float64], step: int
                   ) -> NDArray[np.float64]:
        """Alignment error from drift plus the background the admission caused."""
        polarisation = self._polarisation[min(step, len(self._polarisation) - 1)]
        return coexistence_qber(self._signal, admission * self._full_power,
                                self._length_km, polarisation, self.coexistence)

    def raman_qber(self, admission: ArrayLike) -> NDArray[np.float64]:
        """The QBER a given admission would add on a perfectly aligned fibre.

        This is what a policy is allowed to subtract from its measurement to
        recover the drift underneath: a controller knows its own transmit power
        and its own fibre's Raman coefficient. It does **not** know the drift,
        which is the thing being forecast.
        """
        admission = np.asarray(admission, dtype=float)
        clean = coexistence_qber(self._signal, admission * self._full_power,
                                 self._length_km, 0.0, self.coexistence)
        floor = coexistence_qber(self._signal, 0.0, self._length_km, 0.0,
                                 self.coexistence)
        return clean - floor

    def _key_bits(self, qber: NDArray[np.float64], dt: float
                  ) -> NDArray[np.float64]:
        """Secret bits each link produces this control period.

        With ``finite_key`` the block is one control period of sifted bits and
        the finite-size costs are charged against it; otherwise this is the
        asymptotic rate, which is what phases 3 and 4 originally reported.
        """
        payload = self._sifted * (1.0 - self.config.estimation_fraction)
        if self.config.finite_key:
            return np.asarray(bb84_finite_key_rate(payload, qber, dt),
                              dtype=float) * dt
        return np.asarray(bb84_secret_key_rate(payload, qber), dtype=float) * dt

    def _measure(self, qber: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate the QBER from the sifted bits sacrificed to measure it.

        Binomial, with the sample size the link's own rate gives it. A fast
        short link measures itself precisely; a long one does not, which is
        exactly the link where the decision is hardest.
        """
        sample = np.maximum(
            (self._sifted * self.config.control_period_s
             * self.config.estimation_fraction).astype(np.int64), 1)
        return self._rng.binomial(sample, np.clip(qber, 0.0, 1.0)) / sample

    # -- stepping -----------------------------------------------------------

    def step(self, admission: ArrayLike) -> StepResult:
        """Decide an admission vector, and run the one that is due.

        The vector passed in is a *decision*, not an action: it joins the queue
        and takes effect ``reconfiguration_latency_s`` later. What runs this
        period is whatever was decided that long ago. With zero latency the two
        coincide and this reduces to the obvious thing.

        Args:
            admission: classical admission fraction per link, in ``[0, 1]``.

        Returns:
            StepResult: production, delivery, and the next observation.

        Raises:
            ValueError: on the wrong shape or an out-of-range fraction.
        """
        decision = np.asarray(admission, dtype=float)
        if decision.shape != (self.num_links,):
            raise ValueError(
                f"admission must have shape ({self.num_links},), got {decision.shape}")
        if np.any(decision < 0.0) or np.any(decision > 1.0):
            raise ValueError("admission fractions must lie in [0, 1]")

        if len(self._pending):
            admission = self._pending[0].copy()
            self._pending = np.vstack([self._pending[1:], decision[None, :]])
        else:
            admission = decision

        config = self.config
        dt = config.control_period_s
        state = self.state

        qber = self._true_qber(admission, self._step_index)
        key_bits = self._key_bits(qber, dt)
        state.key_pool_bits = state.key_pool_bits + key_bits

        classical_available = admission * self._classical_capacity * dt
        pair_fidelity = visibility_fidelity(qber)

        now = state.time_s
        deadline_end = now + dt
        completed: list[int] = []
        qkd_delivered = 0.0
        classical_delivered = 0.0
        pairs_delivered = 0.0

        active = state.active_demands(self.demands, deadline_end)
        quantum = sorted((demand for demand in active if demand.service != "classical"),
                         key=lambda demand: demand.deadline_s)
        classical = sorted((demand for demand in active if demand.service == "classical"),
                           key=lambda demand: demand.deadline_s)

        # -- quantum services draw on the pools, earliest deadline first ----
        pair_budget = self._pair_rate * dt
        for demand in quantum:
            progress = state.progress[demand.demand_id]
            remaining = progress.remaining(demand.volume)
            if remaining <= 0:
                continue
            links = np.array(demand.path, dtype=int)

            if demand.service == "qkd":
                # Trusted-node relay: one bit of end-to-end key costs one bit
                # of link key on every hop, so the path is limited by its
                # poorest pool.
                served = float(min(remaining, state.key_pool_bits[links].min()))
                if served <= 0:
                    continue
                state.key_pool_bits[links] -= served
                qkd_delivered += served
            else:
                # Entanglement: swapping composes the hops, and the result has
                # to clear the request's own fidelity threshold. Below it the
                # pairs are not worth distributing at all.
                end_to_end = _path_fidelity(pair_fidelity[links])
                if end_to_end < demand.min_fidelity:
                    continue
                served = float(min(remaining, pair_budget[links].min()))
                if served <= 0:
                    continue
                pair_budget[links] -= served
                pairs_delivered += served

            progress.delivered += served
            if np.isnan(progress.started_s):
                progress.started_s = now
            if progress.remaining(demand.volume) <= 1e-9:
                progress.completed_s = deadline_end
                completed.append(demand.demand_id)

        # -- classical flows take the admitted capacity, same discipline ----
        for demand in classical:
            progress = state.progress[demand.demand_id]
            remaining = progress.remaining(demand.volume)
            if remaining <= 0:
                continue
            links = np.array(demand.path, dtype=int)
            served = float(min(remaining, classical_available[links].min()))
            if served <= 0:
                continue
            classical_available[links] -= served
            progress.delivered += served
            classical_delivered += served
            if np.isnan(progress.started_s):
                progress.started_s = now
            if progress.remaining(demand.volume) <= 1e-9:
                progress.completed_s = deadline_end
                completed.append(demand.demand_id)

        # -- deadlines ------------------------------------------------------
        expired: list[int] = []
        for demand in self.demands:
            progress = state.progress[demand.demand_id]
            if progress.active and demand.deadline_s <= deadline_end:
                progress.expired = True
                expired.append(demand.demand_id)

        classical_bits = admission * self._classical_capacity * dt - classical_available

        state.time_s = deadline_end
        state.qber = qber
        state.admission = admission
        self._step_index += 1
        state.measured_qber = self._measure(self._true_qber(admission,
                                                            self._step_index))

        return StepResult(
            observation=self._observe(admission),
            key_bits=key_bits,
            classical_bits=classical_bits,
            qkd_delivered_bits=qkd_delivered,
            classical_delivered_bits=classical_delivered,
            pairs_delivered=pairs_delivered,
            expired=expired,
            completed=completed,
            done=self._step_index >= config.steps,
        )

    # -- observation --------------------------------------------------------

    def _observe(self, admission: NDArray[np.float64]) -> Observation:
        state = self.state
        backlog = np.zeros(self.num_links)
        classical_backlog = np.zeros(self.num_links)
        urgency = np.full(self.num_links, np.inf)

        for demand in state.active_demands(self.demands, state.time_s):
            progress = state.progress[demand.demand_id]
            remaining = progress.remaining(demand.volume)
            links = list(demand.path)
            if demand.service == "qkd":
                backlog[links] += remaining
                slack = demand.deadline_s - state.time_s
                urgency[links] = np.minimum(urgency[links], slack)
            elif demand.service == "classical":
                classical_backlog[links] += remaining

        key_rate = self._key_bits(state.qber, 1.0)

        return Observation(
            time_s=state.time_s,
            measured_qber=state.measured_qber.copy(),
            admission=admission.copy(),
            pending_admission=self._pending.copy(),
            raman_qber_estimate=self.raman_qber(admission),
            key_pool_bits=state.key_pool_bits.copy(),
            key_rate_hz=np.asarray(key_rate, dtype=float),
            backlog_bits=backlog,
            urgency_s=urgency,
            classical_backlog_bits=classical_backlog,
        )


def _path_fidelity(link_fidelities: NDArray[np.float64]) -> float:
    """Compose hop fidelities by entanglement swapping.

    Uses the phase-1 Bell-diagonal ``swap``, which was checked against
    SeQUeNCe to machine precision, rather than the product-of-fidelities
    approximation — the two differ by several percent at the fidelities where
    a request's threshold actually bites.
    """
    if len(link_fidelities) == 0:
        return 0.0
    state = werner(float(link_fidelities[0]))
    for value in link_fidelities[1:]:
        state = swap(state, werner(float(value)))
    return float(fidelity(state))
