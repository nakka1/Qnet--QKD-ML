"""Who decides how much classical traffic each link carries.

The comparison this module is built for
---------------------------------------
Every policy here answers one question per link per control period: what
fraction of the classical capacity to admit. They differ only in what they look
at, which is what makes the differences between them interpretable.

* :class:`ClassicalFirst` looks at nothing. It is the status quo of a network
  whose transport layer does not know a quantum channel is there.
* :class:`QuantumFirst` looks at the demand book. Maximally protective, and it
  is the other trivial answer.
* :class:`StaticAdmission` looks at nothing but has a tuned constant, which is
  what a network actually ships when someone measures once and writes the
  number into a config file.
* :class:`ReactiveThreshold` looks at the current measured QBER. This is the
  strong baseline: a closed-loop controller, tuned by grid search, doing the
  obvious correct thing. Everything a predictive policy claims has to be
  claimed against this.
* :class:`PredictiveAllocator` looks at a *forecast* of where the drift is
  going, and at whether the key it can bank between now and each deadline will
  be enough.

The last two share their entire control law
-------------------------------------------
:class:`PredictiveAllocator` with a :class:`~.forecast.PersistenceForecaster` is
not a different controller from one with a fitted forecaster — it is the same
arithmetic reading a different forecast. So the difference between them is the
value of the forecast and nothing else: not a better objective, not a better
tuned constant, not more information about the demand book. That is the only
way to make "prediction helps" a measurement rather than an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..physics.coexistence import CoexistenceParameters, coexistence_qber
from ..physics.link import bb84_secret_key_rate
from .env import HybridAllocationEnv, Observation
from .forecast import Forecaster, PersistenceForecaster, drift_estimate

__all__ = ["AllocationPolicy", "ClassicalFirst", "QuantumFirst", "StaticAdmission",
           "ReactiveThreshold", "PredictiveAllocator"]


@runtime_checkable
class AllocationPolicy(Protocol):
    """Turns an observation into an admission fraction per link."""

    name: str

    def reset(self, env: HybridAllocationEnv) -> None:
        """Called once at the start of an episode, after ``env.reset``."""

    def act(self, observation: Observation, step: int) -> NDArray[np.float64]:
        """Admission fractions in ``[0, 1]``, shape ``(links,)``."""


@dataclass
class ClassicalFirst:
    """Admit everything, always. The network that does not know QKD is there.

    Not a straw man — it is what a transport layer does by default, and the
    reason coexistence needs managing at all. Its classical throughput is the
    ceiling every other policy is measured against.
    """

    name: str = "clássico primeiro"

    def reset(self, env: HybridAllocationEnv) -> None:
        self._links = env.num_links

    def act(self, observation: Observation, step: int) -> NDArray[np.float64]:
        return np.ones(self._links)


@dataclass
class QuantumFirst:
    """Silence the classical channel on any link with quantum work outstanding.

    The other trivial answer, and the one a security team asks for. It protects
    the key perfectly and it is expensive: a link with one small QKD session
    queued carries no classical traffic at all, for as long as the session
    lasts.
    """

    name: str = "quântico primeiro"

    def reset(self, env: HybridAllocationEnv) -> None:
        self._links = env.num_links

    def act(self, observation: Observation, step: int) -> NDArray[np.float64]:
        busy = (observation.backlog_bits > 0) | np.isfinite(observation.urgency_s)
        return np.where(busy, 0.0, 1.0)


@dataclass
class StaticAdmission:
    """One constant everywhere, chosen once by measurement.

    What a real network ships. Its weakness is precisely that it cannot be
    right twice: the constant that protects the 145 km span wastes the 30 km
    one, and the constant that suits a quiet hour fails a drifting one.
    """

    level: float = 0.5
    name: str = "admissão estática"

    def __post_init__(self) -> None:
        if not 0.0 <= self.level <= 1.0:
            raise ValueError("level must lie in [0, 1]")
        self.name = f"admissão estática ({self.level:.2f})".replace(".", ",")

    def reset(self, env: HybridAllocationEnv) -> None:
        self._links = env.num_links

    def act(self, observation: Observation, step: int) -> NDArray[np.float64]:
        return np.full(self._links, self.level)


@dataclass
class ReactiveThreshold:
    """Throttle when the measured QBER is high, open up when it is low.

    A hysteretic closed-loop controller: step the admission down while the link
    is above ``high``, step it up while it is below ``low``. The dead band
    between them stops it oscillating on measurement noise, which a single
    threshold does on the long spans where the estimate is worst.

    This is the baseline that matters. It is what a control engineer builds, it
    needs no model of the fibre, and on a slowly drifting link it is hard to
    beat — the forecast has to earn its margin on the transitions.

    Attributes:
        low, high: the dead band, in QBER.
        step: how far admission moves per control period.
        only_when_needed: open fully on links with no quantum work outstanding.
            On by default because leaving an idle link throttled is a pure loss
            that no baseline should be charged with.
    """

    low: float = 0.05
    high: float = 0.08
    step: float = 0.25
    only_when_needed: bool = True
    name: str = "limiar reativo"

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high:
            raise ValueError("need 0 <= low <= high")
        self.name = (f"limiar reativo ({self.low:.2f}–{self.high:.2f})"
                     .replace(".", ","))

    def reset(self, env: HybridAllocationEnv) -> None:
        self._admission = np.ones(env.num_links)

    def act(self, observation: Observation, step: int) -> NDArray[np.float64]:
        qber = observation.measured_qber
        self._admission = np.where(qber > self.high, self._admission - self.step,
                                   np.where(qber < self.low,
                                            self._admission + self.step,
                                            self._admission))
        self._admission = np.clip(self._admission, 0.0, 1.0)

        if not self.only_when_needed:
            return self._admission.copy()
        busy = (observation.backlog_bits > 0) | np.isfinite(observation.urgency_s)
        return np.where(busy, self._admission, 1.0)


@dataclass
class PredictiveAllocator:
    """Admit as much classical traffic as the forecast says the key can spare.

    The control law, per link, per step:

    1. Strip the controller's own Raman contribution from the measurement to
       get the drift underneath (:func:`~.forecast.drift_estimate`).
    2. Ask the forecaster where that drift will be over the next ``horizon``
       control periods.
    3. Work out what the link owes: the outstanding QKD volume routed over it,
       less the key already banked, and how long until the earliest of those
       deadlines. Convert that into a **pacing target** — the share of the debt
       that falls inside the planning window, pro rata. A session due in ten
       minutes does not have to be finished in the next minute, and asking for
       that collapses the decision to all-or-nothing.
    4. Integrate the key the link will produce over that window. The first part
       of the window is already committed — those admissions are in the
       reconfiguration queue and cannot be changed — so only the part after the
       latency responds to this decision. Take the most generous admission level
       whose total still covers the pacing target.

    Step 4 is the whole point, and it is why the reconfiguration latency
    matters. A decision made now lands a reconfiguration period from now and
    stays in force after that, so the drift it has to survive is the drift
    *then*, not the drift being measured. A controller that reacts to the
    present is answering a question about the past; with a slowly mean-reverting
    process and a short latency the two answers coincide, and as the latency
    grows they come apart. Whether they come apart enough to matter is measured,
    not assumed.

    Attributes:
        forecaster: supplies the drift forecast. Swap it to change *only* the
            forecast — the control law above is fixed, which is what makes the
            comparison between forecasters a measurement of the forecast.
        coverage_target: how much of the debt the forecast must promise before
            classical traffic is admitted. Above 1 the allocator insists on a
            margin over what it owes; below 1 it knowingly under-serves the key
            to buy classical throughput. This is the operating point, not a
            safety factor, and sweeping it is what traces the policy's frontier.
        idle_level: admission on links with nothing quantum outstanding.
        horizon_cap_s: how far ahead to plan. Deadlines beyond it are still
            honoured, through the pro-rata pacing target — the horizon bounds
            how far the forecast is trusted, not which demands count.

    Raises:
        ValueError: at ``reset``, if the forecaster does not see past the
            reconfiguration latency. A decision only affects the window *after*
            it lands, so a forecast that stops at or before the latency leaves
            nothing for the decision to act on: every admission level scores
            identically, the controller silently stops using the forecast, and
            the symptom is that a latency sweep shows the forecast mattering
            *less* the longer the commitment — the opposite of the truth.
    """

    forecaster: Forecaster = field(default_factory=PersistenceForecaster)
    coverage_target: float = 1.0
    idle_level: float = 1.0
    horizon_cap_s: float = 150.0
    name: str = "alocador preditivo"

    def __post_init__(self) -> None:
        if self.coverage_target <= 0:
            raise ValueError("coverage_target must be positive")
        self.name = f"preditivo ({self.forecaster.name})"

    def reset(self, env: HybridAllocationEnv) -> None:
        self._env = env
        self._config = env.config
        self._levels = np.asarray(sorted(env.config.admission_levels, reverse=True))
        self._optical = (env.config.coexistence or CoexistenceParameters()
                         ).optical_error_rate
        self._payload = 1.0 - env.config.estimation_fraction
        self._latency = env.config.latency_steps
        if self._latency >= self.forecaster.horizon:
            raise ValueError(
                f"forecast horizon ({self.forecaster.horizon} control periods) "
                f"must exceed the reconfiguration latency "
                f"({self._latency} periods): a decision that lands after the "
                f"forecast ends cannot be evaluated against it")
        self.forecaster.reset(env)

    def act(self, observation: Observation, step: int) -> NDArray[np.float64]:
        env = self._env
        dt = self._config.control_period_s

        drift = drift_estimate(observation.measured_qber,
                               observation.raman_qber_estimate, self._optical)
        forecast = self.forecaster.predict(drift, step)          # (links, horizon)

        # How much of the forecast is relevant to each link: out to its nearest
        # QKD deadline, capped so a distant one does not make everything look
        # easy, and never past what the forecaster actually produced.
        window = np.minimum(observation.urgency_s, self.horizon_cap_s)
        periods = np.clip(np.ceil(window / dt).astype(int), 1, forecast.shape[1])

        debt = observation.backlog_bits - observation.key_pool_bits
        pending = observation.pending_admission
        admission = np.full(env.num_links, self.idle_level)

        # Pro rata: only the share of the debt that falls inside the planning
        # window has to be produced inside it. Without this the controller
        # compares a whole session's backlog against one horizon of key, which
        # every link fails, and the decision degenerates to all-or-nothing.
        pace = np.where(np.isfinite(observation.urgency_s),
                        np.minimum(window / np.maximum(observation.urgency_s, dt), 1.0),
                        1.0)

        for link in range(env.num_links):
            if debt[link] <= 0:
                continue

            horizon = forecast[link, :periods[link]]
            committed = min(self._latency, len(horizon))

            # Already in the queue: this decision cannot touch it, so it is a
            # constant in the comparison rather than something to optimise.
            locked = self._key_over(link, horizon[:committed],
                                    pending[:committed, link]) * dt
            free = horizon[committed:]

            for level in self._levels:
                promised = locked + self._key_over(link, free, level) * dt
                if promised >= self.coverage_target * pace[link] * debt[link]:
                    admission[link] = level
                    break
            else:
                admission[link] = 0.0

        return admission

    def _key_over(self, link: int, drift_forecast: NDArray[np.float64],
                  level) -> float:
        """Secret key rate summed over a forecast window at given admission."""
        if len(drift_forecast) == 0:
            return 0.0
        profile = self._env.links[link]
        power = np.asarray(level, dtype=float) * profile.classical_power_full_w
        qber = coexistence_qber(profile.signal_counts_per_gate, power,
                                profile.length_km, drift_forecast,
                                self._env.coexistence)
        return float(np.sum(bb84_secret_key_rate(
            profile.sifted_rate_hz * self._payload, qber)))
