"""Keeping the detector gate on top of the pulse while the fibre changes length.

A gated receiver opens a window of a few hundred picoseconds around the moment
each pulse is expected.  The window is what keeps the dark counts out: widen it
and the noise scales with it; miss it and the photons land outside and are
simply lost.  Two things move the expected moment, and both are slow enough to
be predicted and fast enough to matter:

**The fibre changes length.**  Silica's refractive index and its physical length
both depend on temperature, and together they give about
:data:`~qnet_autopilot.physics.constants.DEFAULT_DELAY_THERMAL_COEFFICIENT_S_PER_KM_K`
— 40 ps per kilometre per kelvin.  Over 50 km of buried fibre a single kelvin is
two nanoseconds, ten times a typical gate.  A link that does not track this stops
producing key within a degree of wherever it was last calibrated.

**The clocks disagree.**  Alice's and Bob's oscillators differ in frequency, so
the offset accumulates linearly, and the difference itself wanders.

What a receiver can do about it is the same shape as the polarisation problem in
phase 3, and this module is deliberately built to the same plan:

* :class:`ClockModel` plays the role of the drift model — it says how the offset
  moves, given the temperature.
* :func:`measure_offset` plays the role of the monitoring channel — the offset is
  estimated from the arrival times of a finite number of photons, so the estimate
  is noisy, and the noise is what limits everything.
* the predictors below play the role of the compensators, and are the same four
  in scalar form: do nothing, re-calibrate periodically, track with an
  alpha-beta filter, or learn.
* :func:`run_sync_loop` plays the role of the closed loop, and scores the same
  way: not by the RMS of the offset, but by the detection efficiency and the key
  rate that survive.

The second half of the module is the part the key-rate agent needs:
:func:`timing_spread` and :func:`intersymbol_error` say how wide the pulse has
become by the time it arrives, and therefore how fast the clock can run before
neighbouring slots start bleeding into each other.  Over 50 km the dominant term
is chromatic dispersion of the source's own linewidth, not the detector — which
is why the ceiling on the clock rate moves with the span and is not a property of
the receiver alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics import constants as C

__all__ = [
    "TimingBudget", "timing_spread", "intersymbol_error", "gate_efficiency",
    "ClockModel", "TimingTrajectory", "MeasurementConfig", "measure_offset",
    "GateConfig", "SyncPredictor", "NoCorrection", "PeriodicCalibration",
    "AlphaBetaTracker", "LearnedSync", "SyncResult", "run_sync_loop",
    "build_sync_dataset", "fit_sync_predictor",
]


# ---------------------------------------------------------------------------
# how wide the pulse is when it arrives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingBudget:
    """Everything that smears a pulse in time, added in quadrature.

    Attributes:
        source_pulse_s: transmitter pulse width (RMS).
        source_linewidth_nm: spectral width of the source.  Multiplied by the
            fibre's dispersion and its length, this is usually the largest term
            on a metropolitan span and the reason the clock ceiling depends on
            reach.
        dispersion_ps_per_nm_km: chromatic dispersion of standard SMF at
            1550 nm; 17 is the datasheet figure.
        detector_jitter_s: timing jitter of the single-photon detector (RMS).
            50 ps is a routine superconducting-nanowire figure.
        electronics_jitter_s: jitter of the gating electronics (RMS).

    Note:
        Quadrature addition assumes the terms are independent and roughly
        Gaussian, which they are not exactly — dispersion produces a broadened
        but still structured pulse — and which is the usual engineering
        approximation.  It is accurate enough for a budget and wrong enough that
        a real deployment measures the arrival histogram instead.
    """

    source_pulse_s: float = 30.0e-12
    source_linewidth_nm: float = 0.1
    dispersion_ps_per_nm_km: float = 17.0
    detector_jitter_s: float = 50.0e-12
    electronics_jitter_s: float = 20.0e-12


def timing_spread(budget: TimingBudget, length_km: float) -> float:
    """RMS width of the arrival-time distribution over a span, in seconds.

    Args:
        budget: the contributing terms.
        length_km: span length — only the dispersion term depends on it.

    Returns:
        The RMS spread in seconds.
    """
    dispersion_s = (budget.dispersion_ps_per_nm_km * 1e-12
                    * budget.source_linewidth_nm * max(length_km, 0.0))
    return float(np.sqrt(budget.source_pulse_s ** 2
                         + dispersion_s ** 2
                         + budget.detector_jitter_s ** 2
                         + budget.electronics_jitter_s ** 2))


def gate_efficiency(offset_s: ArrayLike, gate_width_s: float,
                    spread_s: float) -> NDArray[np.float64]:
    """Share of photons that land inside the gate, given a timing offset.

    The arrival time is Gaussian about the expected moment with standard
    deviation ``spread_s``; the gate accepts everything within ``gate_width_s/2``
    of where the receiver *thinks* the pulse is.  With an offset the window sits
    off-centre and the accepted share is the difference of two error functions.

    Args:
        offset_s: residual timing error — true arrival minus expected.
        gate_width_s: full width of the acceptance window.
        spread_s: RMS arrival spread from :func:`timing_spread`.

    Returns:
        Efficiency in ``[0, 1]``, same shape as ``offset_s``.

    Note:
        This is a *multiplicative* loss on the channel transmittance, so its
        effect on the key rate is exactly that of extra attenuation — except
        that it does not reduce the dark counts, which keep arriving at the same
        rate whatever the gate is aligned to.  A badly synchronised link
        therefore loses key faster than the same amount of honest loss would
        cost it, and that asymmetry is what makes tracking worth doing.
    """
    from math import sqrt

    offset = np.asarray(offset_s, dtype=float)
    half = gate_width_s / 2.0
    scale = max(spread_s, 1e-18) * sqrt(2.0)
    upper = (half - offset) / scale
    lower = (-half - offset) / scale
    return 0.5 * (_erf(upper) - _erf(lower))


def intersymbol_error(clock_rate_hz: ArrayLike, spread_s: float) -> NDArray[np.float64]:
    """Error rate from photons arriving in a neighbouring pulse slot.

    At slot period :math:`T = 1/f`, a photon further than :math:`T/2` from its
    own slot centre is attributed to the next slot, where it is uncorrelated
    with the bit that slot carries and is therefore wrong half the time:

    .. math:: e_{\\text{ISI}} = \\tfrac12
              \\operatorname{erfc}\\!\\left(\\frac{T}{2\\sqrt2 \\sigma}\\right).

    Args:
        clock_rate_hz: pulse repetition rate.
        spread_s: RMS arrival spread.

    Returns:
        The added error rate.

    Note:
        This is what stops "turn the clock up" from being a free win, and it is
        the reason the ceiling on the clock rate belongs to the *span* and not to
        the receiver: at 20 km the spread is dominated by detector jitter and a
        five-gigahertz clock is harmless, while at 80 km dispersion has widened
        the pulse past the slot and the same clock destroys the key.
    """
    from math import sqrt

    period = 1.0 / np.asarray(clock_rate_hz, dtype=float)
    argument = period / (2.0 * sqrt(2.0) * max(spread_s, 1e-18))
    return 0.5 * (1.0 - _erf(argument))


def _erf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Error function, from scipy where available and Abramowitz-Stegun otherwise.

    The fallback is 7.1.26, accurate to 1.5e-7 — far below anything that matters
    here, and it keeps the module usable in the same stripped environments
    :mod:`qnet_autopilot.compat` exists for.
    """
    try:
        from scipy.special import erf

        return np.asarray(erf(values), dtype=float)
    except Exception:  # pragma: no cover - exercised only without scipy
        values = np.asarray(values, dtype=float)
        sign = np.sign(values)
        x = np.abs(values)
        t = 1.0 / (1.0 + 0.3275911 * x)
        polynomial = t * (0.254829592 + t * (-0.284496736 + t * (
            1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return sign * (1.0 - polynomial * np.exp(-x * x))


# ---------------------------------------------------------------------------
# how the offset moves
# ---------------------------------------------------------------------------

@dataclass
class TimingTrajectory:
    """One simulated history of the timing offset and what drove it.

    Attributes:
        time_s: sample times.
        offset_s: true arrival time minus the nominal one, per sample.
        temperature_c: cable temperature.
        length_km: span length, for the budget.
        components: the offset split into its causes — ``thermal`` and
            ``clock`` — kept because the two are separable in principle and a
            predictor that only tracks the sum should be shown what it is
            missing.
    """

    time_s: NDArray[np.float64]
    offset_s: NDArray[np.float64]
    temperature_c: NDArray[np.float64]
    length_km: float
    components: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.time_s.size)


@dataclass
class ClockModel:
    """Thermal delay drift plus two free-running oscillators.

    Attributes:
        length_km: span length.
        thermal_coefficient_s_per_km_k: delay change per km per kelvin.
        fractional_offset: constant fractional frequency difference between the
            two clocks.  Produces an offset that grows linearly with time and is
            trivially predictable — which is exactly why it must be in the model:
            a predictor that gets the easy term right and the thermal term wrong
            still looks good on RMS, and only the closed loop shows the
            difference.
        frequency_diffusion: random walk of that fractional offset, per
            sqrt(second).
        ambient_temperature_c: mean cable temperature.
        diurnal_amplitude_k: peak amplitude of the daily cycle.
        temperature_noise_k: short-term temperature fluctuation, per sqrt(second).
    """

    length_km: float = 50.0
    thermal_coefficient_s_per_km_k: float = C.DEFAULT_DELAY_THERMAL_COEFFICIENT_S_PER_KM_K
    fractional_offset: float = C.DEFAULT_CLOCK_FRACTIONAL_OFFSET
    frequency_diffusion: float = C.DEFAULT_CLOCK_FREQUENCY_DIFFUSION
    ambient_temperature_c: float = 21.0
    diurnal_amplitude_k: float = 3.0
    temperature_noise_k: float = 0.02

    def simulate(self, duration_s: float, dt_s: float,
                 temperature_c: ArrayLike | None = None,
                 rng: np.random.Generator | int | None = None) -> TimingTrajectory:
        """Simulate the offset over time.

        Args:
            duration_s: total simulated time.
            dt_s: sampling interval — the synchronisation update period.
            temperature_c: driving temperature per sample; synthesised when
                ``None``.
            rng: seed or generator.

        Returns:
            TimingTrajectory.

        Raises:
            ValueError: on a non-positive step, or a temperature series of the
                wrong length.
        """
        if dt_s <= 0 or duration_s <= 0:
            raise ValueError("duration_s and dt_s must be positive")

        generator = np.random.default_rng(rng)
        samples = int(round(duration_s / dt_s))
        time_s = np.arange(samples, dtype=float) * dt_s

        if temperature_c is None:
            phase = float(generator.uniform(0.0, 2.0 * np.pi))
            wander = (self.temperature_noise_k * np.sqrt(dt_s)
                      * generator.standard_normal(samples).cumsum())
            temperature = (self.ambient_temperature_c
                           + self.diurnal_amplitude_k
                           * np.sin(2.0 * np.pi * time_s / 86400.0 + phase)
                           + wander)
        else:
            temperature = np.asarray(temperature_c, dtype=float)
            if temperature.shape != time_s.shape:
                raise ValueError(
                    f"temperature_c has {temperature.size} samples but the run "
                    f"has {time_s.size}")

        thermal = (self.thermal_coefficient_s_per_km_k * self.length_km
                   * (temperature - temperature[0]))

        fractional = (self.fractional_offset
                      + self.frequency_diffusion * np.sqrt(dt_s)
                      * generator.standard_normal(samples).cumsum())
        clock = np.cumsum(fractional * dt_s)

        return TimingTrajectory(time_s=time_s, offset_s=thermal + clock,
                                temperature_c=temperature, length_km=self.length_km,
                                components={"thermal": thermal, "clock": clock})


# ---------------------------------------------------------------------------
# what the receiver can measure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasurementConfig:
    """How the offset is estimated, and what that costs.

    Attributes:
        photons: detections behind one estimate.  The arrival-time histogram's
            mean has standard error ``spread / sqrt(photons)``, so this is the
            same trade as the polarisation monitoring channel: every photon
            spent locating the pulse is a photon not spent on key.
        spread_s: RMS arrival spread, from :func:`timing_spread`.

    Note:
        A real receiver does not need a separate probe channel for this — it can
        build the histogram from the detections it already has, as long as it
        keeps a fine time-tag.  The cost is therefore not photons but
        *resolution*: the estimate is only as good as the number of detections
        in the last update period, which is why the useful update rate is set by
        the link's own gain and falls off with distance exactly when the thermal
        drift is at its worst.
    """

    photons: int = 2000
    spread_s: float = 100.0e-12

    def __post_init__(self) -> None:
        if self.photons < 1:
            raise ValueError("photons must be at least 1")
        if self.spread_s <= 0:
            raise ValueError("spread_s must be positive")

    @property
    def noise_s(self) -> float:
        """Standard error of one offset estimate."""
        return self.spread_s / np.sqrt(self.photons)


def measure_offset(offset_s: ArrayLike, config: MeasurementConfig,
                   rng: np.random.Generator | int | None = None
                   ) -> NDArray[np.float64]:
    """Estimate the offset from a finite arrival-time histogram.

    Args:
        offset_s: the true offset(s).
        config: how many photons, and how wide the arrival distribution is.
        rng: seed or generator.

    Returns:
        Noisy estimates, same shape.
    """
    generator = np.random.default_rng(rng)
    offset = np.asarray(offset_s, dtype=float)
    return offset + generator.normal(0.0, config.noise_s, size=offset.shape)


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

@runtime_checkable
class SyncPredictor(Protocol):
    """Anything that turns offset estimates into where the gate should sit."""

    name: str

    def reset(self) -> None:
        """Clear any state at the start of a run."""

    def predict(self, time_s: float, measured_s: float,
                temperature_c: float, horizon_s: float) -> float:
        """Predict the offset ``horizon_s`` from now, in seconds."""


@dataclass
class NoCorrection:
    """Leave the gate where it was calibrated.  The floor.

    On a link whose temperature moves at all, this is not a strawman but a
    prediction of what happens to an unsynchronised system: the key rate decays
    to zero within about a kelvin, and stays there.
    """

    name: str = "sem correção"

    def reset(self) -> None:
        self._first: float | None = None

    def __post_init__(self) -> None:
        self.reset()

    def predict(self, time_s: float, measured_s: float, temperature_c: float,
                horizon_s: float) -> float:
        if self._first is None:
            self._first = measured_s
        return self._first


@dataclass
class PeriodicCalibration:
    """Re-centre the gate every ``interval_s``, and hold in between.

    This is what deployed systems do: pause, scan the delay line for the peak,
    resume.  The scan costs time — the link produces no key during it — and
    between scans the gate is wherever the last one left it.  Both costs are
    modelled: :attr:`interval_s` sets how stale the correction gets, and
    :attr:`downtime_s` is charged against the key rate by
    :func:`run_sync_loop`.

    Attributes:
        interval_s: seconds between calibrations.
        downtime_s: how long each one takes, during which no key is produced.
        name: display name.
    """

    interval_s: float = 60.0
    downtime_s: float = 1.0
    name: str = "recalibração periódica"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._held = 0.0
        #: When the most recent calibration started; read by :func:`run_sync_loop`
        #: to charge the downtime. Public because the loop needs it and a private
        #: attribute read from outside is a worse contract than a documented one.
        self.last_calibration_s: float | None = None

    def predict(self, time_s: float, measured_s: float, temperature_c: float,
                horizon_s: float) -> float:
        if (self.last_calibration_s is None
                or time_s - self.last_calibration_s >= self.interval_s):
            self._held = measured_s
            self.last_calibration_s = time_s
        return self._held


@dataclass
class AlphaBetaTracker:
    """Smoothed offset and rate — the scalar twin of phase 3's alpha-beta filter.

    The same argument applies here as there: on a drift that is close to a
    random walk in rate, this is near optimal, it costs nothing, and it is the
    bar a learned predictor has to clear.  It should do well, because the clock
    term *is* a constant rate; where it should struggle is the thermal term,
    which turns when the temperature does.

    Attributes:
        alpha: weight on the newest measurement for the position estimate.
        beta: weight on the newest rate estimate.
        name: display name.
    """

    alpha: float = 0.35
    beta: float = 0.10
    name: str = "filtro alfa-beta"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state: float | None = None
        self._rate = 0.0
        self._previous_time: float | None = None

    def predict(self, time_s: float, measured_s: float, temperature_c: float,
                horizon_s: float) -> float:
        if self._state is None or self._previous_time is None:
            self._state, self._previous_time = measured_s, time_s
            return measured_s

        interval = time_s - self._previous_time
        if interval <= 0:
            return self._state

        predicted = self._state + self._rate * interval
        innovation = measured_s - predicted
        self._state = predicted + self.alpha * innovation
        self._rate = self._rate + (self.beta / interval) * innovation
        self._previous_time = time_s
        return self._state + self._rate * horizon_s


@dataclass
class LearnedSync:
    """A fitted regressor on a window of offsets and the thermometer.

    The features are the same shape as phase 3's: the last ``window`` estimates
    expressed as *differences from the current one*, so the model predicts an
    increment and is invariant to where the offset happens to be, plus the
    temperature and its rate.  The one thing it can do that the alpha-beta
    filter cannot is use dT/dt — and since the thermal term is proportional to
    the temperature *change*, that is not a marginal feature, it is the whole
    mechanism.

    Attributes:
        model: anything with ``predict(X) -> (n, 1)``.
        window: how many past estimates it was fitted on.
        scale: the units the increments were divided by at fit time.
        name: display name.
    """

    model: object
    window: int = 8
    scale: float = 1.0e-9
    name: str = "aprendido"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._history: deque = deque(maxlen=self.window)
        self._temperatures: deque = deque(maxlen=self.window)
        self._fallback = AlphaBetaTracker()
        self._fallback.reset()

    def predict(self, time_s: float, measured_s: float, temperature_c: float,
                horizon_s: float) -> float:
        self._history.append(measured_s)
        self._temperatures.append(temperature_c)
        fallback = self._fallback.predict(time_s, measured_s, temperature_c, horizon_s)
        if len(self._history) < self.window:
            return fallback

        features = _sync_features(list(self._history), list(self._temperatures),
                                  self.scale)
        increment = float(np.asarray(self.model.predict(features[None, :])).reshape(-1)[0])
        return measured_s + increment * self.scale


def _sync_features(history: Sequence[float], temperatures: Sequence[float],
                   scale: float) -> NDArray[np.float64]:
    """Lagged differences from the current estimate, plus temperature and dT/dt."""
    now = history[-1]
    lags = [(history[-1 - lag] - now) / scale for lag in range(1, len(history))]
    temperature_rate = (temperatures[-1] - temperatures[0]) / max(len(temperatures) - 1, 1)
    return np.asarray(lags + [temperatures[-1] - temperatures[0], temperature_rate],
                      dtype=float)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfig:
    """The receiver's gate and the loop that steers it.

    Attributes:
        width_s: full acceptance window.
        update_period_s: how often the offset is re-estimated.
        latency_s: delay between an estimate and the gate moving.  Must be a
            whole number of update periods, for the same reason phase 3's
            control loop insists on it: a predictor trained to forecast whole
            steps cannot be asked for a fraction of one.
        measurement: how the estimate is made.
        budget: the timing budget, for the arrival spread.
    """

    width_s: float = C.DEFAULT_GATE_WIDTH_S
    update_period_s: float = 1.0
    latency_s: float = 2.0
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    budget: TimingBudget = field(default_factory=TimingBudget)

    def __post_init__(self) -> None:
        if self.width_s <= 0:
            raise ValueError("width_s must be positive")
        if self.update_period_s <= 0:
            raise ValueError("update_period_s must be positive")
        if self.latency_s < 0:
            raise ValueError("latency_s cannot be negative")
        steps = self.latency_s / self.update_period_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"latency_s ({self.latency_s:g} s) must be an integer multiple of "
                f"update_period_s ({self.update_period_s:g} s); got {steps:.3f} steps")


@dataclass
class SyncResult:
    """Per-sample outcome of one synchronisation run.

    Attributes:
        time_s: sample times.
        residual_s: true offset minus where the gate was put — what the receiver
            is accountable for.
        efficiency: share of photons inside the gate at each sample.
        downtime: 1 where the link was stopped for a calibration.
        name: the predictor's name.
    """

    time_s: NDArray[np.float64]
    residual_s: NDArray[np.float64]
    efficiency: NDArray[np.float64]
    downtime: NDArray[np.float64]
    name: str = ""

    def summary(self) -> dict[str, float]:
        """The numbers worth quoting, in the order they matter."""
        live = 1.0 - self.downtime
        return {
            "sincronizador": self.name,
            "mean_efficiency": float(np.mean(self.efficiency * live)),
            "p05_efficiency": float(np.percentile(self.efficiency * live, 5)),
            "uptime": float(np.mean(live)),
            "mean_residual_ps": float(np.mean(np.abs(self.residual_s)) * 1e12),
            "p95_residual_ps": float(np.percentile(np.abs(self.residual_s), 95) * 1e12),
            "periods_below_half": float(np.mean(self.efficiency < 0.5)),
        }


def run_sync_loop(trajectory: TimingTrajectory, predictor: SyncPredictor,
                  gate: GateConfig | None = None,
                  rng: np.random.Generator | int | None = None) -> SyncResult:
    """Drive a predictor against a drifting delay and score what survives.

    The loop, per update *k*:

    1. the arrival histogram gives a noisy estimate of the offset at
       :math:`t_k`;
    2. the predictor forecasts the offset at :math:`t_k + \\tau`;
    3. the gate moves there at :math:`t_k + \\tau` and stays until the next
       update lands;
    4. at every sample in between, the residual is the true offset minus the
       gate's position, and the efficiency follows from :func:`gate_efficiency`.

    Args:
        trajectory: from :meth:`ClockModel.simulate`.
        predictor: the policy under test.
        gate: the gate and loop timing.
        rng: seed or generator for the measurement noise.

    Returns:
        SyncResult.

    Raises:
        ValueError: if the trajectory is sampled more coarsely than the update
            period.
    """
    gate = gate or GateConfig()
    generator = np.random.default_rng(rng)

    time_s = trajectory.time_s
    samples = len(time_s)
    dt_s = float(time_s[1] - time_s[0]) if samples > 1 else 1.0
    if gate.update_period_s < dt_s - 1e-12:
        raise ValueError(
            f"update_period_s ({gate.update_period_s} s) is shorter than the "
            f"trajectory's sampling interval ({dt_s} s)")

    spread_s = timing_spread(gate.budget, trajectory.length_km)
    measurement = MeasurementConfig(photons=gate.measurement.photons, spread_s=spread_s)

    stride = max(int(round(gate.update_period_s / dt_s)), 1)
    latency_steps = int(round(gate.latency_s / dt_s))
    horizon_s = latency_steps * dt_s

    update_indices = np.arange(0, samples, stride)
    measured = measure_offset(trajectory.offset_s[update_indices], measurement,
                              rng=generator)

    predictor.reset()
    predictions = np.empty(len(update_indices))
    calibrating = np.zeros(samples, dtype=float)
    downtime_steps = max(int(round(getattr(predictor, "downtime_s", 0.0) / dt_s)), 0)
    previous_calibration = None

    for step, index in enumerate(update_indices):
        predictions[step] = float(predictor.predict(
            float(time_s[index]), float(measured[step]),
            float(trajectory.temperature_c[index]), horizon_s))
        # A predictor that re-calibrates announces it by moving
        # ``last_calibration_s``; the link produces nothing while that runs.
        latest = getattr(predictor, "last_calibration_s", None)
        if latest is not None and latest != previous_calibration:
            previous_calibration = latest
            calibrating[index: min(index + downtime_steps, samples)] = 1.0

    effective = np.minimum(update_indices + latency_steps, samples - 1)
    in_force = np.searchsorted(effective, np.arange(samples), side="right") - 1
    gate_position = np.zeros(samples)
    uncorrected = in_force < 0
    gate_position[~uncorrected] = predictions[in_force[~uncorrected]]

    residual = trajectory.offset_s - gate_position
    efficiency = gate_efficiency(residual, gate.width_s, spread_s)

    return SyncResult(time_s=time_s, residual_s=residual, efficiency=efficiency,
                      downtime=calibrating,
                      name=getattr(predictor, "name", type(predictor).__name__))


# ---------------------------------------------------------------------------
# fitting the learned predictor
# ---------------------------------------------------------------------------

def build_sync_dataset(trajectories: Sequence[TimingTrajectory], gate: GateConfig,
                       window: int = 8, scale: float = 1.0e-9,
                       rng: np.random.Generator | int | None = None
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                  NDArray[np.int64], NDArray[np.float64]]:
    """Windowed features and the offset increment to predict.

    The target is the **future measurement**, not the future truth, for the same
    reason as phase 3: a deployed receiver can only ever retrain against what it
    measured, and regressing on ground truth credits the model with denoising it
    could only do because it was shown the answer.

    Args:
        trajectories: runs from :meth:`ClockModel.simulate`.
        gate: the loop timing; sets the update period and the horizon.
        window: past estimates per feature vector.
        scale: units the increments are expressed in — nanoseconds by default,
            which puts the target near unity.
        rng: seed for the measurement noise.

    Returns:
        ``(features, targets, episode, time_s)``.
    """
    generator = np.random.default_rng(rng)
    features: list[NDArray[np.float64]] = []
    targets: list[float] = []
    episodes: list[int] = []
    times: list[float] = []

    for index, trajectory in enumerate(trajectories):
        dt_s = float(trajectory.time_s[1] - trajectory.time_s[0])
        stride = max(int(round(gate.update_period_s / dt_s)), 1)
        horizon_steps = max(int(round(gate.latency_s / gate.update_period_s)), 1)

        update_indices = np.arange(0, len(trajectory.time_s), stride)
        spread_s = timing_spread(gate.budget, trajectory.length_km)
        measurement = MeasurementConfig(photons=gate.measurement.photons,
                                        spread_s=spread_s)
        measured = measure_offset(trajectory.offset_s[update_indices], measurement,
                                  rng=generator)
        temperature = trajectory.temperature_c[update_indices]

        for step in range(window - 1, len(update_indices) - horizon_steps):
            history = measured[step - window + 1: step + 1]
            features.append(_sync_features(list(history),
                                           list(temperature[step - window + 1: step + 1]),
                                           scale))
            targets.append((measured[step + horizon_steps] - measured[step]) / scale)
            episodes.append(index)
            times.append(float(trajectory.time_s[update_indices[step]]))

    return (np.asarray(features), np.asarray(targets),
            np.asarray(episodes, dtype=np.int64), np.asarray(times))


def fit_sync_predictor(features: NDArray[np.float64], targets: NDArray[np.float64],
                       train_mask: NDArray[np.bool_] | None = None,
                       alpha: float = 1.0, window: int = 8, scale: float = 1.0e-9,
                       name: str = "ridge") -> LearnedSync:
    """Fit a ridge regression on the windowed features.

    Linear, and deliberately so.  The offset is the sum of a linear clock term
    and a thermal term proportional to the temperature change, and both are
    linear in the features; a network would have nothing to add and would make
    the result harder to read.  Where a nonlinearity *would* help is a fibre
    whose thermal response saturates, which this model does not claim.

    Args:
        features, targets: from :func:`build_sync_dataset`.
        train_mask: rows to fit on; all of them when ``None``.
        alpha: ridge penalty.
        window, scale: carried into the predictor so inference matches training.
        name: display name.

    Returns:
        LearnedSync: ready to run in the loop.
    """
    from sklearn.linear_model import Ridge

    mask = np.ones(len(features), dtype=bool) if train_mask is None else train_mask
    model = Ridge(alpha=alpha)
    model.fit(features[mask], targets[mask])
    return LearnedSync(model=model, window=window, scale=scale, name=name)
