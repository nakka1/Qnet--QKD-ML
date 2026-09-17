"""A decoy-state link, watched over time: what an operator's log actually contains.

The classification problem in phase 5 is not "is the QBER high".  It is "the
QBER went up — is someone on the line, or is it the weather".  That question
only has an answer if the thing being simulated is a *session*: a link running
for tens of minutes, settling up every second, with the fibre drifting
underneath it the whole time and an event starting at a moment nobody announced.

So this module runs the link, not the formula.  Each settling period:

1. the fibre's polarisation has moved, and whatever the phase-3 compensator did
   not remove shows up as extra optical misalignment;
2. the link's loss and dark counts are whatever the current conditions make
   them;
3. those give an honest :class:`~qnet_autopilot.physics.decoy.ChannelResponse`;
4. an attack, if one is running, modifies it;
5. Bob counts a finite number of clicks and errors at each intensity, and the
   two ends publish the six numbers that make up
   :class:`~qnet_autopilot.physics.decoy.DecoyObservables`.

Everything downstream sees only those six numbers, the transmitter settings, and
the environmental sensors an operator really has — a thermometer on the cable
and the calibration taken when the link was commissioned.  The photon-number
yields, the attack label and the true misalignment are recorded alongside for
scoring, and are never features.

The events, and why these
-------------------------
Five of the seven classes are the ones a monitoring system confuses in practice:

* ``normal`` — the link as commissioned, drifting gently.
* ``thermal_drift`` — a temperature excursion.  The drift rate of the fibre's
  birefringence is coupled to *dT/dt* (see
  :class:`~qnet_autopilot.physics.fiber.PolarizationDriftModel`), so a ramp
  makes the residual polarisation error climb for as long as the temperature is
  moving and settle again afterwards.  This is the class that looks like an
  attack, and the reason the thermometer is a feature.
* ``loss_event`` — a bend, a disturbed splice, a connector gone dirty.  Both
  gains drop together, the error rate barely moves.
* ``intercept_resend`` — errors climb with no thermal cause.
* ``pns``, ``detector_blinding``, ``beam_splitting`` — from
  :mod:`qnet_autopilot.qkd.attacks`.

Every episode drifts thermally, not just the ``thermal_drift`` ones. An attack
episode with a frozen fibre would be separable by "does anything move at all",
which is not a discriminator any deployed link would have.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..physics import constants as C
from ..physics.decoy import (ChannelResponse, DecoySettings, channel_response,
                             decoy_key_rate_hz, sample_observables,
                             single_photon_bounds)
from ..physics.fiber import PolarizationDriftModel, transmittance
from .attacks import (Attack, BeamSplitting, DetectorBlinding, InterceptResend,
                      NoAttack, PhotonNumberSplitting)

__all__ = [
    "LinkConditions", "Event", "SessionConfig", "EVENTS", "event_labels",
    "simulate_session", "SESSION_COLUMNS",
]


# ---------------------------------------------------------------------------
# the link
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkConditions:
    """The physical link, at one moment.

    Attributes:
        length_km: span between Alice and Bob.
        attenuation_db_per_km: fibre loss.
        coupling_efficiency: interface loss at Bob, applied once — Alice's
            attenuator sets the intensity *after* her own optics, so her
            coupling is already inside ``signal_intensity``.
        detector_efficiency: single-photon detection efficiency.
        excess_loss_db: everything the model does not name — connectors,
            patch panels, a filter, and whatever a bend is doing today.  This
            is the term a ``loss_event`` moves.
        dark_count_rate_hz: per detector.
        gate_width_s: acceptance window around the expected arrival time.
        detectors: how many detectors are gated per pulse.  Two for a standard
            polarisation receiver, and it matters: dark counts scale with it.
        misalignment_error: optical misalignment floor, after compensation.

    Note:
        ``transmittance`` deliberately excludes Alice's intensity and Bob's
        gate: the first belongs to :class:`~qnet_autopilot.physics.decoy.DecoySettings`
        and the second to :mod:`qnet_autopilot.qkd.timing`, which reduces it by
        whatever fraction of the pulse falls outside the window.
    """

    length_km: float = 50.0
    attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM
    coupling_efficiency: float = 0.85
    detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY
    excess_loss_db: float = 1.0
    dark_count_rate_hz: float = C.DEFAULT_DARK_COUNT_RATE_HZ
    gate_width_s: float = C.DEFAULT_GATE_WIDTH_S
    detectors: int = 2
    misalignment_error: float = C.DEFAULT_MISALIGNMENT_ERROR

    def __post_init__(self) -> None:
        if self.length_km <= 0:
            raise ValueError("length_km must be positive")
        if self.detectors < 1:
            raise ValueError("detectors must be at least one")

    @property
    def transmittance(self) -> float:
        """Probability one photon leaving Alice fires one of Bob's detectors."""
        fibre = float(transmittance(self.length_km, self.attenuation_db_per_km))
        excess = 10.0 ** (-self.excess_loss_db / 10.0)
        return fibre * excess * self.coupling_efficiency * self.detector_efficiency

    @property
    def dark_count_probability(self) -> float:
        """``Y_0``: probability that a gate produces a click with no photon in it."""
        return float(np.clip(self.dark_count_rate_hz * self.gate_width_s
                             * self.detectors, 0.0, 1.0))

    def response(self, misalignment_error: float | None = None) -> ChannelResponse:
        """The honest channel these conditions produce."""
        error = (self.misalignment_error if misalignment_error is None
                 else misalignment_error)
        return channel_response(self.transmittance, self.dark_count_probability,
                                float(np.clip(error, 0.0, 0.5)))


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Something that starts partway through an episode, and what it does.

    An event is either an attack (an :class:`~qnet_autopilot.qkd.attacks.Attack`
    switched on at ``onset_s``) or a fault (a change to the link conditions or
    to the environment).  Both are expressed here so an episode generator does
    not have to know which kind it is holding.

    Attributes:
        label: the class name — the classification target.
        attack: the attack to run from the onset, or ``None``.
        profile: how the event arrives — ``"step"`` (full strength at the
            onset), ``"ramp"`` (linearly over ``ramp_s``) or ``"intermittent"``
            (on for ``duty_cycle`` of every ``cycle_s``).  A dataset whose
            attacks only ever arrive as a step teaches a classifier to look for
            steps, and an eavesdropper who knows a monitoring system is watching
            is exactly the one who will not give it one.
        ramp_s: how long a ramped event takes to reach full strength.
        duty_cycle: share of each cycle an intermittent event is running.
        cycle_s: period of an intermittent event.
        excess_loss_step_db: extra loss appearing at the onset.
        dark_count_factor: multiplier on the dark count rate from the onset.
        temperature_ramp_k: temperature excursion applied from the onset.
        temperature_ramp_s: how long that excursion takes to arrive.  The
            *rate* is what drives polarisation drift, so a slower ramp of the
            same size is a milder event, not a longer one.
        attack_is_detectable: whether the physics leaves any signature at all.
            ``False`` for :class:`~qnet_autopilot.qkd.attacks.BeamSplitting`,
            and the report scores that class separately rather than counting it
            as an accuracy failure.
    """

    label: str
    attack: Attack | None = None
    profile: str = "step"
    ramp_s: float = 180.0
    duty_cycle: float = 0.5
    cycle_s: float = 60.0
    excess_loss_step_db: float = 0.0
    dark_count_factor: float = 1.0
    temperature_ramp_k: float = 0.0
    temperature_ramp_s: float = 120.0
    attack_is_detectable: bool = True

    def __post_init__(self) -> None:
        if self.profile not in ("step", "ramp", "intermittent"):
            raise ValueError(
                f"unknown profile {self.profile!r}; expected 'step', 'ramp' or "
                "'intermittent'")

    def strength(self, time_s: NDArray[np.float64], onset_s: float
                 ) -> NDArray[np.float64]:
        """Severity in force at each sample, in ``[0, 1]``.

        Args:
            time_s: sample times.
            onset_s: when the event starts; ``nan`` for an event-free episode.

        Returns:
            One weight per sample.  Zero before the onset, and zero everywhere
            for an event-free episode.

        Note:
            The weight multiplies *everything* the event does — the attack's
            severity, the loss step and the dark count factor — so a ramped
            intercept-and-resend really does intercept a growing share of the
            traffic rather than switching on a fixed one late.
        """
        if not np.isfinite(onset_s):
            return np.zeros_like(time_s)

        elapsed = time_s - onset_s
        running = elapsed >= 0.0

        if self.profile == "ramp":
            weight = np.clip(elapsed / max(self.ramp_s, 1e-9), 0.0, 1.0)
        elif self.profile == "intermittent":
            phase = np.mod(np.maximum(elapsed, 0.0), max(self.cycle_s, 1e-9))
            weight = (phase < self.duty_cycle * self.cycle_s).astype(float)
        else:
            weight = np.ones_like(time_s)

        return np.where(running, weight, 0.0)


#: The seven classes, with the severities the phase-5 datasets sample around.
#: Every magnitude here is drawn per episode in
#: :func:`~qnet_autopilot.datagen.qkd.generate_attack_dataset`; these are the
#: centres, not the values.
EVENTS: tuple[Event, ...] = (
    Event(label="normal"),
    Event(label="thermal_drift", temperature_ramp_k=10.0),
    Event(label="loss_event", excess_loss_step_db=3.0),
    Event(label="intercept_resend", attack=InterceptResend(fraction=0.25)),
    Event(label="pns", attack=PhotonNumberSplitting(blocked_fraction=0.5)),
    Event(label="detector_blinding", attack=DetectorBlinding()),
    Event(label="beam_splitting", attack=BeamSplitting(), attack_is_detectable=False),
)


def event_labels() -> tuple[str, ...]:
    """The class names, in the order the report tabulates them."""
    return tuple(event.label for event in EVENTS)


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """How one episode is run.

    Attributes:
        link: the link as commissioned.
        settings: the transmitter configuration.
        period_s: settling period — how long counts accumulate before the two
            ends publish a set of observables.  One second at a gigahertz clock
            is 10\\ :sup:`9` pulses, which is enough for the gains and nowhere
            near enough to make the single-photon bound quiet.
        duration_s: episode length.
        min_active_periods: periods the event is guaranteed to run for.  The
            onset is drawn uniformly between the end of the baseline and this
            many periods before the end, so every episode contributes a
            comparable number of event-active rows.  Without it the onset
            sometimes lands near the end of the log and that episode teaches
            almost nothing, which shows up as wildly uneven class counts in the
            test split rather than as anything interesting.
        baseline_periods: periods at the start guaranteed to be event-free, so
            the episode carries its own calibration.  A deployed system has one
            of these from commissioning; without it the features would have to
            be absolute, and an absolute gain says as much about the span length
            as about the channel.
        drift: the fibre's polarisation drift model.
        residual_fraction: the share of the drift-induced error the phase-3
            compensator fails to remove.  1.0 is an uncompensated link — a
            useful stress case and a terrible link.  The default 0.2 is a
            deliberately *pessimistic* reading of the phase-3 result: the
            learned compensator does better than that on the phase-3 operating
            point, but it runs at a kilohertz against a monitoring channel that
            costs photons, and a link settling up once a second is not entitled
            to assume the good case.  It is also the number that sets how hard
            this problem is — at 0.05 the thermal class barely moves the QBER
            and the classifier's job is trivial, which would flatter it.
        ambient_temperature_c: mean cable temperature.
        diurnal_amplitude_k: peak-to-mean swing of the slow daily cycle.
    """

    link: LinkConditions = field(default_factory=LinkConditions)
    settings: DecoySettings = field(default_factory=DecoySettings)
    period_s: float = 1.0
    duration_s: float = 600.0
    baseline_periods: int = 60
    min_active_periods: int = 60
    drift: PolarizationDriftModel = field(default_factory=PolarizationDriftModel)
    residual_fraction: float = 0.2
    ambient_temperature_c: float = 21.0
    diurnal_amplitude_k: float = 2.0

    def __post_init__(self) -> None:
        if self.period_s <= 0:
            raise ValueError("period_s must be positive")
        if self.duration_s < self.period_s * (self.baseline_periods
                                              + self.min_active_periods + 1):
            raise ValueError(
                f"duration_s ({self.duration_s} s) leaves no room for "
                f"{self.baseline_periods} baseline periods and "
                f"{self.min_active_periods} active ones of {self.period_s} s")
        if not 0.0 <= self.residual_fraction <= 1.0:
            raise ValueError("residual_fraction must be in [0, 1]")

    @property
    def periods(self) -> int:
        """Settling periods in one episode."""
        return int(round(self.duration_s / self.period_s))

    @property
    def pulses_per_period(self) -> float:
        """Pulses sent per settling period, across all three intensities."""
        return self.settings.clock_rate_hz * self.period_s


#: What one row of a session log means.  Columns beginning with ``true_`` are
#: recorded for scoring and must never be used as features — they are not
#: things Alice and Bob can measure.
SESSION_COLUMNS: dict[str, str] = {
    "episode": "episode index",
    "time_s": "seconds since the start of the episode",
    "period": "settling period index",
    "label": "event class, the classification target",
    "onset_s": "when the event started; NaN for an event-free episode",
    "active": "1 while the event is actually running, 0 before and between bursts",
    "strength": "severity in force this period, 0 to 1 (ramped and intermittent events)",
    "signal_gain": "clicks per signal pulse, Q_mu",
    "decoy_gain": "clicks per weak-decoy pulse, Q_nu",
    "vacuum_gain": "clicks per empty pulse, Y_0",
    "signal_error": "error rate among signal clicks, E_mu",
    "decoy_error": "error rate among decoy clicks, E_nu",
    "vacuum_error": "error rate among vacuum clicks",
    "signal_intensity": "mean photon number of the signal state",
    "decoy_intensity": "mean photon number of the weak decoy state",
    "single_photon_yield_lower": "Y_1 lower bound from the decoy analysis",
    "single_photon_error_upper": "e_1 upper bound from the decoy analysis",
    "key_rate_hz": "secret key rate implied by this period's observables",
    "temperature_c": "cable temperature, from the operator's sensor",
    "expected_gain": "signal gain the commissioning calibration predicts",
    "pulses": "pulses sent in this period",
    "true_misalignment": "optical misalignment actually in force (not a feature)",
    "true_transmittance": "channel transmittance actually in force (not a feature)",
    "true_single_photon_yield": "Y_1 actually in force (not a feature)",
}


def simulate_session(config: SessionConfig, event: Event,
                     rng: np.random.Generator | int | None = None,
                     episode: int = 0) -> pd.DataFrame:
    """Run one episode and return its log, one row per settling period.

    Args:
        config: how the link is run.
        event: what happens, and what class the episode belongs to.
        rng: seed or generator.  Drives the fibre, the temperature and the
            photon counting.
        episode: index recorded in the ``episode`` column.

    Returns:
        A DataFrame with the columns of :data:`SESSION_COLUMNS`.

    Note:
        The onset is drawn uniformly in the part of the episode after the
        baseline.  Fixing it would let a classifier learn the clock instead of
        the physics — and a windowed feature builder would learn it especially
        fast.
    """
    generator = np.random.default_rng(rng)
    periods = config.periods
    time_s = np.arange(periods, dtype=float) * config.period_s

    earliest = config.baseline_periods * config.period_s
    latest = config.duration_s - config.min_active_periods * config.period_s
    onset_s = (float(generator.uniform(earliest, latest))
               if event.label != "normal" else float("nan"))
    strength = event.strength(time_s, onset_s)
    running = (time_s >= onset_s) if np.isfinite(onset_s) else np.zeros(periods, bool)
    active = strength > 0.0

    temperature = _temperature_series(config, event, time_s, running, generator)
    misalignment = _misalignment_series(config, time_s, temperature, generator)

    calibrated = config.link.response()
    expected_gain = float(calibrated.gain(config.settings.signal_intensity))

    rows: list[dict[str, float | str]] = []
    for index in range(periods):
        weight = float(strength[index])
        link = config.link
        if weight > 0.0:
            link = replace(
                link,
                excess_loss_db=link.excess_loss_db + event.excess_loss_step_db * weight,
                dark_count_rate_hz=link.dark_count_rate_hz
                * (1.0 + (event.dark_count_factor - 1.0) * weight))

        response = link.response(misalignment[index])
        attack = (event.attack.with_strength(weight)
                  if (weight > 0.0 and event.attack is not None) else NoAttack())
        attacked = attack.apply(response, config.settings)

        observables = sample_observables(attacked, config.settings,
                                         config.pulses_per_period, rng=generator)
        single_yield, single_error = single_photon_bounds(observables)

        rows.append({
            "episode": episode,
            "time_s": float(time_s[index]),
            "period": index,
            "label": event.label,
            "onset_s": onset_s,
            "active": float(active[index]),
            "strength": weight,
            "signal_gain": observables.signal_gain,
            "decoy_gain": observables.decoy_gain,
            "vacuum_gain": observables.vacuum_gain,
            "signal_error": observables.signal_error,
            "decoy_error": observables.decoy_error,
            "vacuum_error": observables.vacuum_error,
            "signal_intensity": config.settings.signal_intensity,
            "decoy_intensity": config.settings.decoy_intensity,
            "single_photon_yield_lower": single_yield,
            "single_photon_error_upper": single_error,
            "key_rate_hz": decoy_key_rate_hz(observables),
            "temperature_c": float(temperature[index]),
            "expected_gain": expected_gain,
            "pulses": config.pulses_per_period,
            "true_misalignment": float(misalignment[index]),
            "true_transmittance": link.transmittance,
            "true_single_photon_yield": float(attacked.yields[1]),
        })

    return pd.DataFrame(rows, columns=list(SESSION_COLUMNS))


def _temperature_series(config: SessionConfig, event: Event,
                        time_s: NDArray[np.float64], active: NDArray[np.bool_],
                        generator: np.random.Generator) -> NDArray[np.float64]:
    """Cable temperature: a slow diurnal cycle, sensor noise, and the event.

    The excursion is a saturating ramp rather than a step, because the fibre's
    drift rate is driven by *dT/dt* and a step would put an impulse into it that
    no real environment produces.
    """
    phase = float(generator.uniform(0.0, 2.0 * np.pi))
    diurnal = config.diurnal_amplitude_k * np.sin(
        2.0 * np.pi * time_s / 86400.0 + phase)
    noise = 0.02 * generator.standard_normal(len(time_s)).cumsum() * np.sqrt(
        config.period_s / 60.0)

    excursion = np.zeros_like(time_s)
    if event.temperature_ramp_k and np.any(active):
        onset = float(time_s[np.argmax(active)])
        elapsed = np.maximum(time_s - onset, 0.0)
        excursion = event.temperature_ramp_k * (
            1.0 - np.exp(-elapsed / max(event.temperature_ramp_s, 1e-9)))

    return config.ambient_temperature_c + diurnal + noise + excursion


def _misalignment_series(config: SessionConfig, time_s: NDArray[np.float64],
                         temperature: NDArray[np.float64],
                         generator: np.random.Generator) -> NDArray[np.float64]:
    """Optical misalignment per period: the floor plus what compensation missed.

    The fibre is simulated by the same drift model phase 3 uses, driven by this
    episode's temperature, and its instantaneous QBER is scaled by
    ``residual_fraction`` — the share a compensator running underneath does not
    remove.  Using the phase-3 model rather than an invented error process is
    what makes the ``thermal_drift`` class hard in the right way: the error
    climbs while the temperature is *moving*, not while it is high, which is a
    distinction no attack reproduces and no threshold captures.
    """
    trajectory = config.drift.simulate(
        duration_s=float(time_s[-1] + config.period_s),
        dt_s=config.period_s,
        temperature_c=temperature,
        rng=generator)
    residual = config.residual_fraction * trajectory.qber[:len(time_s)]
    return np.clip(config.link.misalignment_error + residual, 0.0, 0.5)
