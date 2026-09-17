"""Decoy-state BB84: what Bob observes, and what can be proved from it.

Why this module exists at all
-----------------------------
:mod:`.link` answers the entanglement question — how often a heralded pair
appears and what it is worth.  A prepare-and-measure QKD link asks a different
one.  Its transmitter is an attenuated laser, so the photon number of each
pulse is Poisson, and a fraction of pulses carry two photons or more.  An
eavesdropper can keep one of those photons and forward the rest, learning the
bit without disturbing it: the *photon number splitting* attack.  Against it,
counting errors is useless — the attack is error-free by construction.

Decoy states are the answer, and they are the reason this module is written
around **yields per photon number** rather than around an overall error rate.
Alice varies the intensity between pulses at random.  Eve cannot tell which
intensity a pulse was sent at, so whatever she does to *n*-photon pulses she
does to all of them equally, and the yields :math:`Y_n` she leaves behind are a
property of the channel, not of the intensity.  Bob's observed gain at
intensity :math:`\\mu` is then a Poisson mixture of those yields,

.. math:: Q_\\mu = \\sum_n \\frac{\\mu^n e^{-\\mu}}{n!} Y_n ,

and measuring :math:`Q` at several intensities constrains the individual
:math:`Y_n` — in particular the single-photon yield :math:`Y_1`, which is the
only part of the signal that can produce secret key.  An attack that suppresses
single photons and compensates with multi-photon pulses keeps :math:`Q_\\mu`
fixed and is invisible in the gain alone; it cannot keep :math:`Q_\\mu` *and*
:math:`Q_\\nu` fixed, and that is what :func:`single_photon_bounds` detects.

The structure this imposes on the code
--------------------------------------
Everything downstream — the attack models in :mod:`qnet_autopilot.qkd.attacks`,
the classifier's features, the agent's reward — is written in terms of a
:class:`ChannelResponse`: a vector of yields and a vector of per-photon-number
error rates.  An honest channel produces one from :func:`channel_response`; an
attack produces a different one from the same interface.  Gains and error rates
are then *derived*, never assumed, so an attack cannot accidentally be given a
signature the physics would not produce.

The bounds are the vacuum+weak-decoy analysis of Ma, Qi, Zhao & Lo (Phys. Rev.
A 72 012326, 2005), and the key rate is GLLP (Gottesman, Lo, Lütkenhaus &
Preskill, 2004) as used there.  Both are asymptotic in the number of pulses;
:func:`sample_observables` draws the finite counts a real run produces, and
:func:`estimate_from_counts` runs the same bounds on those counts, which is
where the statistical noise that the phase-5 classifier has to see through
comes from.

Conventions, because they differ between papers
-----------------------------------------------
* Intensity ``mu`` is the mean photon number *leaving Alice*.  Every loss after
  that, including Bob's detector efficiency, lives in ``transmittance``.
* ``Y_0`` is the probability of a click with no photon sent — dark counts and
  stray light, per gate, both detectors together.
* Rates are per *pulse*.  Multiply by the clock and the sifting factor for bits
  per second; :func:`decoy_key_rate_hz` does it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as C
from .link import binary_entropy

__all__ = [
    "DecoySettings",
    "ChannelResponse",
    "DecoyObservables",
    "poisson_weights",
    "channel_response",
    "single_photon_bounds",
    "decoy_key_rate",
    "decoy_key_rate_hz",
    "observe",
    "apply_detector_saturation",
    "sample_observables",
    "estimate_from_counts",
    "optimal_signal_intensity",
]

#: Photon numbers carried explicitly.  The Poisson tail beyond this is below
#: 1e-12 for any intensity this module is used at (mu <= 1), and folding it into
#: the last term rather than dropping it keeps the gain exactly normalised.
MAX_PHOTONS = 12


# ---------------------------------------------------------------------------
# transmitter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecoySettings:
    """What Alice transmits: the intensities and how often each is sent.

    Attributes:
        signal_intensity: mean photon number of the signal state, ``mu``.  Key
            comes from this one; the others exist only to bound the yields.
        decoy_intensity: mean photon number of the weak decoy state, ``nu``.
            Must be below the signal, or the bound is vacuous.
        vacuum_probability: share of pulses sent empty.  These measure ``Y_0``
            directly, which is what makes the vacuum+weak protocol tight.
        signal_probability: share of pulses sent at ``signal_intensity``.
        clock_rate_hz: pulse repetition rate.
        sifting_factor: fraction of detections that survive basis sifting.

    Note:
        The decoy probability is whatever the other two leave.  Keeping it
        implicit means the three always sum to one, which is otherwise an easy
        thing to get subtly wrong in an agent that moves them.
    """

    signal_intensity: float = C.DEFAULT_SIGNAL_INTENSITY
    decoy_intensity: float = C.DEFAULT_DECOY_INTENSITY
    vacuum_probability: float = C.DEFAULT_DECOY_PROBABILITIES[2]
    signal_probability: float = C.DEFAULT_DECOY_PROBABILITIES[0]
    clock_rate_hz: float = C.DEFAULT_CLOCK_RATE_HZ
    sifting_factor: float = C.DEFAULT_SIFTING_FACTOR

    def __post_init__(self) -> None:
        if not 0.0 < self.signal_intensity <= 2.0:
            raise ValueError("signal_intensity must be in (0, 2]")
        if not 0.0 < self.decoy_intensity < self.signal_intensity:
            raise ValueError(
                f"decoy_intensity ({self.decoy_intensity}) must be positive and "
                f"below signal_intensity ({self.signal_intensity}); a decoy at or "
                "above the signal constrains nothing")
        if self.decoy_probability <= 0.0:
            raise ValueError(
                "signal_probability and vacuum_probability leave no pulses for "
                "the weak decoy")
        if self.clock_rate_hz <= 0:
            raise ValueError("clock_rate_hz must be positive")
        if not 0.0 < self.sifting_factor <= 1.0:
            raise ValueError("sifting_factor must be in (0, 1]")

    @property
    def decoy_probability(self) -> float:
        """Share of pulses sent at ``decoy_intensity``."""
        return 1.0 - self.signal_probability - self.vacuum_probability

    @property
    def intensities(self) -> tuple[float, float, float]:
        """``(signal, decoy, vacuum)`` mean photon numbers."""
        return (self.signal_intensity, self.decoy_intensity, 0.0)

    @property
    def probabilities(self) -> tuple[float, float, float]:
        """``(signal, decoy, vacuum)`` transmission probabilities."""
        return (self.signal_probability, self.decoy_probability,
                self.vacuum_probability)

    def replace(self, **changes) -> "DecoySettings":
        """A copy with some fields changed — the agent's action, applied."""
        fields = {"signal_intensity": self.signal_intensity,
                  "decoy_intensity": self.decoy_intensity,
                  "vacuum_probability": self.vacuum_probability,
                  "signal_probability": self.signal_probability,
                  "clock_rate_hz": self.clock_rate_hz,
                  "sifting_factor": self.sifting_factor}
        fields.update(changes)
        return DecoySettings(**fields)


def poisson_weights(intensity: ArrayLike,
                    max_photons: int = MAX_PHOTONS) -> NDArray[np.float64]:
    """Probability that a pulse of mean photon number ``mu`` carries *n* photons.

    Args:
        intensity: mean photon number(s).
        max_photons: highest photon number carried explicitly.

    Returns:
        Weights of shape ``(..., max_photons + 1)``, summing to exactly one:
        the Poisson tail above ``max_photons`` is folded into the last entry
        rather than discarded, so a gain computed from these is never quietly
        short of probability.
    """
    mu = np.asarray(intensity, dtype=float)[..., None]
    photons = np.arange(max_photons + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_weights = (photons * np.log(np.where(mu > 0, mu, 1.0))
                       - mu - _log_factorial(photons))
    weights = np.where(mu > 0, np.exp(log_weights), (photons == 0).astype(float))
    weights[..., -1] += 1.0 - weights.sum(axis=-1)
    return weights


def _log_factorial(n: NDArray[np.int64]) -> NDArray[np.float64]:
    """``log(n!)`` without scipy, exact for the small n this module uses."""
    return np.cumsum(np.log(np.concatenate([[1.0], np.arange(1, len(n), dtype=float)])))


# ---------------------------------------------------------------------------
# channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelResponse:
    """What the channel does to an *n*-photon pulse, for every *n*.

    This is the object an attack modifies.  Eve does not get to choose what Bob
    observes; she chooses what happens to each photon number, and the
    observables follow.

    Attributes:
        yields: ``Y_n``, the probability Bob registers a click given Alice sent
            *n* photons, shape ``(max_photons + 1,)``.
        errors: ``e_n``, the probability that click is wrong, same shape.
        label: what produced this response — ``"normal"`` for an honest
            channel, otherwise the attack or fault name.  Carried through to
            the datasets as the classification target.

    Note:
        ``yields[0]`` is the dark count probability and ``errors[0]`` is 0.5 by
        physics, not by assumption: a click with no photon behind it is
        uncorrelated with the bit Alice sent.
    """

    yields: NDArray[np.float64]
    errors: NDArray[np.float64]
    label: str = "normal"

    def __post_init__(self) -> None:
        if self.yields.shape != self.errors.shape:
            raise ValueError("yields and errors must have the same shape")
        if np.any(self.yields < 0) or np.any(self.yields > 1):
            raise ValueError("yields must be probabilities")
        if np.any(self.errors < 0) or np.any(self.errors > 1):
            raise ValueError("errors must be probabilities")

    @property
    def max_photons(self) -> int:
        return len(self.yields) - 1

    def gain(self, intensity: ArrayLike) -> NDArray[np.float64]:
        """``Q_mu``: probability of a click, per pulse sent at this intensity."""
        weights = poisson_weights(intensity, self.max_photons)
        return weights @ self.yields

    def error_rate(self, intensity: ArrayLike) -> NDArray[np.float64]:
        """``E_mu``: fraction of those clicks that are wrong.

        Returns:
            The error rate, or 0.5 where the gain is zero — an un-measurable
            error rate is not a low one, and returning zero there would let a
            key-rate formula read a dead link as a perfect one.
        """
        weights = poisson_weights(intensity, self.max_photons)
        gain = weights @ self.yields
        errors = weights @ (self.yields * self.errors)
        return np.where(gain > 0, errors / np.maximum(gain, 1e-300), 0.5)

    def replace(self, yields: NDArray[np.float64] | None = None,
                errors: NDArray[np.float64] | None = None,
                label: str | None = None) -> "ChannelResponse":
        """A copy with some components replaced — how an attack is expressed."""
        return ChannelResponse(
            yields=self.yields.copy() if yields is None else np.asarray(yields, float),
            errors=self.errors.copy() if errors is None else np.asarray(errors, float),
            label=self.label if label is None else label)


def channel_response(transmittance: float,
                     dark_count_probability: float,
                     misalignment_error: float = C.DEFAULT_MISALIGNMENT_ERROR,
                     max_photons: int = MAX_PHOTONS) -> ChannelResponse:
    """The honest channel: loss, dark counts and a fixed optical misalignment.

    An *n*-photon pulse produces a click if any of its photons survives, or if a
    dark count fires:

    .. math:: Y_n = Y_0 + \\eta_n - Y_0 \\eta_n, \\qquad
              \\eta_n = 1 - (1 - \\eta)^n .

    The error attached to it mixes the two sources — a real photon is wrong with
    probability ``misalignment_error``, a dark count with probability one half —
    weighted by how often each was responsible:

    .. math:: e_n = \\frac{e_0 Y_0 + e_d \\eta_n}{Y_n}\\, .

    Args:
        transmittance: end-to-end probability that one photon leaving Alice
            fires one of Bob's detectors — fibre loss times coupling times
            detector efficiency.
        dark_count_probability: ``Y_0``, per gate, both detectors.
        misalignment_error: intrinsic error of a genuine detection.
        max_photons: truncation of the photon-number axis.

    Returns:
        ChannelResponse: the honest channel's yields and errors.

    Raises:
        ValueError: if any argument is outside ``[0, 1]``.
    """
    for name, value in (("transmittance", transmittance),
                        ("dark_count_probability", dark_count_probability),
                        ("misalignment_error", misalignment_error)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    photons = np.arange(max_photons + 1)
    photon_yield = 1.0 - (1.0 - transmittance) ** photons
    yields = (dark_count_probability + photon_yield
              - dark_count_probability * photon_yield)

    numerator = (C.DARK_COUNT_ERROR_RATE * dark_count_probability
                 + misalignment_error * photon_yield)
    errors = np.where(yields > 0, numerator / np.maximum(yields, 1e-300),
                      C.DARK_COUNT_ERROR_RATE)
    return ChannelResponse(yields=yields, errors=errors, label="normal")


# ---------------------------------------------------------------------------
# what Bob sees, and what Alice and Bob can prove from it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecoyObservables:
    """One settling period's worth of what the two ends actually exchange.

    These eight numbers are the *entire* public record of a decoy-state run —
    everything the security proof uses, everything a monitoring system can log,
    and, in phase 5, everything the attack classifier is allowed to see.  The
    per-photon-number yields behind them are not observable and are deliberately
    not carried here.

    Attributes:
        signal_gain: ``Q_mu``, clicks per signal pulse.
        decoy_gain: ``Q_nu``, clicks per weak-decoy pulse.
        vacuum_gain: ``Y_0``, clicks per empty pulse.
        signal_error: ``E_mu``.
        decoy_error: ``E_nu``.
        vacuum_error: error rate on vacuum clicks; one half up to statistics,
            and a useful check that the run is what it says it is.
        settings: the transmitter configuration these were measured under.
        pulses: how many pulses the period covered, in total across the three
            intensities.  ``inf`` for asymptotic observables.
    """

    signal_gain: float
    decoy_gain: float
    vacuum_gain: float
    signal_error: float
    decoy_error: float
    vacuum_error: float
    settings: DecoySettings
    pulses: float = float("inf")

    def as_dict(self) -> dict[str, float]:
        """Flat view, for a DataFrame row or a feature vector."""
        return {
            "signal_gain": self.signal_gain,
            "decoy_gain": self.decoy_gain,
            "vacuum_gain": self.vacuum_gain,
            "signal_error": self.signal_error,
            "decoy_error": self.decoy_error,
            "vacuum_error": self.vacuum_error,
            "signal_intensity": self.settings.signal_intensity,
            "decoy_intensity": self.settings.decoy_intensity,
            "pulses": self.pulses,
        }


def observe(response: ChannelResponse, settings: DecoySettings) -> DecoyObservables:
    """Asymptotic observables — the infinite-statistics limit of a run.

    Args:
        response: the channel, honest or attacked.
        settings: the transmitter configuration.

    Returns:
        DecoyObservables: exact gains and error rates, with ``pulses=inf``.
    """
    signal, decoy, _ = settings.intensities
    return DecoyObservables(
        signal_gain=float(response.gain(signal)),
        decoy_gain=float(response.gain(decoy)),
        vacuum_gain=float(response.yields[0]),
        signal_error=float(response.error_rate(signal)),
        decoy_error=float(response.error_rate(decoy)),
        vacuum_error=float(response.errors[0]),
        settings=settings,
    )


def single_photon_bounds(observables: DecoyObservables
                         ) -> tuple[float, float]:
    """Bound the single-photon yield from below and its error rate from above.

    The vacuum+weak decoy bounds of Ma et al. (2005), eqs. 34 and 37.  With
    :math:`\\nu < \\mu` and the vacuum measuring :math:`Y_0` directly,

    .. math::

        Y_1 \\ge Y_1^L = \\frac{\\mu}{\\mu\\nu - \\nu^2}
            \\Bigl( Q_\\nu e^{\\nu} - Q_\\mu e^{\\mu} \\frac{\\nu^2}{\\mu^2}
                    - \\frac{\\mu^2 - \\nu^2}{\\mu^2} Y_0 \\Bigr),

    .. math::

        e_1 \\le e_1^U = \\frac{E_\\nu Q_\\nu e^{\\nu} - e_0 Y_0}{Y_1^L \\nu} .

    Both are *bounds*, and both are what an attack has to survive rather than
    what it has to fool: photon number splitting keeps :math:`Q_\\mu` fixed while
    hollowing out :math:`Y_1`, and the first inequality is what notices.

    Args:
        observables: one settling period.

    Returns:
        ``(Y_1_lower, e_1_upper)``.  The yield bound is clamped at zero — a
        negative bound means the observed gains are inconsistent with *any*
        single-photon contribution, which is a real answer and yields no key.
        The error bound is clamped at one half, where it is uninformative.

    Note:
        The bound uses only the observables, never the underlying yields, so it
        is exactly as blind as the real protocol is.  Running it on sampled
        counts is what makes the phase-5 statistics honest.
    """
    mu = observables.settings.signal_intensity
    nu = observables.settings.decoy_intensity
    if not 0.0 < nu < mu:
        raise ValueError("bounds require 0 < decoy_intensity < signal_intensity")

    vacuum = observables.vacuum_gain
    lower = (mu / (mu * nu - nu * nu)) * (
        observables.decoy_gain * np.exp(nu)
        - observables.signal_gain * np.exp(mu) * (nu * nu) / (mu * mu)
        - ((mu * mu - nu * nu) / (mu * mu)) * vacuum)
    single_yield = float(max(lower, 0.0))

    if single_yield <= 0.0:
        return 0.0, 0.5

    numerator = (observables.decoy_error * observables.decoy_gain * np.exp(nu)
                 - C.DARK_COUNT_ERROR_RATE * vacuum)
    error = float(np.clip(numerator / (single_yield * nu), 0.0, 0.5))
    return single_yield, error


def decoy_key_rate(observables: DecoyObservables,
                   error_correction_inefficiency: float =
                   C.DEFAULT_ERROR_CORRECTION_INEFFICIENCY
                   ) -> float:
    """GLLP secret key rate per pulse, from the observables alone.

    .. math::

        R \\ge q\\,\\bigl[\\, Q_1^L \\left(1 - h(e_1^U)\\right)
              - f Q_\\mu h(E_\\mu) \\,\\bigr],
        \\qquad Q_1^L = Y_1^L \\mu e^{-\\mu}

    The first term is what the single-photon pulses are worth once privacy
    amplification has removed what Eve could know about them; the second is what
    error correction leaks about *all* detections, multi-photon ones included.
    Multi-photon pulses contribute nothing positive — Eve is assumed to have
    them — so pushing the intensity up buys gain and pays for it twice over past
    the optimum.  That trade is the thing phase 5's agent is actually solving.

    Args:
        observables: one settling period.
        error_correction_inefficiency: the ``f`` above.

    Returns:
        Secret bits per pulse sent at the signal intensity, never negative.
    """
    single_yield, single_error = single_photon_bounds(observables)
    mu = observables.settings.signal_intensity
    single_gain = single_yield * mu * np.exp(-mu)

    secure = single_gain * (1.0 - binary_entropy(single_error))
    leaked = (error_correction_inefficiency * observables.signal_gain
              * binary_entropy(observables.signal_error))
    return float(max(observables.settings.sifting_factor * (secure - leaked), 0.0))


def decoy_key_rate_hz(observables: DecoyObservables,
                      error_correction_inefficiency: float =
                      C.DEFAULT_ERROR_CORRECTION_INEFFICIENCY
                      ) -> float:
    """:func:`decoy_key_rate` in bits per second.

    Only the pulses actually sent at the signal intensity carry key, so the
    clock is multiplied by ``signal_probability``.  Spending more pulses on
    decoys tightens the bound and costs raw rate; that trade is real, and an
    agent that can move the probabilities has to face it.
    """
    settings = observables.settings
    per_pulse = decoy_key_rate(observables, error_correction_inefficiency)
    return per_pulse * settings.clock_rate_hz * settings.signal_probability


# ---------------------------------------------------------------------------
# the receiver's own limits
# ---------------------------------------------------------------------------

def apply_detector_saturation(response: ChannelResponse, settings: DecoySettings,
                              dead_time_s: float,
                              afterpulse_probability: float = 0.0
                              ) -> ChannelResponse:
    """Fold detector dead time and afterpulsing into the channel's yields.

    Without this, the clock rate is a free parameter: key scales with pulses per
    second and the best policy is "turn it up".  Real detectors say otherwise.

    **Dead time.**  After a click a detector is blind for :math:`\\tau_d`.  For a
    non-paralysable detector the measured rate is
    :math:`r_{\\text{meas}} = r / (1 + r \\tau_d)`, so every yield is scaled by
    :math:`1/(1 + r\\tau_d)` with :math:`r` the raw click rate the current clock
    and intensities produce.  At 50 ns and a gigahertz clock, saturation starts
    biting around a gain of 2%, which is a metropolitan span — not an exotic
    corner.

    **Afterpulsing.**  A trapped carrier from a previous click fires the detector
    again on a later gate.  Those clicks are uncorrelated with the bit Alice
    sent, so they behave exactly like dark counts, and their rate is
    proportional to the click rate — which is what makes them the *second*
    reason a faster clock is not free, and the one that shows up as error rather
    than as missing counts.

    Args:
        response: the channel before the receiver's own limits.
        settings: the transmitter configuration; sets the clock and the mean
            gain over the three intensities.
        dead_time_s: detector dead time.
        afterpulse_probability: probability that a click produces an afterpulse
            in a later gate of the same settling period.

    Returns:
        ChannelResponse: with saturation and afterpulsing applied.

    Raises:
        ValueError: on a negative dead time or an afterpulse probability outside
            ``[0, 1]``.

    Note:
        The scaling is applied to the yields and not to the observables, so
        everything downstream — the decoy bound included — sees a channel that
        is merely worse, which is exactly how a saturating detector looks from
        the outside.  Alice and Bob cannot tell saturation from loss without
        varying the clock, and that is a real ambiguity, not a modelling
        shortcut.
    """
    if dead_time_s < 0:
        raise ValueError("dead_time_s cannot be negative")
    if not 0.0 <= afterpulse_probability <= 1.0:
        raise ValueError("afterpulse_probability must be in [0, 1]")

    mean_gain = float(sum(
        probability * (response.gain(intensity) if intensity > 0 else response.yields[0])
        for intensity, probability in zip(settings.intensities, settings.probabilities)))
    click_rate = settings.clock_rate_hz * mean_gain
    factor = 1.0 / (1.0 + click_rate * dead_time_s)

    afterpulse = afterpulse_probability * mean_gain * factor
    live = response.yields * factor
    yields = live + afterpulse * (1.0 - live)

    numerator = (live * response.errors
                 + afterpulse * (1.0 - live) * C.DARK_COUNT_ERROR_RATE)
    errors = np.where(yields > 0, numerator / np.maximum(yields, 1e-300),
                      C.DARK_COUNT_ERROR_RATE)
    return ChannelResponse(yields=np.clip(yields, 0.0, 1.0),
                           errors=np.clip(errors, 0.0, 1.0), label=response.label)


# ---------------------------------------------------------------------------
# finite statistics
# ---------------------------------------------------------------------------

def sample_observables(response: ChannelResponse, settings: DecoySettings,
                       pulses: float,
                       rng: np.random.Generator | int | None = None
                       ) -> DecoyObservables:
    """Draw one settling period's counts, then reduce them to observables.

    Detections are binomial in the pulses sent at each intensity, and errors are
    binomial in the detections.  That is the whole noise model, and it is enough:
    at a gain of 1e-4 and a second of 1 GHz pulses, the signal gain is known to
    a part in a thousand and the vacuum gain to a few percent — which is why
    :func:`single_photon_bounds`, which differences two exponentially amplified
    gains, is much noisier than either input and why a short settling period can
    report no key on a perfectly healthy link.

    Args:
        response: the channel, honest or attacked.
        settings: the transmitter configuration.
        pulses: total pulses in the period, split by
            :attr:`DecoySettings.probabilities`.
        rng: seed or generator.

    Returns:
        DecoyObservables: the estimates, with ``pulses`` recorded.

    Raises:
        ValueError: on a non-positive pulse count.
    """
    if pulses <= 0:
        raise ValueError("pulses must be positive")

    generator = np.random.default_rng(rng)
    estimates: list[tuple[float, float]] = []

    for intensity, probability in zip(settings.intensities, settings.probabilities):
        sent = int(round(pulses * probability))
        if sent <= 0:
            estimates.append((0.0, C.DARK_COUNT_ERROR_RATE))
            continue
        gain = float(response.gain(intensity)) if intensity > 0 else float(response.yields[0])
        error = (float(response.error_rate(intensity)) if intensity > 0
                 else float(response.errors[0]))
        clicks = int(generator.binomial(sent, min(max(gain, 0.0), 1.0)))
        if clicks == 0:
            estimates.append((0.0, C.DARK_COUNT_ERROR_RATE))
            continue
        wrong = int(generator.binomial(clicks, min(max(error, 0.0), 1.0)))
        estimates.append((clicks / sent, wrong / clicks))

    (signal_gain, signal_error), (decoy_gain, decoy_error), (vacuum_gain, vacuum_error) = estimates
    return DecoyObservables(
        signal_gain=signal_gain, decoy_gain=decoy_gain, vacuum_gain=vacuum_gain,
        signal_error=signal_error, decoy_error=decoy_error, vacuum_error=vacuum_error,
        settings=settings, pulses=float(pulses))


def estimate_from_counts(signal_clicks: int, signal_pulses: int,
                         signal_errors: int,
                         decoy_clicks: int, decoy_pulses: int,
                         decoy_errors: int,
                         vacuum_clicks: int, vacuum_pulses: int,
                         settings: DecoySettings) -> DecoyObservables:
    """Build observables from raw counts, as a real receiver would log them.

    Args:
        signal_clicks, signal_pulses, signal_errors: signal-intensity counts.
        decoy_clicks, decoy_pulses, decoy_errors: weak-decoy counts.
        vacuum_clicks, vacuum_pulses: vacuum counts.  No error count: vacuum
            clicks are half wrong by construction and measuring it adds nothing.
        settings: the transmitter configuration in force.

    Returns:
        DecoyObservables: ready for :func:`single_photon_bounds`.

    Raises:
        ValueError: if any click count exceeds its pulse count.
    """
    for clicks, sent, name in ((signal_clicks, signal_pulses, "signal"),
                               (decoy_clicks, decoy_pulses, "decoy"),
                               (vacuum_clicks, vacuum_pulses, "vacuum")):
        if sent <= 0:
            raise ValueError(f"{name}_pulses must be positive")
        if clicks > sent:
            raise ValueError(f"{name}_clicks ({clicks}) exceeds {name}_pulses ({sent})")

    def ratio(numerator: int, denominator: int, fallback: float) -> float:
        return numerator / denominator if denominator > 0 else fallback

    return DecoyObservables(
        signal_gain=ratio(signal_clicks, signal_pulses, 0.0),
        decoy_gain=ratio(decoy_clicks, decoy_pulses, 0.0),
        vacuum_gain=ratio(vacuum_clicks, vacuum_pulses, 0.0),
        signal_error=ratio(signal_errors, signal_clicks, C.DARK_COUNT_ERROR_RATE),
        decoy_error=ratio(decoy_errors, decoy_clicks, C.DARK_COUNT_ERROR_RATE),
        vacuum_error=C.DARK_COUNT_ERROR_RATE,
        settings=settings,
        pulses=float(signal_pulses + decoy_pulses + vacuum_pulses))


# ---------------------------------------------------------------------------
# the operating point, for reference
# ---------------------------------------------------------------------------

def optimal_signal_intensity(response: ChannelResponse, settings: DecoySettings,
                             grid: Sequence[float] | None = None) -> float:
    """The signal intensity that maximises the asymptotic key rate, by search.

    There is no closed form — the optimum depends on the dark count rate, the
    misalignment and the decoy intensity together — so this scans a grid.  It
    exists as the *reference* the phase-5 agent is measured against: an agent
    that cannot beat a static intensity placed here has learned nothing, and one
    that matches it under a *changing* channel has learned exactly the thing
    that a static setting cannot do.

    Args:
        response: the channel.
        settings: the rest of the transmitter configuration.
        grid: intensities to try.  Defaults to 0.05 to 1.0 in 40 steps.

    Returns:
        The best intensity on the grid.

    Note:
        The grid is clipped so the decoy stays strictly below the signal; an
        intensity at or below ``decoy_intensity`` makes the bound vacuous rather
        than merely bad, and a search that wandered there would report a
        spurious zero.
    """
    candidates = np.asarray(grid if grid is not None else np.linspace(0.05, 1.0, 40),
                            dtype=float)
    candidates = candidates[candidates > settings.decoy_intensity * 1.05]
    if candidates.size == 0:
        raise ValueError("no candidate intensity sits above the decoy intensity")

    rates = [decoy_key_rate(observe(response, settings.replace(signal_intensity=float(value))))
             for value in candidates]
    return float(candidates[int(np.argmax(rates))])
