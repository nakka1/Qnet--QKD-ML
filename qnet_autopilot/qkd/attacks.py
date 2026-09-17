"""Attacks, as modifications of the channel's yields — not of Bob's numbers.

An attack model is only useful if it cannot cheat.  The tempting way to write
one is to say "this attack raises the QBER by two points", which produces a
dataset in which the classifier learns the constant the author typed.  Every
attack here instead modifies a :class:`~qnet_autopilot.physics.decoy.ChannelResponse`
— what happens to a pulse carrying *n* photons — and the gains and error rates
Bob observes are *derived* from that afterwards, by the same code path an honest
channel goes through.  An attack that leaves no signature therefore leaves none
in the dataset either, and one whose signature is a second-order consequence of
what Eve does gets that second-order signature and no help.

The four attacks are chosen to span what a monitoring system has to tell apart:

``InterceptResend``
    The textbook attack.  Eve measures in a random basis and resends what she
    found.  Loud: a quarter of the attacked pulses come back wrong.  It is here
    as the easy case, and because the *interesting* question is whether a
    classifier can separate it from an equally large error rise caused by
    thermal drift in the fibre — which it can, but not from the QBER alone.

``PhotonNumberSplitting``
    The attack decoy states exist for.  Eve keeps a photon from every
    multi-photon pulse and blocks single-photon pulses to pay for the extra
    transmission, choosing the compensation so the *signal* gain is exactly
    unchanged.  Error-free, gain-neutral at the signal intensity: nothing in
    :math:`Q_\\mu` or :math:`E_\\mu` moves at all.  The weak decoy, whose pulses
    are mostly single-photon, cannot be compensated at the same time, and that
    inconsistency is the entire signal.

``DetectorBlinding``
    The implementation attack (Lydersen et al., *Nat. Photonics* 4, 686, 2010):
    bright light holds the detectors in linear mode, where Eve controls exactly
    when they click.  It is error-free like PNS but breaks a different
    invariant — clicks stop being limited by the channel's transmittance, and
    empty pulses start producing them.

``BeamSplitting``
    The honest negative control.  Eve taps the light she is entitled to by the
    channel's own loss and forwards the rest through a better fibre.  Nothing
    Bob can measure changes, and this module does not pretend otherwise: the
    attack returns the response it was given.  A classifier that reports it as
    detectable has learned an artefact, which is exactly what a class like this
    is for.

None of these are the natural degradations they have to be told apart from —
thermal drift, a bend or a splice, a detector going noisy.  Those are not
attacks and are not modelled here; they are conditions of the honest channel and
live in :mod:`qnet_autopilot.qkd.session`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..physics import constants as C
from ..physics.decoy import ChannelResponse, DecoySettings, poisson_weights

__all__ = [
    "Attack", "NoAttack", "InterceptResend", "PhotonNumberSplitting",
    "DetectorBlinding", "BeamSplitting", "ATTACKS", "attack_labels",
]


@runtime_checkable
class Attack(Protocol):
    """Anything that turns an honest channel into an attacked one.

    Attributes:
        label: the class name used in datasets and reports.
    """

    label: str

    def apply(self, response: ChannelResponse,
              settings: DecoySettings) -> ChannelResponse:
        """Return the attacked channel.

        Args:
            response: the honest channel at this moment — it already carries
                whatever loss, misalignment and dark counts the link has.
            settings: the transmitter configuration.  Eve knows the protocol and
                the intensities, so an attack may use them; she does not know
                which intensity any individual pulse was sent at, and no attack
                here uses that.
        """

    def with_strength(self, fraction: float) -> "Attack":
        """The same attack dialled to ``fraction`` of its severity.

        ``0`` must be indistinguishable from no attack at all, and ``1`` is the
        attack as configured.  This is what lets an attack ramp in slowly or
        run intermittently — an eavesdropper who knows a monitoring system is
        watching does not switch on at full strength, and a model whose attacks
        only ever arrive as a step teaches a classifier to look for steps.
        """


@dataclass(frozen=True)
class NoAttack:
    """Nobody is listening.  The majority class, and the baseline."""

    label: str = "normal"

    def apply(self, response: ChannelResponse,
              settings: DecoySettings) -> ChannelResponse:
        return response

    def with_strength(self, fraction: float) -> "NoAttack":
        return self


@dataclass(frozen=True)
class InterceptResend:
    """Eve measures a fraction of the pulses in a random basis and resends.

    On an attacked pulse she picks one of the two BB84 bases at random.  Half
    the time she picks Alice's, learns the bit, and resends it faithfully; half
    the time she picks the other one, learns nothing, and resends a state that
    is uncorrelated with Alice's bit, so Bob is wrong half the time.  The error
    on an attacked detection is therefore

    .. math:: e' = \\tfrac12 e + \\tfrac12 \\bigl(\\tfrac12\\bigr) = \\tfrac12 e + \\tfrac14 ,

    where ``e`` is the error the honest channel would have produced.

    The gain is left alone.  Eve resends from her own source, sitting wherever
    she likes on the line, and can trivially choose its intensity so that Bob's
    click rate is what he expected; an attack that also dimmed the link would be
    detectable by the gain alone and would be a worse attack, not a better test.

    Attributes:
        fraction: share of pulses intercepted.  One is the full attack, a
            quarter is the more realistic partial one — an eavesdropper who
            attacks everything is easy to catch, and the whole difficulty of
            this problem lives at small fractions.
        label: class name.
    """

    fraction: float = 0.25
    label: str = "intercept_resend"

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")

    def with_strength(self, fraction: float) -> "InterceptResend":
        return InterceptResend(fraction=self.fraction * float(np.clip(fraction, 0.0, 1.0)),
                               label=self.label)

    def apply(self, response: ChannelResponse,
              settings: DecoySettings) -> ChannelResponse:
        if self.fraction <= 0.0:
            return response
        attacked = 0.5 * response.errors + 0.25
        errors = ((1.0 - self.fraction) * response.errors
                  + self.fraction * attacked)
        # A dark count is wrong half the time whatever Eve does; leave n=0 alone
        # so the vacuum statistics stay clean and the attack has to show up
        # where it physically does.
        errors[0] = response.errors[0]
        return response.replace(errors=np.clip(errors, 0.0, 0.5), label=self.label)


@dataclass(frozen=True)
class PhotonNumberSplitting:
    """Eve blocks single photons and forwards multi-photon pulses she has tapped.

    What she does, per photon number:

    * ``n = 1`` — blocked with probability ``blocked_fraction``.  These are the
      only pulses that can produce secret key, and removing them is the point.
    * ``n >= 2`` — one photon is split off and stored, the rest are forwarded
      through a channel better than the one Alice and Bob believe they have.
      Bob's yield on those rises by whatever factor is needed to keep the
      *signal* gain exactly where it was.
    * ``n = 0`` — untouched.

    Errors are unchanged throughout: Eve never measures a photon Bob also
    receives, so she introduces no disturbance.  That is what makes this attack
    invisible to error monitoring and what decoy states were invented for.

    Attributes:
        blocked_fraction: share of single-photon pulses suppressed.
        compensate: solve the multi-photon boost so the signal gain is preserved
            exactly.  ``False`` leaves the multi-photon yields alone, which is a
            *worse* attack — the gain then drops and the attack is visible in
            :math:`Q_\\mu` — and exists to show what the compensation buys.
        label: class name.

    Note:
        The boost is solved at the signal intensity and cannot simultaneously
        hold at the decoy intensity, because the two intensities weight the
        photon numbers differently.  That is not a limitation of this model; it
        is the security argument.  When the boost required exceeds what physics
        allows (a yield cannot pass one), the compensation is partial and the
        gain drops — an honest outcome, and one that happens on a short span
        where the multi-photon yields are already high.
    """

    blocked_fraction: float = 0.5
    compensate: bool = True
    label: str = "pns"

    def __post_init__(self) -> None:
        if not 0.0 <= self.blocked_fraction <= 1.0:
            raise ValueError("blocked_fraction must be in [0, 1]")

    def with_strength(self, fraction: float) -> "PhotonNumberSplitting":
        return PhotonNumberSplitting(
            blocked_fraction=self.blocked_fraction * float(np.clip(fraction, 0.0, 1.0)),
            compensate=self.compensate, label=self.label)

    def apply(self, response: ChannelResponse,
              settings: DecoySettings) -> ChannelResponse:
        if self.blocked_fraction <= 0.0:
            return response
        yields = response.yields.copy()
        yields[1] = response.yields[1] * (1.0 - self.blocked_fraction)

        if self.compensate and response.max_photons >= 2:
            weights = np.asarray(poisson_weights(settings.signal_intensity,
                                                 response.max_photons)).reshape(-1)
            target = float(weights @ response.yields)
            fixed = float(weights[0] * yields[0] + weights[1] * yields[1])
            multi = float(weights[2:] @ response.yields[2:])
            if multi > 0:
                boost = max((target - fixed) / multi, 0.0)
                yields[2:] = np.minimum(response.yields[2:] * boost, 1.0)

        return response.replace(yields=yields, label=self.label)


@dataclass(frozen=True)
class DetectorBlinding:
    """Eve holds the detectors in linear mode and triggers them on demand.

    A blinded single-photon detector is no longer a single-photon detector: it
    responds to a bright enough classical pulse and ignores everything else.  Eve
    measures each pulse, then sends a tailored bright pulse that makes Bob's
    apparatus produce the outcome she measured.  She learns every bit and the
    error rate stays low.

    Two things break, and they are what makes this detectable at all:

    * **Clicks stop obeying the transmittance.**  Bob's click rate is set by
      Eve's faked states, not by how much light survives the fibre, so the gain
      rises above what the link budget allows.
    * **Empty pulses start clicking.**  Eve cannot tell a vacuum pulse from a
      dim one that she failed to detect, and a faked state sent on a pulse Alice
      left empty produces a click Bob should never have seen.

    Attributes:
        click_probability: probability that a faked state produces a click on a
            pulse carrying at least one photon.  Typically far above the honest
            yield, which is the signature.
        vacuum_leak: extra ``Y_0`` from faked states landing on empty pulses.
        residual_error: error rate Eve's faked states leave behind — low, but
            not zero, because her own detection is imperfect.
        label: class name.
    """

    click_probability: float = 0.05
    vacuum_leak: float = 2.0e-4
    residual_error: float = 0.01
    label: str = "detector_blinding"

    def __post_init__(self) -> None:
        for name, value in (("click_probability", self.click_probability),
                            ("vacuum_leak", self.vacuum_leak),
                            ("residual_error", self.residual_error)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def with_strength(self, fraction: float) -> "DetectorBlinding":
        # The click probability is *absolute*, so scaling it towards zero would
        # not weaken the attack, it would invert it. What is scaled is the part
        # that is hers: the leak, the residual error, and how far the forced
        # click rate sits above whatever the honest channel already produces —
        # which ``apply`` recovers by taking the maximum.
        weight = float(np.clip(fraction, 0.0, 1.0))
        return DetectorBlinding(click_probability=self.click_probability * weight,
                                vacuum_leak=self.vacuum_leak * weight,
                                residual_error=self.residual_error,
                                label=self.label)

    def apply(self, response: ChannelResponse,
              settings: DecoySettings) -> ChannelResponse:
        if self.click_probability <= 0.0 and self.vacuum_leak <= 0.0:
            return response
        yields = response.yields.copy()
        errors = response.errors.copy()

        yields[0] = min(response.yields[0] + self.vacuum_leak, 1.0)
        forced = self.click_probability > response.yields[1:]
        yields[1:] = np.maximum(response.yields[1:], self.click_probability)
        # Only the photon numbers Eve is actually forcing carry her error; at
        # partial strength she does not control the ones the channel was already
        # delivering faster than she is.
        errors[1:] = np.where(forced, self.residual_error, response.errors[1:])
        # The vacuum clicks Eve fakes are as good as her own measurement, so the
        # vacuum error rate falls below the one half that dark counts give —
        # a second, independent tell, and one no honest fault produces.
        honest_vacuum = response.yields[0]
        faked = yields[0] - honest_vacuum
        if yields[0] > 0:
            errors[0] = float((honest_vacuum * C.DARK_COUNT_ERROR_RATE
                               + faked * self.residual_error) / yields[0])

        return ChannelResponse(yields=yields, errors=np.clip(errors, 0.0, 0.5),
                               label=self.label)


@dataclass(frozen=True)
class BeamSplitting:
    """Eve taps the loss she is already entitled to, and changes nothing.

    The channel loses most of the light anyway.  Eve can intercept that loss
    with a beamsplitter and forward the remainder through a fibre better than
    the one Alice and Bob calibrated against, arriving at exactly the yields they
    expect.  There is no disturbance, no gain change and no error change, and
    this class returns the response it was given, relabelled.

    It is in the set on purpose.  A classifier trained on the other classes will
    be asked to say something about this one, and the only correct answer is
    "normal".  Reporting a detection rate above chance here is the clearest
    possible sign that the dataset leaked something the physics did not put
    there — which is why the phase-5 report quotes it separately rather than
    folding it into an accuracy figure.

    Attributes:
        tapped_fraction: the share of the light Eve keeps.  It affects what she
            learns, and nothing Bob can measure; it is recorded for the dataset
            and used nowhere else.
        label: class name.
    """

    tapped_fraction: float = 0.5
    label: str = "beam_splitting"

    def __post_init__(self) -> None:
        if not 0.0 <= self.tapped_fraction <= 1.0:
            raise ValueError("tapped_fraction must be in [0, 1]")

    def with_strength(self, fraction: float) -> "BeamSplitting":
        return BeamSplitting(tapped_fraction=self.tapped_fraction * float(
            np.clip(fraction, 0.0, 1.0)), label=self.label)

    def apply(self, response: ChannelResponse,
              settings: DecoySettings) -> ChannelResponse:
        return response.replace(label=self.label)


#: The attack classes, in the order the report tabulates them.
ATTACKS: tuple[type, ...] = (NoAttack, InterceptResend, PhotonNumberSplitting,
                             DetectorBlinding, BeamSplitting)


def attack_labels() -> tuple[str, ...]:
    """Labels of every attack class, for building a confusion matrix."""
    return tuple(attack().label for attack in ATTACKS)
