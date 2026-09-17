"""Heralded entanglement generation over one span: how often, and how good.

These are closed-form rate models.  They exist because an RL agent cannot
afford to run a discrete-event simulation inside its reward function, and
because a closed form makes the scaling visible — you can see the
:math:`\\eta^2` and the round-trip term, which you cannot see in a stream of
simulated detector clicks.

They are also, unavoidably, **approximations of the protocol SeQUeNCe actually
simulates**.  SeQUeNCe generates entanglement by emitting photons, propagating
them through a lossy channel, and processing detector clicks; the success
probability is an emergent property of that process, not a number anyone typed
in.  Where a formula here disagrees with the simulation, the simulation is
right.  ``validation/`` measures both and reports the gap; treat these
functions as usable only inside the regime where that comparison says they
hold, which the validation report plots explicitly.

Every function documents its scheme and its assumptions, because the factor-of-two
conventions in the repeater literature are genuinely inconsistent between
papers and a rate quoted without its scheme is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as C
from .fiber import FiberParameters, propagation_delay_s, transmittance

__all__ = [
    "arm_efficiency",
    "bsm_success_probability",
    "attempt_success_probability",
    "max_attempt_rate_hz",
    "entanglement_rate_hz",
    "expected_generation_time_s",
    "binary_entropy",
    "bb84_finite_key_length",
    "bb84_finite_key_rate",
    "bb84_secret_key_rate",
    "LinkBudget",
]

Scheme = Literal["single_heralded", "barrett_kok"]


def arm_efficiency(length_km: ArrayLike,
                   attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM,
                   coupling_efficiency: float = C.DEFAULT_COUPLING_EFFICIENCY,
                   detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY,
                   memory_efficiency: float = 1.0) -> NDArray[np.float64]:
    """End-to-end probability that one emitted photon produces a detector click.

    The product of every efficiency along one arm: memory readout, the
    fibre-to-fibre coupling at each end, the fibre itself, and the detector.

    Args:
        length_km: length of *this arm*.  For midpoint heralding that is half
            the span, not the whole span — the commonest factor-of-two error in
            repeater rate estimates.
        attenuation_db_per_km: fibre loss coefficient.
        coupling_efficiency: per-interface coupling; applied twice.
        detector_efficiency: single-photon detection efficiency.
        memory_efficiency: probability the memory emits a photon when excited.

    Returns:
        Efficiency in [0, 1].
    """
    return (memory_efficiency
            * coupling_efficiency ** 2
            * transmittance(length_km, attenuation_db_per_km)
            * detector_efficiency)


def bsm_success_probability(photon_number_resolving: bool = False) -> float:
    """Ceiling on a Bell-state measurement done with linear optics.

    A beamsplitter-and-detectors BSM can distinguish only two of the four Bell
    states, so it succeeds at most half the time even with perfect detectors.
    Photon-number-resolving detectors do not lift that ceiling for the standard
    two-photon interference BSM; they remove a different error (multi-photon
    miscounts), which is modelled elsewhere.
    """
    return C.LINEAR_OPTICS_BSM_CEILING


def attempt_success_probability(length_km: ArrayLike,
                                scheme: Scheme = "single_heralded",
                                attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM,
                                coupling_efficiency: float = C.DEFAULT_COUPLING_EFFICIENCY,
                                detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY,
                                memory_efficiency: float = 1.0) -> NDArray[np.float64]:
    """Probability that one heralding attempt over a span succeeds.

    Both schemes assume **midpoint heralding**: the two end nodes each send a
    photon to a Bell-state measurement station halfway along the span, so each
    photon travels ``length_km / 2``.

    ``single_heralded``
        One round.  Both photons must arrive and the linear-optics BSM must
        succeed: :math:`p = \\tfrac12 \\eta_A \\eta_B`.  Fast, but a single
        click cannot distinguish "both arrived" from "one arrived twice", so
        the heralded state carries a vacuum admixture — the raw fidelity is
        lower than Barrett-Kok's.

    ``barrett_kok``
        Two rounds, both of which must succeed:
        :math:`p = (\\tfrac12 \\eta_A \\eta_B)^2`.  Quadratically rarer, but
        the second round rejects the vacuum term, so the heralded state is much
        closer to a Bell state.  This is what SeQUeNCe's
        ``EntanglementGenerationA`` implements.

    The choice between them is the first real rate-fidelity trade in the stack,
    and it is a decision an agent can be given.

    Args:
        length_km: total span length between the two end nodes.
        scheme: as above.
        attenuation_db_per_km, coupling_efficiency, detector_efficiency,
        memory_efficiency: passed to :func:`arm_efficiency`.

    Returns:
        Per-attempt success probability.

    Raises:
        ValueError: on an unknown scheme.
    """
    half_span = np.asarray(length_km, dtype=float) / 2.0
    eta = arm_efficiency(half_span, attenuation_db_per_km, coupling_efficiency,
                         detector_efficiency, memory_efficiency)
    single_round = bsm_success_probability() * eta * eta

    if scheme == "single_heralded":
        return single_round
    if scheme == "barrett_kok":
        return single_round ** 2
    raise ValueError(f"unknown scheme {scheme!r}; expected 'single_heralded' or 'barrett_kok'")


def max_attempt_rate_hz(length_km: ArrayLike,
                        group_index: float = C.FIBER_GROUP_INDEX,
                        hardware_rate_hz: float = C.DEFAULT_ATTEMPT_RATE_HZ
                        ) -> NDArray[np.float64]:
    """Attempts per second, capped by the herald round trip.

    A node cannot start the next attempt until it learns whether the last one
    worked, and that news travels at the speed of light in fibre from the
    midpoint and back: one full span-length of propagation per attempt.  Past a
    few tens of kilometres this, not the hardware, is what sets the rate.

    Returns:
        ``min(hardware_rate, 1 / round_trip)``.
    """
    round_trip_s = propagation_delay_s(length_km, group_index)
    latency_limit = np.divide(1.0, round_trip_s,
                              out=np.full_like(np.asarray(round_trip_s, dtype=float), np.inf),
                              where=np.asarray(round_trip_s) > 0)
    return np.minimum(latency_limit, hardware_rate_hz)


def entanglement_rate_hz(length_km: ArrayLike,
                         scheme: Scheme = "single_heralded",
                         fiber: FiberParameters | None = None,
                         coupling_efficiency: float = C.DEFAULT_COUPLING_EFFICIENCY,
                         detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY,
                         memory_efficiency: float = 1.0,
                         hardware_rate_hz: float = C.DEFAULT_ATTEMPT_RATE_HZ
                         ) -> NDArray[np.float64]:
    """Heralded pairs per second over one span.

    The product of :func:`max_attempt_rate_hz` and
    :func:`attempt_success_probability` — the classic exponential-in-distance
    curve, bent further down by the round-trip cap.
    """
    fiber = fiber or FiberParameters(length_km=float(np.max(length_km)))
    probability = attempt_success_probability(
        length_km, scheme, fiber.attenuation_db_per_km, coupling_efficiency,
        detector_efficiency, memory_efficiency)
    return max_attempt_rate_hz(length_km, fiber.group_index, hardware_rate_hz) * probability


def expected_generation_time_s(length_km: ArrayLike, **kwargs) -> NDArray[np.float64]:
    """Mean wait for one heralded pair — the reciprocal of the rate.

    This is the number that belongs in a scheduler: it is directly comparable
    with the memory coherence time, and the comparison is the whole game.  When
    the expected generation time on a span approaches the coherence time of the
    memories holding the *other* legs of a path, that path cannot be built.
    """
    rate = entanglement_rate_hz(length_km, **kwargs)
    return np.divide(1.0, rate, out=np.full_like(np.asarray(rate, dtype=float), np.inf),
                     where=np.asarray(rate) > 0)


# ---------------------------------------------------------------------------
# QKD
# ---------------------------------------------------------------------------

def binary_entropy(probability: ArrayLike) -> NDArray[np.float64]:
    """Binary Shannon entropy :math:`h(p)`, in bits, safe at 0 and 1."""
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where((p <= 0) | (p >= 1), 0.0, entropy)


def bb84_secret_key_rate(sifted_rate_hz: ArrayLike, qber: ArrayLike,
                         error_correction_inefficiency: float =
                         C.DEFAULT_ERROR_CORRECTION_INEFFICIENCY
                         ) -> NDArray[np.float64]:
    """Asymptotic BB84 secret key rate.

    .. math:: R = R_{\\text{sift}} \\,\\bigl[\\,1 - f\\,h(e) - h(e)\\,\\bigr]

    The two entropy terms are the two things an eavesdropper costs you: ``f
    h(e)`` is what error correction leaks over the public channel, ``h(e)`` is
    what privacy amplification must remove to cover what Eve might know.  The
    rate hits zero near :math:`e = 11\\%` (:data:`~qnet_autopilot.physics.constants.BB84_QBER_THRESHOLD`);
    this function clamps at zero rather than returning a negative key rate.

    This is the asymptotic, one-way-reconciliation figure.  Finite-key effects
    reduce it, sometimes sharply for short blocks, and are not modelled here.

    Args:
        sifted_rate_hz: sifted key rate before error correction.
        qber: quantum bit error rate — for a polarisation-encoded link this is
            what :func:`qnet_autopilot.physics.fiber.stokes_qber` produces,
            plus detector noise.
        error_correction_inefficiency: the ``f`` above.

    Returns:
        Secret bits per second, never negative.
    """
    entropy = binary_entropy(qber)
    fraction = 1.0 - error_correction_inefficiency * entropy - entropy
    return np.maximum(np.asarray(sifted_rate_hz, dtype=float) * fraction, 0.0)


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

#: Candidate shares of the block spent estimating the error rate. The optimum
#: moves with the block size — a short block cannot afford to give much away,
#: a long one cannot afford a loose bound — so it is searched rather than fixed.
_TEST_FRACTIONS = np.concatenate([np.logspace(-5, -1, 33), np.linspace(0.1, 0.5, 17)])


def bb84_finite_key_length(block_bits: ArrayLike, qber: ArrayLike,
                           error_correction_inefficiency: float =
                           C.DEFAULT_ERROR_CORRECTION_INEFFICIENCY,
                           epsilon_secrecy: float = 1e-10,
                           epsilon_correctness: float = 1e-15,
                           test_fraction: float | None = None
                           ) -> NDArray[np.float64]:
    """Secret bits extractable from a **finite** sifted block.

    :func:`bb84_secret_key_rate` is the asymptotic rate, and every key figure in
    phases 3 and 4 was quoted from it. It is a ceiling, and for short blocks a
    generous one: a real protocol pays for not knowing the true error rate, only
    an estimate from a finite sample, and for the finite-size cost of privacy
    amplification.

    The bound follows the standard composable finite-key analysis for BB84
    (Tomamichel, Lim, Gisin & Renner, *Nat. Commun.* 3 634, 2012):

    .. math::

        \ell \le n\bigl[1 - h(e + \mu)\bigr] - \mathrm{leak}_{EC}
                 - \log_2\frac{2}{\varepsilon_{\text{cor}}}
                 - 2\log_2\frac{1}{2\varepsilon_{\text{sec}}}

    with :math:`n` the bits kept for key, :math:`k` the bits spent estimating
    the error, and :math:`\mu` the statistical correction that turns an
    observed error rate into a high-confidence bound on the phase error rate:

    .. math:: \mu = \sqrt{\frac{(n + k)(k + 1)}{n k^2}
              \ln\frac{1}{\varepsilon_{\text{sec}}}}

    Args:
        block_bits: sifted bits in the block, before splitting into key and
            test portions.
        qber: the error rate observed on the test portion.
        error_correction_inefficiency: the ``f`` in ``leak_EC = f n h(e)``.
        epsilon_secrecy: secrecy parameter.
        epsilon_correctness: correctness parameter of error correction.
        test_fraction: share of the block spent on error estimation. ``None``
            optimises it, which is what a real protocol designer does and what
            makes the bound converge to the asymptotic rate.

    Returns:
        Secret bits, never negative. Zero for a block too short to pay the
        finite-size costs — the honest answer, and one the asymptotic formula
        cannot give.

    Note:
        Fixing the test fraction gets the limit wrong, and quietly. Hold it at
        10% and the bound converges to 90% of the asymptotic rate rather than to
        it, because the asymptotic formula assumes parameter estimation is free
        — true only if the fraction shrinks as the block grows. Searching the
        fraction reproduces the asymptotic rate to a fraction of a percent at
        10\ :sup:`10` bits, which is the check that the bound is right.

        The finite-size penalty is what makes a *short* block worthless even at
        a QBER the asymptotic formula would happily produce key from. A link
        making a few thousand sifted bits per second and settling up every
        second is exactly there, so this is not academic for phases 3 and 4.
    """
    block = np.asarray(block_bits, dtype=float)
    error = np.clip(np.asarray(qber, dtype=float), 0.0, 1.0)

    for name, value in (("epsilon_secrecy", epsilon_secrecy),
                        ("epsilon_correctness", epsilon_correctness)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")

    if test_fraction is None:
        fractions = _TEST_FRACTIONS
    else:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")
        fractions = np.array([test_fraction])

    shape = np.broadcast(block, error).shape
    best = np.zeros(shape)

    for fraction in fractions:
        test = np.maximum(block * fraction, 0.0)
        keep = np.maximum(block - test, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = np.sqrt(
                (keep + test) * (test + 1.0)
                / np.maximum(keep * test * test, 1e-300)
                * np.log(1.0 / epsilon_secrecy))
            # Clipped at one half, not at one. The binary entropy is
            # symmetric about 0.5, so clipping a diverging bound at 1.0 sends
            # h back to zero and the formula reports *more* key the worse the
            # bound gets — which is how a finite-key bound ends up above the
            # asymptotic rate it is supposed to sit under.
            phase_error = np.clip(error + deviation, 0.0, 0.5)
            leak = error_correction_inefficiency * keep * binary_entropy(error)
            length = (keep * (1.0 - binary_entropy(phase_error)) - leak
                      - np.log2(2.0 / epsilon_correctness)
                      - 2.0 * np.log2(1.0 / (2.0 * epsilon_secrecy)))
        best = np.maximum(best, np.where(np.isfinite(length), length, 0.0))

    return best


def bb84_finite_key_rate(sifted_rate_hz: ArrayLike, qber: ArrayLike,
                         block_duration_s: ArrayLike = 1.0,
                         **kwargs) -> NDArray[np.float64]:
    """Finite-key secret rate, from a sifted rate and how long a block runs.

    A convenience over :func:`bb84_finite_key_length` for the common case of
    settling up on a fixed schedule.

    Args:
        sifted_rate_hz: sifted bits per second.
        qber: observed error rate.
        block_duration_s: how long sifted bits accumulate before a block is
            processed. Longer blocks pay the finite-size costs once over more
            bits and therefore lose less, at the price of latency.
        **kwargs: passed to :func:`bb84_finite_key_length`.

    Returns:
        Secret bits per second.
    """
    duration = np.asarray(block_duration_s, dtype=float)
    block = np.asarray(sifted_rate_hz, dtype=float) * duration
    return bb84_finite_key_length(block, qber, **kwargs) / np.maximum(duration, 1e-30)


@dataclass(frozen=True)
class LinkBudget:
    """All the per-span numbers together, so a scheduler can read them off once.

    Attributes:
        length_km: span length.
        scheme: heralding scheme assumed.
        arm_efficiency: per-arm photon survival probability.
        attempt_probability: per-attempt success probability.
        attempt_rate_hz: attempts per second after the round-trip cap.
        rate_hz: heralded pairs per second.
        expected_time_s: mean wait per pair.
        delay_s: one-way propagation delay over the span.
    """

    length_km: float
    scheme: Scheme
    arm_efficiency: float
    attempt_probability: float
    attempt_rate_hz: float
    rate_hz: float
    expected_time_s: float
    delay_s: float

    @classmethod
    def compute(cls, length_km: float, scheme: Scheme = "single_heralded",
                fiber: FiberParameters | None = None,
                coupling_efficiency: float = C.DEFAULT_COUPLING_EFFICIENCY,
                detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY,
                memory_efficiency: float = 1.0,
                hardware_rate_hz: float = C.DEFAULT_ATTEMPT_RATE_HZ) -> "LinkBudget":
        """Compute the whole budget for one span."""
        fiber = fiber or FiberParameters(length_km=length_km)
        eta = float(arm_efficiency(length_km / 2.0, fiber.attenuation_db_per_km,
                                   coupling_efficiency, detector_efficiency,
                                   memory_efficiency))
        probability = float(attempt_success_probability(
            length_km, scheme, fiber.attenuation_db_per_km, coupling_efficiency,
            detector_efficiency, memory_efficiency))
        attempt_rate = float(max_attempt_rate_hz(length_km, fiber.group_index, hardware_rate_hz))
        rate = attempt_rate * probability
        return cls(
            length_km=length_km,
            scheme=scheme,
            arm_efficiency=eta,
            attempt_probability=probability,
            attempt_rate_hz=attempt_rate,
            rate_hz=rate,
            expected_time_s=(1.0 / rate if rate > 0 else float("inf")),
            delay_s=float(propagation_delay_s(length_km, fiber.group_index)),
        )
