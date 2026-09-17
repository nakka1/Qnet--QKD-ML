"""What classical traffic does to the quantum channel sharing its fibre.

Why this module has to exist before any allocator does
------------------------------------------------------
Without it, "hybrid allocation" is two independent scheduling problems that
happen to be drawn on the same graph: classical flows contend for classical
capacity, quantum sessions contend for quantum capacity, and a policy that
simply runs both at full tilt is optimal. There is nothing to learn.

The coupling is real and it is one-directional. A quantum channel carries at
most a photon per pulse and does nothing to its neighbours. A classical channel
carries milliwatts, and a tiny fraction of that power is scattered *into* the
quantum channel's wavelength band by **spontaneous Raman scattering** in the
silica itself. Those scattered photons are indistinguishable from signal at the
detector: they arrive in the right band, in the right fibre, within the gate.
Half of them land in the wrong basis, so they are errors.

That is what makes the allocation a genuine trade-off. Classical throughput is
bought with QBER, QBER is what decides whether BB84 produces key at all, and
the exchange rate moves with the polarisation state of the fibre: when drift
has already eaten most of the 11% budget, there is no room left to sell.

The model
---------
Spontaneous Raman power reaching the receiver in a filter of width
:math:`\\Delta\\lambda`, for a classical channel launched at :math:`P_0` into a
span of length :math:`L` with attenuation :math:`\\alpha`
(Chapuran et al., *New J. Phys.* 11 105001, 2009; Eraerds et al.,
*New J. Phys.* 12 063027, 2010; Peters et al., *New J. Phys.* 11 045012, 2009):

.. math::

    P_{\\text{fwd}} &= P_0\\, e^{-\\alpha L}\\, \\rho\\, \\Delta\\lambda\\, L \\\\
    P_{\\text{bwd}} &= P_0\\, \\rho\\, \\Delta\\lambda\\,
                       \\frac{1 - e^{-2\\alpha L}}{2\\alpha}

Both are linear in the launched power, which is the property the allocator
actually exploits, and they differ in how they grow with length: forward
scattering is generated along the span and then attenuated together with the
signal, so it peaks and then falls; backward scattering saturates at
:math:`P_0 \\rho \\Delta\\lambda / 2\\alpha` because light scattered far down the
span never makes it back. Counter-propagating classical traffic is therefore
the worse neighbour on a long span and the better one on a short span, which
is a real engineering decision and not a detail — :func:`raman_noise_power`
takes the direction as an argument rather than picking one.

Those photons then compete with the signal:

.. math:: e = \\frac{\\tfrac{1}{2}(N_{\\text{raman}} + N_{\\text{dark}})
          + e_{\\text{opt}} N_{\\text{sig}}}
          {N_{\\text{sig}} + N_{\\text{raman}} + N_{\\text{dark}}}

The factor of one half is the whole reason Raman noise is expensive: a
scattered photon is unpolarised, so it is equally likely to be detected in
either basis outcome and contributes an error half the time.

What is *not* modelled
----------------------
Four-wave mixing and cross-phase modulation also transfer classical power into
the quantum band. Both are strongly suppressed by the wide channel spacing this
module assumes, and both are far more sensitive to the exact channel plan than
Raman scattering is. Including them would mean committing to a specific DWDM
grid, and the conclusions here would then be about that grid. Raman is the
dominant and the grid-independent term, so it is the one modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as C

__all__ = [
    "CoexistenceParameters",
    "raman_noise_power",
    "raman_noise_counts",
    "coexistence_qber",
    "max_classical_power",
    "Direction",
]

Direction = Literal["forward", "backward"]


@dataclass(frozen=True)
class CoexistenceParameters:
    """Hardware and fibre numbers that set the Raman penalty.

    Attributes:
        raman_coefficient_per_km_nm: :math:`\\rho`, the fraction of launched
            power scattered per kilometre per nanometre of filter bandwidth at
            the quantum channel's offset. Strongly dependent on the classical
            and quantum wavelengths — the Raman gain spectrum of silica peaks
            around 13 THz of separation and falls away on either side. The
            default is the order of magnitude reported for a quantum channel in
            the C band a few nanometres from a classical channel; measured
            values across the literature span roughly 1e-10 to 1e-8, which is
            why :func:`coexistence_qber` sweeps well in the report rather than
            being quoted as a single number.
        filter_bandwidth_nm: :math:`\\Delta\\lambda` of the filter in front of
            the quantum detector. Narrower is better and linearly so; 0.1 nm is
            a standard DWDM-grade filter and about as narrow as one can go
            before the signal itself starts being clipped.
        detection_window_s: the gate the detector is open for, per sifted bit.
            Noise counts are proportional to it, so a short gate is the other
            linear knob a designer has.
        detector_efficiency: probability a photon arriving at the detector is
            registered.
        dark_count_rate_hz: detector counts with no light at all.
        optical_error_rate: intrinsic error of the encoder/decoder optics —
            the QBER an ideal, noiseless, perfectly aligned link would still
            show. Distinct from the polarisation drift the allocator is
            watching, and from the Raman noise it is causing.
        attenuation_db_per_km: fibre loss at the classical wavelength, which
            sets how much of the scattered light survives to the receiver.
        wavelength_nm: quantum channel wavelength, for the photon energy.
    """

    raman_coefficient_per_km_nm: float = 4.0e-9
    filter_bandwidth_nm: float = 0.1
    detection_window_s: float = 1.0e-9
    detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY
    dark_count_rate_hz: float = C.DEFAULT_DARK_COUNT_RATE_HZ
    optical_error_rate: float = 0.005
    attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM
    wavelength_nm: float = C.DEFAULT_WAVELENGTH_NM

    def __post_init__(self) -> None:
        if self.raman_coefficient_per_km_nm < 0:
            raise ValueError("raman_coefficient_per_km_nm cannot be negative")
        if self.filter_bandwidth_nm <= 0:
            raise ValueError("filter_bandwidth_nm must be positive")
        if self.detection_window_s <= 0:
            raise ValueError("detection_window_s must be positive")
        if not 0 < self.detector_efficiency <= 1:
            raise ValueError("detector_efficiency must be in (0, 1]")
        if not 0 <= self.optical_error_rate < 0.5:
            raise ValueError("optical_error_rate must be in [0, 0.5)")

    @property
    def photon_energy_j(self) -> float:
        """Energy of one photon at the quantum channel's wavelength."""
        frequency_hz = C.SPEED_OF_LIGHT_VACUUM / (self.wavelength_nm * 1e-9)
        return C.PLANCK * frequency_hz

    @property
    def attenuation_per_km(self) -> float:
        """Loss as a Napierian coefficient, which is what the integrals use."""
        return self.attenuation_db_per_km * np.log(10.0) / 10.0


def raman_noise_power(classical_power_w: ArrayLike,
                      length_km: ArrayLike,
                      parameters: CoexistenceParameters | None = None,
                      direction: Direction = "forward") -> NDArray[np.float64]:
    """Spontaneous Raman power reaching the quantum receiver, in watts.

    Args:
        classical_power_w: power launched into the fibre by the classical
            channels, summed over them. Zero means an idle fibre.
        length_km: span length.
        parameters: fibre and filter numbers.
        direction: ``"forward"`` for classical traffic co-propagating with the
            quantum signal, ``"backward"`` for counter-propagating.

    Returns:
        Noise power in the quantum channel's filter band, same shape as the
        broadcast of the inputs.

    Raises:
        ValueError: on an unknown direction or a negative power.

    Note:
        Linear in the launched power by construction: spontaneous scattering
        has no threshold and no saturation at the powers a telecom channel
        runs at. That linearity is what lets an allocator treat classical
        admission as a continuous dial on QBER.
    """
    parameters = parameters or CoexistenceParameters()
    power = np.asarray(classical_power_w, dtype=float)
    length = np.asarray(length_km, dtype=float)

    if np.any(power < 0):
        raise ValueError("classical_power_w cannot be negative")
    if np.any(length < 0):
        raise ValueError("length_km cannot be negative")

    alpha = parameters.attenuation_per_km
    scale = parameters.raman_coefficient_per_km_nm * parameters.filter_bandwidth_nm

    if direction == "forward":
        # Generated uniformly along the span, then attenuated over what is left
        # of it — the two effects combine to L * exp(-alpha L).
        return power * np.exp(-alpha * length) * scale * length
    if direction == "backward":
        # Scattered light travels back to the transmitter end, so contributions
        # from far down the span are attenuated twice over. The integral
        # saturates: beyond ~1/(2 alpha) km, extra fibre adds no extra noise.
        if alpha <= 0:
            return power * scale * length
        return power * scale * (1.0 - np.exp(-2.0 * alpha * length)) / (2.0 * alpha)

    raise ValueError(f"unknown direction {direction!r}; expected 'forward' or 'backward'")


def raman_noise_counts(classical_power_w: ArrayLike,
                       length_km: ArrayLike,
                       parameters: CoexistenceParameters | None = None,
                       direction: Direction = "forward") -> NDArray[np.float64]:
    """Expected Raman counts per detection gate.

    Converts the scattered power into the quantity the QBER formula wants: how
    many noise photons the detector registers in one gate.

    Args:
        classical_power_w: launched classical power, in watts.
        length_km: span length.
        parameters: fibre, filter and detector numbers.
        direction: co- or counter-propagating classical traffic.

    Returns:
        Mean noise counts per gate.
    """
    parameters = parameters or CoexistenceParameters()
    power = raman_noise_power(classical_power_w, length_km, parameters, direction)
    photon_rate_hz = power / parameters.photon_energy_j
    return (photon_rate_hz * parameters.detection_window_s
            * parameters.detector_efficiency)


def coexistence_qber(signal_counts_per_gate: ArrayLike,
                     classical_power_w: ArrayLike,
                     length_km: ArrayLike,
                     polarisation_qber: ArrayLike = 0.0,
                     parameters: CoexistenceParameters | None = None,
                     direction: Direction = "forward") -> NDArray[np.float64]:
    """Total QBER: alignment error plus unpolarised background.

    The three error sources do not simply add, and treating them as if they did
    overstates the total. Alignment errors (polarisation drift, imperfect
    optics) are a *fraction of the signal*; background counts (Raman, dark) are
    an *absolute rate* that is wrong half the time. They combine through a
    shared denominator — the total number of detections — so a strong signal
    dilutes background noise and a weak one is swamped by it:

    .. math:: e = \\frac{e_{\\text{align}} N_{\\text{sig}}
              + \\tfrac{1}{2} N_{\\text{bg}}}{N_{\\text{sig}} + N_{\\text{bg}}}

    Args:
        signal_counts_per_gate: detected signal counts per gate. For a
            weak-coherent-pulse link this is roughly ``mu * transmittance *
            detector_efficiency``.
        classical_power_w: launched classical power sharing the fibre.
        length_km: span length.
        polarisation_qber: the alignment error from fibre birefringence — what
            :mod:`qnet_autopilot.compensation` spends phase 3 minimising. Added
            to the optics' own error rate.
        parameters: hardware numbers.
        direction: co- or counter-propagating classical traffic.

    Returns:
        QBER in ``[0, 0.5]``. Half, not one: pure background is a coin flip,
        and a link cannot do worse than that without an active adversary.

    Note:
        The ceiling at 0.5 matters for the allocator. QBER does not grow
        without bound as classical power rises — it asymptotes, and the link
        has been useless since roughly 11% anyway. A policy that treats QBER as
        an unbounded cost will over-throttle classical traffic on a link that
        is already dead.
    """
    parameters = parameters or CoexistenceParameters()
    signal = np.asarray(signal_counts_per_gate, dtype=float)

    if np.any(signal < 0):
        raise ValueError("signal_counts_per_gate cannot be negative")

    raman = raman_noise_counts(classical_power_w, length_km, parameters, direction)
    dark = parameters.dark_count_rate_hz * parameters.detection_window_s
    background = raman + dark

    alignment = np.clip(np.asarray(polarisation_qber, dtype=float)
                        + parameters.optical_error_rate, 0.0, 1.0)

    total = signal + background
    with np.errstate(invalid="ignore", divide="ignore"):
        qber = np.where(total > 0,
                        (alignment * signal + 0.5 * background) / np.maximum(total, 1e-300),
                        0.5)
    return np.clip(qber, 0.0, 0.5)


def max_classical_power(signal_counts_per_gate: ArrayLike,
                        length_km: ArrayLike,
                        qber_budget: ArrayLike,
                        polarisation_qber: ArrayLike = 0.0,
                        parameters: CoexistenceParameters | None = None,
                        direction: Direction = "forward") -> NDArray[np.float64]:
    """Invert the QBER model: how much classical power fits under a budget.

    This is the function an allocator actually wants. Given how badly the fibre
    is currently drifting and how much QBER the QKD session can tolerate, it
    returns the classical launch power that exactly exhausts the budget —
    everything below is free, everything above costs key.

    Solving :math:`e = (e_a N_s + \\tfrac{1}{2} N_b) / (N_s + N_b)` for
    :math:`N_b` gives :math:`N_b = N_s (e - e_a) / (\\tfrac{1}{2} - e)`, which
    is then divided by the dark counts' share and converted back to power.

    Args:
        signal_counts_per_gate: detected signal counts per gate.
        length_km: span length.
        qber_budget: the QBER the link is allowed to reach.
        polarisation_qber: current alignment error from drift.
        parameters: hardware numbers.
        direction: co- or counter-propagating classical traffic.

    Returns:
        Launchable classical power in watts. **Zero** where the budget is
        already spent — where drift plus optics plus dark counts alone exceed
        it, no amount of throttling helps and the honest answer is that the
        link cannot carry classical traffic at all right now.

    Note:
        Returns zero rather than a negative power, and that clamp carries the
        engineering meaning: a link whose drift has eaten the budget is not one
        the allocator can trade on. Phase 3's compensator is what gives the
        budget back.
    """
    parameters = parameters or CoexistenceParameters()
    signal = np.asarray(signal_counts_per_gate, dtype=float)
    budget = np.asarray(qber_budget, dtype=float)
    length = np.asarray(length_km, dtype=float)

    alignment = np.clip(np.asarray(polarisation_qber, dtype=float)
                        + parameters.optical_error_rate, 0.0, 1.0)
    dark = parameters.dark_count_rate_hz * parameters.detection_window_s

    headroom = 0.5 - budget
    allowed_background = np.where(
        headroom > 0, signal * (budget - alignment) / np.maximum(headroom, 1e-300),
        np.inf)
    allowed_raman = allowed_background - dark

    # Counts back to power: undo the gate, the efficiency and the geometry.
    per_watt = raman_noise_counts(1.0, length, parameters, direction)
    power = np.where(per_watt > 0, allowed_raman / np.maximum(per_watt, 1e-300), np.inf)
    return np.maximum(np.nan_to_num(power, nan=0.0, posinf=np.inf), 0.0)
