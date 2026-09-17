"""Measured numbers from deployed fibre, with the source of each one.

Every other parameter in this package is either a transcription of SeQUeNCe or a
plausible operating point chosen from the literature and swept. This module is
different: it holds figures that somebody measured on fibre that is in the
ground or on poles, and it names where each came from. That is the whole point
of it — a constant without a citation is a guess with a decimal point.

What the search actually found
------------------------------
**No public dataset of polarisation on deployed terrestrial fibre exists.** The
raw state-of-polarisation time series that can be downloaded are all submarine
and all sampled at 20 Hz or below — enough for thermal drift and for jump
statistics, useless for anything in the acoustic band. The three best
characterisations of *terrestrial* links (ORNL's 15 km aerial/in-ground, the
NIST-UMD 62 km aerial link, and a 50 km mixed link in Los Angeles) publish the
statistics and withhold the data. So this module records the statistics, and
:mod:`qnet_autopilot.datagen.measured` provides the path that consumes a real
time series if one is obtained.

Two findings that change a result
---------------------------------
**The acoustic band is at the wrong frequency.** Phase 3 puts it at 40 Hz.
Nothing measured supports that. The only 4 kHz polarimeter recording on aerial
fibre that has been published (Peterson et al., 220 km of optical-ground-wire)
reports a strong tone at the **power-line frequency with its second and third
harmonics** — 60, 120 and 180 Hz in North America, 50/100/150 Hz on a 50 Hz grid
— because the fibre is strung inside the ground wire of a transmission line and
the current in that conductor is what moves it.

**The drift rate is too fast, and the spectrum says by how much.** ORNL reports
a drift power spectrum going as :math:`f^{-1.0 \pm 0.2}`. That exponent is a
sharp function of the drift rate in this model — the state of polarisation wraps
around the sphere, so drifting faster whitens the spectrum — and matching it
pins the rate to **0.015-0.020 rad/s**, against the 0.05 rad/s phase 3 assumed.
The exponent runs 0.25, 0.30, 0.88, 1.18, 1.66 as the rate falls 0.05, 0.03,
0.02, 0.015, 0.010 — a factor of five in rate covers the whole plausible range
of the exponent, so this is a tight constraint and not a coincidence.
:data:`AERIAL_OPGW` is the resulting operating point and
``scripts/evaluate_field.py`` re-runs the phase-3 comparison there.

A shape that did **not** need fixing: a superposition of Ornstein-Uhlenbeck
processes was tried first, on the theory that one relaxation time cannot make a
straight spectrum. It gives a *steeper* exponent than a single process at the
same rate, and is not needed — a single OU at the calibrated rate reproduces the
measured exponent within its uncertainty. :func:`flicker_components` is kept
because multi-timescale drift is a real thing to be able to express, not because
this calibration wanted it.

Sources
-------
Czegledi, Karlsson, Agrell & Johannisson, "Polarization drift channel model for
coherent fibre-optic systems", *Sci. Rep.* **6**, 21217 (2016). 127 km of buried
fibre, Jones matrices every ~2.2 h for 36 days; fitted polarisation linewidth
:math:`\\Delta_p` = 60 nHz in Stokes space, 80 nHz in Jones space.

Peterson, Cyr, Nyman *et al.* (NIST), field polarisation dynamics on deployed
fibre. Site A: 110 km direct-buried beside railroad tracks, three spans, 25 ps
PMD, PDL < 0.5 dB. Site B: 220 km aerial optical-ground-wire inside a 160 kV
transmission line. Stokes polarimeter at 4 kHz, 9 h continuous; polarisation
excursions per 1 ms bin are Rayleigh-distributed, and the aerial link carries a
60 Hz tone with 120 and 180 Hz harmonics tracing circles on the Poincare sphere.
https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=31486

Ramaswamy, Rao, Chapman & Alshowkan (ORNL), "Analysis of polarization drift of
optical signals over deployed aerial-inground fiber connections", arXiv
2607.07629. 15 km hybrid aerial/in-ground armoured ribbon cable, 1550 nm,
11 months (Dec 2024 - Nov 2025); drift PSD is flicker-like with
:math:`\\beta = 1.0 \\pm 0.2`, spectral centroid (3.2 ± 0.9) mHz, spectral
variance (1270 ± 20) mHz², strong diurnal and seasonal cycles. Data on request.

Shi, Su, Rahmouni *et al.* (NIST/UMD), "Entanglement distribution over a
polarization-stabilized aerial fiber", arXiv 2601.11753. 62 km, ~70% aerial,
18 dB fibre loss. At night the output polarisation is stable for tens of
minutes; in the day it moves by tens of degrees within seconds, and the fidelity
to a reference state falls below 95% in under 20 s unstabilised.

Kumar, Qin & Alleaume, "Coexistence of continuous variable QKD with intense DWDM
classical channels", *New J. Phys.* **17**, 043027 (2015). Measured spontaneous
Raman coefficient :math:`\\beta` = 1.5e-9 to 3.1e-9 km⁻¹ nm⁻¹ across the C-band,
25 km of standard single-mode fibre.
https://ar5iv.arxiv.org/html/1412.1403

Dynes, Wonfor, Tam *et al.*, "Cambridge quantum network", *npj Quantum Inf.*
**5**, 101 (2019). Three metro links, 5.0-10.6 km, 0.43 dB/km, 580 days
continuous; QBER 2.4-2.7%, mean secret key rate 2.2-2.6 Mbit/s. A 100 Gbit/s
DWDM channel on the same fibre raised the QBER from 2.4% to 2.7%. The data
behind the figures is deposited under CC-BY at
https://doi.org/10.17863/CAM.44524
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fiber import PolarizationDriftModel

__all__ = [
    "FieldMeasurement", "MEASUREMENTS",
    "POLARISATION_LINEWIDTH_HZ", "RAMAN_COEFFICIENT_RANGE_PER_KM_NM",
    "AERIAL_DAYTIME_DRIFT_RATE_RAD_PER_S", "AERIAL_NIGHT_DRIFT_RATE_RAD_PER_S",
    "FLICKER_EXPONENT", "SPECTRAL_CENTROID_HZ", "SPECTRAL_FIT_BAND_HZ",
    "CALIBRATED_DRIFT_RATE_RAD_PER_S", "CALIBRATED_DRIFT_RATE_RANGE_RAD_PER_S",
    "linewidth_to_drift_rate", "flicker_components", "spectral_exponent",
    "BURIED_CZEGLEDI", "AERIAL_OPGW", "AERIAL_HYBRID_ORNL",
]


@dataclass(frozen=True)
class FieldMeasurement:
    """One measured quantity and where it came from.

    Attributes:
        quantity: what was measured, in words.
        value: the figure, in the units named.
        units: the units.
        link: the fibre it was measured on.
        source: author, venue, year.
        url: where to check it.
        calibrates: which model parameter this constrains.
    """

    quantity: str
    value: str
    units: str
    link: str
    source: str
    url: str
    calibrates: str


#: Every figure this module rests on, in one list, so a reader can audit the
#: provenance of the presets below without reading the code.
MEASUREMENTS: tuple[FieldMeasurement, ...] = (
    FieldMeasurement(
        quantity="polarisation linewidth", value="60 (Stokes), 80 (Jones)",
        units="nHz", link="127 km buried, 36 days at ~2.2 h intervals",
        source="Czegledi et al., Sci. Rep. 6:21217 (2016)",
        url="https://www.nature.com/articles/srep21217",
        calibrates="thermal drift rate on buried fibre"),
    FieldMeasurement(
        quantity="narrowband content", value="60 Hz tone + 120 and 180 Hz harmonics",
        units="Hz", link="220 km aerial optical-ground-wire, 160 kV line, 4 kHz polarimeter",
        source="Peterson et al. (NIST)",
        url="https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=31486",
        calibrates="vibration frequency and harmonic structure on aerial fibre"),
    FieldMeasurement(
        quantity="excursion distribution", value="Rayleigh, per 1 ms bin",
        units="rad", link="110 km buried and 220 km aerial, 9 h at 4 kHz",
        source="Peterson et al. (NIST)",
        url="https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=31486",
        calibrates="the shape of the fast increment distribution"),
    FieldMeasurement(
        quantity="drift PSD exponent", value="1.0 ± 0.2",
        units="dimensionless (f^-beta)",
        link="15 km hybrid aerial/in-ground, 11 months",
        source="Ramaswamy et al. (ORNL), arXiv 2607.07629",
        url="https://arxiv.org/html/2607.07629v1",
        calibrates="the spectral shape of the slow drift"),
    FieldMeasurement(
        quantity="spectral centroid", value="3.2 ± 0.9", units="mHz",
        link="15 km hybrid aerial/in-ground, 11 months",
        source="Ramaswamy et al. (ORNL), arXiv 2607.07629",
        url="https://arxiv.org/html/2607.07629v1",
        calibrates="where the slow drift puts its power"),
    FieldMeasurement(
        quantity="daytime drift", value="tens of degrees in seconds; fidelity < 95% in < 20 s",
        units="rad/s", link="62 km, ~70% aerial, 18 dB",
        source="Shi et al. (NIST/UMD), arXiv 2601.11753",
        url="https://arxiv.org/html/2601.11753",
        calibrates="the magnitude of the drift rate on aerial fibre"),
    FieldMeasurement(
        quantity="spontaneous Raman coefficient", value="1.5e-9 to 3.1e-9",
        units="1/km/nm", link="25 km standard single-mode, C-band",
        source="Kumar, Qin & Alleaume, New J. Phys. 17:043027 (2015)",
        url="https://ar5iv.arxiv.org/html/1412.1403",
        calibrates="the Raman noise term in the coexistence model"),
    FieldMeasurement(
        quantity="coexistence QBER penalty", value="2.4% -> 2.7% with 100 Gbit/s DWDM",
        units="QBER", link="three metro links 5.0-10.6 km, 580 days",
        source="Dynes et al., npj Quantum Inf. 5:101 (2019)",
        url="https://doi.org/10.17863/CAM.44524",
        calibrates="an upper bound on the coexistence penalty at metro distance"),
)

#: Fitted polarisation linewidth on 127 km of buried fibre, in Hz. The Stokes
#: and Jones estimates differ because the measured Jones matrices are not quite
#: unitary; the pair is the honest uncertainty.
POLARISATION_LINEWIDTH_HZ: tuple[float, float] = (60e-9, 80e-9)

#: Measured spontaneous Raman coefficient, 1/km/nm. The package default of
#: 4.0e-9 sits above this range, so every coexistence figure in phase 4 is
#: pessimistic by 1.3 to 2.7 times rather than optimistic.
RAMAN_COEFFICIENT_RANGE_PER_KM_NM: tuple[float, float] = (1.5e-9, 3.1e-9)

#: "Tens of degrees within a matter of seconds" on a 70%-aerial link, read as
#: 20-40 degrees over 1-3 s. Quoted as a range because that is what the source
#: gives; anyone wanting a single number should say which end they took.
AERIAL_DAYTIME_DRIFT_RATE_RAD_PER_S: tuple[float, float] = (0.12, 0.70)

#: "Stable for over tens of minutes" at night, read as the same excursion over
#: 20-40 minutes.
AERIAL_NIGHT_DRIFT_RATE_RAD_PER_S: tuple[float, float] = (1.5e-4, 6.0e-4)

#: The exponent of the measured drift spectrum, and its uncertainty.
FLICKER_EXPONENT: tuple[float, float] = (1.0, 0.2)

#: Drift rate at which this model reproduces :data:`FLICKER_EXPONENT`, with the
#: 60 s correlation time phase 3 uses, measured over ORNL's own fitting band by
#: ``scripts/evaluate_field.py``. The scan puts the whole interval consistent
#: with the measurement at 0.015-0.020 rad/s; the upper end is taken, because a
#: faster-drifting fibre is the harder one to compensate and a calibration
#: should not quietly make the problem easier. Phase 3 ran at 0.05 rad/s, which
#: whitens the spectrum to an exponent near 0.25 — a fibre drifting so fast that
#: it has forgotten where it was within the band the measurement covers.
CALIBRATED_DRIFT_RATE_RAD_PER_S: float = 0.02

#: The full interval of drift rates whose spectral exponent lands inside the
#: measured uncertainty. Quoted so nobody reads the single value above as
#: sharper than it is.
CALIBRATED_DRIFT_RATE_RANGE_RAD_PER_S: tuple[float, float] = (0.015, 0.020)

#: The band ORNL fitted the exponent over, in Hz. Fitting outside it means
#: fitting through the flat part of a Lorentzian, which reports a slope the
#: process does not have.
SPECTRAL_FIT_BAND_HZ: tuple[float, float] = (3e-4, 1e-2)

#: Spectral centroid of the measured drift, Hz.
SPECTRAL_CENTROID_HZ: float = 3.2e-3


def linewidth_to_drift_rate(linewidth_hz: float, dt_s: float) -> float:
    """Convert a polarisation linewidth into this model's drift-rate parameter.

    Czegledi et al. model drift as an isotropic random walk whose per-axis step
    over an interval :math:`T` is drawn from :math:`N(0, 2\\pi\\Delta_p T)`. This
    package instead integrates an angular velocity, so the two meet at the step:
    a velocity :math:`\\sigma` held for :math:`T` produces a step of standard
    deviation :math:`\\sigma T`, hence
    :math:`\\sigma = \\sqrt{2\\pi\\Delta_p / T}`.

    Args:
        linewidth_hz: :math:`\\Delta_p`, in Hz.
        dt_s: the interval the rate will be applied over. **The answer depends
            on it**, and that dependence is the physics, not a defect: a random
            walk has no characteristic velocity, so "how fast does it drift"
            only has an answer once you say over what interval you are asking.

    Returns:
        Per-axis RMS angular velocity, rad/s.

    Raises:
        ValueError: on a non-positive linewidth or interval.

    Note:
        Sampled at 1 ms, 60 nHz gives 0.02 rad/s; sampled at 1 s it gives
        6e-4 rad/s. Both describe the same fibre. Quote the interval whenever
        quoting the rate.
    """
    if linewidth_hz <= 0:
        raise ValueError("linewidth_hz must be positive")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    return float(np.sqrt(2.0 * np.pi * linewidth_hz / dt_s))


def flicker_components(rate_rad_per_s: float,
                       low_frequency_hz: float = 1e-4,
                       high_frequency_hz: float = 1.0,
                       num_components: int = 5
                       ) -> tuple[tuple[float, float], ...]:
    """Correlation times and weights whose OU sum approximates 1/f noise.

    A single Ornstein-Uhlenbeck process has a Lorentzian spectrum: flat below
    its corner, :math:`f^{-2}` above. Deployed fibre does not do that — the
    measured exponent is :math:`1.0 \\pm 0.2` across four decades. Superposing
    Lorentzians with logarithmically spaced corners and equal power per decade
    is the standard construction that does, and it is what this returns.

    Args:
        rate_rad_per_s: total RMS angular velocity the components should sum to.
        low_frequency_hz: bottom of the band the 1/f shape should hold over.
        high_frequency_hz: top of that band.
        num_components: how many processes. Five over four decades keeps the
            residual ripple in the spectrum under about 1 dB, which is well
            inside the ±0.2 uncertainty on the measured exponent.

    Returns:
        ``(correlation_time_s, rate_rad_per_s)`` pairs for
        :attr:`~qnet_autopilot.physics.fiber.PolarizationDriftModel.thermal_components`.

    Raises:
        ValueError: if the band is empty or fewer than two components are asked
            for — one component is not an approximation of 1/f, it is the
            Lorentzian this exists to replace.
    """
    if not 0 < low_frequency_hz < high_frequency_hz:
        raise ValueError("need 0 < low_frequency_hz < high_frequency_hz")
    if num_components < 2:
        raise ValueError("num_components must be at least 2")
    if rate_rad_per_s < 0:
        raise ValueError("rate_rad_per_s cannot be negative")

    corners = np.logspace(np.log10(low_frequency_hz),
                          np.log10(high_frequency_hz), num_components)
    # Equal variance per component: the sum then has equal power per decade,
    # which is what 1/f means. Total variance is preserved, so the caller's
    # rate stays the rate.
    per_component = rate_rad_per_s / np.sqrt(num_components)
    return tuple((float(1.0 / (2.0 * np.pi * corner)), float(per_component))
                 for corner in corners)


def _buried_czegledi(dt_s: float = 1e-3) -> PolarizationDriftModel:
    """Buried fibre, drift rate from the measured polarisation linewidth."""
    linewidth = float(np.mean(POLARISATION_LINEWIDTH_HZ))
    return PolarizationDriftModel(
        thermal_correlation_time_s=1.0 / (2.0 * np.pi * SPECTRAL_CENTROID_HZ),
        thermal_drift_rate_rad_per_s=linewidth_to_drift_rate(linewidth, dt_s),
        thermal_per_kelvin_rad_per_s=0.0,
        vibration_amplitude_rad=0.0,
        jump_rate_per_s=0.0,
    )


#: Buried fibre as Czegledi et al. measured it: a pure random walk, no acoustic
#: band, no jumps. Quiet enough that compensation is barely a problem, which is
#: itself worth knowing — the interesting case is aerial.
BURIED_CZEGLEDI = _buried_czegledi()

#: Aerial optical-ground-wire, as close to the two field measurements as this
#: model can be put: the power-line fundamental with its second and third
#: harmonics (NIST), and the drift rate that reproduces the measured spectral
#: exponent (ORNL). The vibration *amplitude* is the one number here with no
#: measurement behind it — the NIST report shows the tone but never states its
#: size in radians — so it keeps the phase-3 value and is swept rather than
#: trusted.
AERIAL_OPGW = PolarizationDriftModel(
    thermal_correlation_time_s=60.0,
    thermal_drift_rate_rad_per_s=CALIBRATED_DRIFT_RATE_RAD_PER_S,
    thermal_per_kelvin_rad_per_s=0.2,
    vibration_frequency_hz=60.0,
    vibration_bandwidth_hz=6.0,
    vibration_amplitude_rad=0.30,
    vibration_harmonics=(1.0, 0.5, 0.25),
    vibration_num_tones=8,
    jump_rate_per_s=0.05,
    jump_scale_rad=0.35,
)

#: The ORNL link: hybrid aerial and in-ground, flicker-shaped drift over four
#: decades, no resolved acoustic band (their sampling stops at 1.56 Hz, so the
#: absence is a limit of the measurement and not a statement about the fibre).
AERIAL_HYBRID_ORNL = PolarizationDriftModel(
    thermal_components=flicker_components(
        CALIBRATED_DRIFT_RATE_RAD_PER_S, 1e-4, 1.0, 5),
    thermal_per_kelvin_rad_per_s=0.2,
    vibration_amplitude_rad=0.0,
    jump_rate_per_s=0.05,
    jump_scale_rad=0.35,
)


def spectral_exponent(trajectory_angle_rad: NDArray[np.float64], dt_s: float,
                      low_frequency_hz: float = 1e-4,
                      high_frequency_hz: float = 1.0) -> float:
    """Fit :math:`\\beta` in :math:`S(f) \\propto f^{-\\beta}` over a band.

    The one number that says whether a drift model has the right *shape* rather
    than merely the right size. ORNL measured 1.0 ± 0.2; a single OU process
    fitted over the same band returns something closer to 2.

    Args:
        trajectory_angle_rad: a scalar angle series (for example the angle
            between the launched and received Stokes vectors).
        dt_s: sampling interval.
        low_frequency_hz, high_frequency_hz: the fitting band. Points outside it
            are ignored, which matters: an OU spectrum is flat below its corner
            and fitting through the flat part reports a shallower slope than the
            process has.

    Returns:
        The exponent. Positive means power falls with frequency.

    Raises:
        ValueError: if fewer than eight spectral points fall inside the band —
            a slope through seven points spanning less than a decade is not a
            measurement.
    """
    angle = np.asarray(trajectory_angle_rad, dtype=float)
    spectrum = np.abs(np.fft.rfft(angle - angle.mean())) ** 2
    frequency = np.fft.rfftfreq(angle.size, dt_s)

    inside = ((frequency >= low_frequency_hz) & (frequency <= high_frequency_hz)
              & (spectrum > 0))
    if int(inside.sum()) < 8:
        raise ValueError(
            f"only {int(inside.sum())} spectral points in "
            f"[{low_frequency_hz}, {high_frequency_hz}] Hz; need at least 8")

    slope, _ = np.polyfit(np.log10(frequency[inside]),
                          np.log10(spectrum[inside]), 1)
    return float(-slope)
