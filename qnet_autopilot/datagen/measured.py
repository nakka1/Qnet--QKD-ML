"""Fit the drift model to a real polarimeter recording.

Everything else in this package generates data. This module consumes it: give it
a state-of-polarisation time series measured on real fibre and it returns the
:class:`~qnet_autopilot.physics.fiber.PolarizationDriftModel` that reproduces it,
so phases 3 and 4 can be re-run against a link that exists rather than one that
was assumed.

This is the piece that closes the last gap, and it is deliberately separate from
the constants in :mod:`qnet_autopilot.physics.field_calibration`. Those record
what other people measured and published; this runs on data you have in hand.

Where to get the data
---------------------
No public recording of polarisation on deployed *terrestrial* fibre exists at
the time of writing. What can be downloaded is submarine and sampled at 20 Hz or
below — which is enough for the drift terms and the jump statistics, and cannot
touch the acoustic band:

* Egersund-Yme, ~240 km subsea, 5 Hz, June 2023 - April 2024, CC-BY:
  https://osf.io/q5e2n/
* Med-Nautilus Catania-Tel Aviv, ~2500 km, 3 Stokes parameters aggregated to
  5 Hz, 2.5 years, CC-BY: https://zenodo.org/records/18774527
* EllaLink transatlantic SOP-OTDR, Jan-Apr 2025, CC-BY:
  https://zenodo.org/records/22085542
* Curie, Los Angeles - Valparaiso, HDF5, CC-BY:
  https://data.caltech.edu/records/50509-xhf30

Any of these lands as a table of time and Stokes parameters, which is what
:func:`load_stokes` reads.

What the fit can and cannot recover
-----------------------------------
The three drift terms live in different parts of the spectrum, and a recording
only constrains the ones it resolves. :func:`fit_drift_model` says which those
are rather than returning a number for all of them regardless: a 5 Hz recording
cannot see a 60 Hz tone, and reporting a vibration amplitude fitted from
aliased power would be worse than reporting none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.fiber import PolarizationDriftModel

__all__ = ["StokesRecording", "DriftFit", "load_stokes", "fit_drift_model"]

#: A jump is an increment this many robust standard deviations above the
#: typical one. Six sigma of a median-absolute-deviation estimate is far
#: enough into the tail that thermal drift does not reach it, and close
#: enough that a 0.35 rad jump on a link drifting at 0.02 rad/s does.
JUMP_THRESHOLD_SIGMA = 6.0


@dataclass
class StokesRecording:
    """A measured polarisation history.

    Attributes:
        time_s: sample times, seconds from the start of the recording.
        stokes: ``(n, 3)`` normalised Stokes vectors.
        source: where the data came from, for provenance.
    """

    time_s: NDArray[np.float64]
    stokes: NDArray[np.float64]
    source: str = "unknown"

    @property
    def dt_s(self) -> float:
        """Median sampling interval.

        Median rather than mean: real recordings drop samples, and one gap of
        an hour would move a mean interval enough to misreport every rate
        derived from it.
        """
        return float(np.median(np.diff(self.time_s)))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def angular_increments_rad(self) -> NDArray[np.float64]:
        """Angle between consecutive Stokes vectors, in radians.

        Uses ``arctan2`` of the cross and dot products rather than
        ``arccos`` of the dot product alone: at the small angles that dominate
        a well-sampled recording, ``arccos`` loses half its significant digits.
        """
        return _angle_between(self.stokes[:-1], self.stokes[1:])


@dataclass
class DriftFit:
    """What a recording says about each term of the model.

    Attributes:
        drift_rate_rad_per_s: RMS angular velocity, from the increments.
        correlation_time_s: 1/e decay of the angular-velocity autocorrelation.
        spectral_exponent: fitted beta in ``S(f) ~ f**-beta``.
        jump_rate_per_s: Poisson rate of increments in the far tail.
        jump_scale_rad: mean size of those increments.
        vibration_frequency_hz: strongest narrowband peak, or ``None`` when the
            sampling rate cannot resolve one.
        vibration_amplitude_rad: RMS amplitude of that peak, or ``None``. It is
            a lower bound whenever it is not ``None`` — see
            :func:`_dominant_tone`.
        resolved: which terms this recording actually constrains.
        unresolved: which it does not, and why. Reported rather than silently
            defaulted — a fit that hides what it could not see is how an
            assumption becomes a measurement.
    """

    drift_rate_rad_per_s: float
    correlation_time_s: float
    spectral_exponent: float
    jump_rate_per_s: float
    jump_scale_rad: float
    vibration_frequency_hz: float | None
    vibration_amplitude_rad: float | None
    resolved: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_model(self, include_vibration: bool = False,
                 **overrides) -> PolarizationDriftModel:
        """Build the drift model this fit describes.

        Terms the recording could not resolve are left at zero rather than at
        their library defaults, so a model built from a 5 Hz subsea recording
        has no acoustic band at all — which is the truth about what was
        measured. Pass a band explicitly through ``overrides`` if you want to
        graft one on from somewhere else, and say so when you report it.

        Args:
            include_vibration: carry the fitted band into the model. Off by
                default, and deliberately: the amplitude is a lower bound, and
                a peak below the Nyquist frequency may be an alias of a band
                above it, which no amount of analysis of this recording can
                tell. A band that is wrong in frequency *and* low in amplitude
                is worse than no band, because it looks like a measurement.
            **overrides: any model field, applied last.

        Returns:
            The model.
        """
        parameters = dict(
            thermal_correlation_time_s=max(self.correlation_time_s, 1e-3),
            thermal_drift_rate_rad_per_s=self.drift_rate_rad_per_s,
            thermal_per_kelvin_rad_per_s=0.0,
            vibration_frequency_hz=0.0,
            vibration_amplitude_rad=0.0,
            jump_rate_per_s=self.jump_rate_per_s,
            jump_scale_rad=max(self.jump_scale_rad, 1e-6),
        )
        if include_vibration and self.vibration_frequency_hz:
            parameters["vibration_frequency_hz"] = self.vibration_frequency_hz
            parameters["vibration_amplitude_rad"] = self.vibration_amplitude_rad or 0.0
        parameters.update(overrides)
        return PolarizationDriftModel(**parameters)


def load_stokes(path: str | Path, time_column: str = "time_s",
                stokes_columns: tuple[str, str, str] = ("s1", "s2", "s3"),
                source: str | None = None) -> StokesRecording:
    """Read a polarimeter recording from a delimited text file.

    Args:
        path: CSV or whitespace-delimited file with a header row.
        time_column: name of the time column, in seconds.
        stokes_columns: names of the three normalised Stokes columns. ``S0`` is
            not used: this model is about where the polarisation points, not how
            much light arrives.
        source: provenance string; defaults to the file name.

    Returns:
        The recording, with Stokes vectors normalised to the Poincare sphere.

    Raises:
        ValueError: if a named column is missing, if fewer than 16 samples
            survive, or if any Stokes vector has zero length. A zero-length
            Stokes vector means the polarimeter lost the signal, and silently
            normalising it would invent a direction.
    """
    import pandas as pd

    path = Path(path)
    frame = pd.read_csv(path, sep=None, engine="python")
    missing = [name for name in (time_column, *stokes_columns)
               if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}; "
                         f"it has {list(frame.columns)}")

    frame = frame[[time_column, *stokes_columns]].dropna()
    if len(frame) < 16:
        raise ValueError(f"{path} has {len(frame)} usable rows; need at least 16")

    time_s = frame[time_column].to_numpy(dtype=float)
    stokes = frame[list(stokes_columns)].to_numpy(dtype=float)
    norms = np.linalg.norm(stokes, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(
            f"{path} contains {int((norms <= 0).sum())} zero-length Stokes "
            f"vectors — drop the rows where the polarimeter lost the signal")

    order = np.argsort(time_s)
    return StokesRecording(time_s=time_s[order] - time_s[order][0],
                           stokes=(stokes / norms)[order],
                           source=source or path.name)


def _angle_between(left: NDArray[np.float64], right: NDArray[np.float64]
                   ) -> NDArray[np.float64]:
    """Angle between paired unit vectors, conditioned for small angles."""
    dot = np.sum(left * right, axis=1)
    cross = np.linalg.norm(np.cross(left, right), axis=1)
    return np.arctan2(cross, dot)


def _drift_rate(stokes: NDArray[np.float64], dt_s: float,
                correlation_time_s: float, contamination_ratio: float = 3.0
                ) -> tuple[float, float, bool]:
    """Angular velocity, read off the structure function.

    The RMS angle between samples separated by a lag is the structure function
    of the walk, and the three terms of the model live in different parts of it.
    A vibration is bounded, so its contribution *falls* as a rate when the lag
    grows past its period. Drift is ballistic below its correlation time, so its
    contribution is a constant rate there. So the rate is the flat part: the
    minimum of speed-versus-lag, taken over lags short enough to stay inside the
    ballistic regime.

    Args:
        stokes: ``(n, 3)`` unit vectors.
        dt_s: sampling interval.
        correlation_time_s: the drift's correlation time, which bounds the lags
            worth looking at. Past it the walk is diffusive and the speed falls
            for a reason that has nothing to do with vibration; a quarter of it
            keeps the estimate inside the ballistic part.
        contamination_ratio: how much faster the one-sample speed may be than
            the plateau before the recording is declared to hold an unresolved
            fast component.

    Returns:
        ``(rate_rad_per_s, lag_s, contaminated)``. When ``contaminated`` is
        True there is motion faster than the drift that the lag sweep could not
        get out from under, and the rate is an upper bound.

    Note:
        A rotation-rate vector of per-axis RMS sigma moves a point on the
        sphere at sigma*sqrt(2): only the two components perpendicular to the
        position do anything. Dividing that out returns the model's own
        parameter rather than the observable speed.
    """
    samples = stokes.shape[0]
    ballistic = max(correlation_time_s / 4.0, dt_s)
    maximum_lag = int(np.clip(ballistic / dt_s, 1, max(samples // 8, 1)))

    def speed_at(lag: int) -> float:
        angles = _angle_between(stokes[:-lag], stokes[lag:])
        median = float(np.median(angles))
        typical = (angles[angles <= JUMP_THRESHOLD_SIGMA * 1.4826 * median]
                   if median > 0 else angles)
        if typical.size == 0:
            typical = angles
        return float(np.sqrt(np.mean(typical ** 2))) / (lag * dt_s)

    lags = np.unique(np.geomspace(1, maximum_lag, 10).astype(int))
    speeds = np.array([speed_at(int(lag)) for lag in lags])

    index = int(np.argmin(speeds))
    rate = float(speeds[index]) / np.sqrt(2.0)
    contaminated = bool(speeds[0] > contamination_ratio * speeds[index])
    return rate, float(lags[index] * dt_s), contaminated


def _autocorrelation_time(signal: NDArray[np.float64], dt_s: float) -> float:
    """First 1/e crossing of the autocorrelation, by linear interpolation."""
    centred = signal - signal.mean()
    if not np.any(centred):
        return dt_s
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1:]
    correlation = correlation / correlation[0]
    below = np.flatnonzero(correlation < np.exp(-1.0))
    if below.size == 0:
        # Never decorrelates inside the recording: the correlation time is at
        # least its length, and saying so beats reporting the length as if it
        # were the answer.
        return float(signal.size * dt_s)
    index = int(below[0])
    if index == 0:
        return dt_s
    previous, current = correlation[index - 1], correlation[index]
    fraction = (previous - np.exp(-1.0)) / max(previous - current, 1e-12)
    return float((index - 1 + fraction) * dt_s)


def _dominant_tone(stokes: NDArray[np.float64], dt_s: float,
                   minimum_prominence: float = 8.0,
                   minimum_share: float = 0.05
                   ) -> tuple[float, float] | None:
    """The strongest narrowband peak in the Stokes components, if there is one.

    Two tests have to pass, and both exist because of a way the first draft got
    it wrong. A peak must stand ``minimum_prominence`` times above the median of
    the spectrum — without that, every recording returns a vibration frequency,
    because the largest bin of a noise spectrum is still the largest bin. And it
    must carry at least ``minimum_share`` of the above-drift power, so a real
    but negligible tone is not promoted into a model parameter.

    The spectrum is taken of the **signed Stokes components**, not of the
    angular increments. The angle between consecutive samples is a magnitude and
    therefore rectified, and a rectified sinusoid peaks at twice its own
    frequency: fitting the increments reports a 60 Hz power-line tone as 120 Hz.

    Returns:
        ``(frequency_hz, rms_amplitude_rad)``, the amplitude being per axis, to
        match ``vibration_amplitude_rad``. ``None`` when no peak qualifies.

    Note:
        The frequency is reliable; the amplitude is a **lower bound**. A real
        mechanical band is not one spike but a cluster of them, and integrating
        down to 10 dB below the tallest captures the tallest and misses the
        rest — on a synthetic band of eight tones the amplitude comes back about
        four times low. Widening the integration instead swallows the drift
        shoulder, which is worse. Anyone who needs the amplitude should measure
        it, not infer it; :func:`fit_drift_model` labels it accordingly.
    """
    samples = stokes.shape[0]
    if samples < 64:
        return None

    centred = stokes - stokes.mean(axis=0)
    spectrum = (np.abs(np.fft.rfft(centred, axis=0)) ** 2).sum(axis=1)
    frequency = np.fft.rfftfreq(samples, dt_s)
    if spectrum.size < 16:
        return None

    # Ignore the lowest bins: drift lives there, and drift is not a tone.
    start = max(int(0.02 * spectrum.size), 2)
    background = float(np.median(spectrum[start:]))
    peak = int(np.argmax(spectrum[start:])) + start
    if background <= 0 or spectrum[peak] < minimum_prominence * background:
        return None

    # Integrate the whole band, not the peak bin. A mechanical band is not a
    # pure tone — eight tones spread over 2 Hz put their power across a hundred
    # bins of a minute-long recording — and summing three bins of that reports
    # an amplitude six times too small.
    # Integrate down to 10 dB below the peak, not down to the noise floor. On a
    # clean recording the noise floor is numerical, so a floor defined from it
    # expands the band across the whole spectrum and the tone disappears into
    # its own sidelobes.
    floor = max(float(spectrum[peak]) / 10.0, 3.0 * background)
    low = peak
    while low > start and spectrum[low - 1] > floor:
        low -= 1
    high = peak
    while high + 1 < spectrum.size and spectrum[high + 1] > floor:
        high += 1

    if low <= start:
        # The band runs into the low-frequency cutoff, which means this is the
        # drift shoulder and not a mechanical band. Drift is broadband down to
        # DC; a tone is not, and refusing to call one the other is the whole
        # difference between fitting a vibration and inventing one.
        return None
    if (high - low) > peak:
        # Fractional bandwidth above 100%: broadband, not narrowband.
        return None

    window = float(spectrum[low:high + 1].sum())
    total = float(spectrum[start:].sum())
    if total <= 0 or window / total < minimum_share:
        return None

    # Parseval over the band, divided across the three axes, gives the per-axis
    # RMS the band contributes. The frequency is the power-weighted centroid of
    # the band rather than its largest bin, which is noisier.
    amplitude = float(np.sqrt(2.0 * window / 3.0) / samples)
    weights = spectrum[low:high + 1]
    centre = float(np.sum(frequency[low:high + 1] * weights) / weights.sum())
    return centre, amplitude


def fit_drift_model(recording: StokesRecording,
                    spectral_band_hz: tuple[float, float] | None = None
                    ) -> DriftFit:
    """Estimate every drift-model term a recording can constrain.

    Args:
        recording: the measured history.
        spectral_band_hz: band to fit the spectral exponent over. ``None``
            picks the decade below the Nyquist frequency and above the inverse
            recording length, which is the widest band the data supports.

    Returns:
        The fit, including which terms it could not resolve and why.

    Raises:
        ValueError: if the recording is shorter than 32 samples.

    Note:
        The drift rate is reported at the recording's own sampling interval and
        **depends on it**, because a random walk has no characteristic velocity.
        A 5 Hz recording and a 4 kHz recording of the same fibre report
        different rates and both are right. ``diagnostics["dt_s"]`` carries the
        interval so the number is never quoted without it.
    """
    if recording.stokes.shape[0] < 32:
        raise ValueError("need at least 32 samples to fit anything")

    dt_s = recording.dt_s
    increments = recording.angular_increments_rad()
    nyquist = 0.5 / dt_s

    # Robust scale first: the jump term is defined relative to it, and a mean
    # or a standard deviation would already contain the jumps being looked for.
    median = float(np.median(np.abs(increments)))
    robust_sigma = 1.4826 * median
    threshold = JUMP_THRESHOLD_SIGMA * robust_sigma
    jumps = increments[increments > threshold] if robust_sigma > 0 else increments[:0]

    tone = _dominant_tone(recording.stokes, dt_s)

    correlation_time = _autocorrelation_time(np.linalg.norm(
        recording.stokes - recording.stokes.mean(axis=0), axis=1), dt_s)
    drift_rate, lag_s, contaminated = _drift_rate(recording.stokes, dt_s,
                                                  correlation_time)
    if spectral_band_hz is None:
        low = max(4.0 / recording.duration_s, 1e-6)
        spectral_band_hz = (low, min(low * 100.0, 0.4 * nyquist))

    from ..physics.field_calibration import spectral_exponent

    try:
        exponent = spectral_exponent(recording.stokes[:, 0], dt_s, *spectral_band_hz)
    except ValueError:
        exponent = float("nan")

    resolved = ["correlation time", "jump statistics"]
    unresolved = []

    # Reading the rate at a lag longer than the correlation time measures the
    # diffusive regime, not the ballistic one, and reports a smaller number for
    # the same fibre. So a lag that had to grow past the correlation time to
    # escape the vibration has not escaped anything useful.
    if contaminated:
        unresolved.append(
            "drift rate: motion faster than the drift is present at every lag "
            "inside the ballistic regime, so the rate is an upper bound")
    else:
        resolved.append("drift rate")
    if np.isnan(exponent):
        unresolved.append("spectral exponent: recording too short for the band")
    else:
        resolved.append("spectral exponent")
    if tone is None:
        unresolved.append(
            f"vibration band: no peak stands above the noise below the "
            f"{nyquist:.3g} Hz Nyquist frequency of this recording")
    else:
        resolved.append("vibration frequency")
        unresolved.append(
            "vibration amplitude: a lower bound only — band integration "
            "captures the tallest spike of a cluster and misses the rest")
        unresolved.append(
            f"vibration aliasing: a peak at {tone[0]:.3g} Hz may be a band "
            f"above the {nyquist:.3g} Hz Nyquist frequency folded down, and "
            f"this recording cannot tell the difference — check what the "
            f"instrument filtered before believing the frequency")

    return DriftFit(
        drift_rate_rad_per_s=drift_rate,
        correlation_time_s=correlation_time,
        spectral_exponent=exponent,
        jump_rate_per_s=float(jumps.size / recording.duration_s),
        jump_scale_rad=float(np.mean(jumps)) if jumps.size else 0.0,
        vibration_frequency_hz=tone[0] if tone else None,
        vibration_amplitude_rad=tone[1] if tone else None,
        resolved=tuple(resolved),
        unresolved=tuple(unresolved),
        diagnostics={
            "dt_s": dt_s,
            "duration_s": recording.duration_s,
            "nyquist_hz": nyquist,
            "samples": float(recording.stokes.shape[0]),
            "jump_count": float(jumps.size),
            "drift_lag_s": lag_s,
            "robust_increment_rad": robust_sigma,
            "band_low_hz": spectral_band_hz[0],
            "band_high_hz": spectral_band_hz[1],
        },
    )
