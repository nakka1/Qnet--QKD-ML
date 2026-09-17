"""Training data for real-time polarisation compensation.

The task
--------
A compensator sits at the receiver, watches whatever the link tells it about
its own polarisation alignment, and drives a controller to cancel the drift
before it costs QBER.  To learn that, a model needs pairs of (what the
compensator can see now, what the fibre will do next).

The trap
--------
It is very easy to generate a dataset in which the model can see the answer.
The true Poincare-sphere angle is right there in the simulator, and a dataset
that feeds it in as a feature will produce a compensator that scores
beautifully offline and fails on hardware, because on hardware nobody knows the
true angle.  What a receiver actually has is:

* a **monitoring estimate** — some fraction of the received photons are spent
  measuring alignment, giving a *binomial* estimate of the QBER from a finite
  sample.  With 1000 probe photons a QBER of 1% is known to about ±0.3%, and
  the angle inferred from it inherits that noise, badly amplified near zero
  where :math:`e = \\sin^2(\\Theta/2)` is flat;
* **environmental sensors** — a thermometer on the cable, an accelerometer on
  the duct.  Cheap, noisy, lagged, and only loosely coupled to the fibre.

This module models all three explicitly and keeps the ground truth in clearly
named ``true_*`` columns that are documented as *evaluation only*.
:func:`make_supervised_windows` will not put them in ``X``.

The other trap
--------------
Windowing with a horizon leaks trivially if you shuffle before splitting: rows
overlap, so a random split puts near-duplicates on both sides and the reported
error is meaningless.  :func:`make_supervised_windows` returns a ``segment``
index and :func:`chronological_split` cuts on time with a gap, which is the
only split that answers the question anyone cares about — can it predict
*forward*.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..physics.fiber import PolarizationDriftModel, stokes_qber
from .schema import Dataset, DatasetMetadata, describe_columns

__all__ = [
    "generate_polarization_dataset",
    "make_supervised_windows",
    "chronological_split",
    "COLUMNS",
]

#: What every column means. Enforced by :class:`~.schema.Dataset`.
COLUMNS = {
    "episode": "index of the independent run this row belongs to",
    "time_s": "time since the start of the episode",
    "true_angle_rad": "EVALUATION ONLY — actual Poincare-sphere misalignment",
    "true_qber": "EVALUATION ONLY — actual bit error probability from misalignment",
    "true_stokes_1": "EVALUATION ONLY — actual received Stokes component S1",
    "true_stokes_2": "EVALUATION ONLY — actual received Stokes component S2",
    "true_stokes_3": "EVALUATION ONLY — actual received Stokes component S3",
    "measured_qber": "QBER estimated from a finite monitoring sample (binomial noise)",
    "measured_angle_rad": "angle inferred from measured_qber; noisy, and ill-conditioned near zero",
    "probe_photons": "monitoring photons used for this estimate",
    "temperature_c": "cable thermometer reading, with sensor noise",
    "temperature_rate_c_per_s": "finite-difference dT/dt from the noisy thermometer",
    "vibration_rms": "accelerometer reading, with sensor noise",
    "seconds_since_jump": "time since the last discrete polarisation jump (evaluation aid)",
}


def _infer_angle_from_qber(qber: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert ``e = sin^2(Theta/2)``, clipped to the physical range.

    The inversion is well behaved in the middle and terrible at both ends: near
    :math:`\\Theta = 0` the derivative vanishes, so a small QBER error becomes a
    large angle error.  That ill-conditioning is real — a receiver genuinely
    cannot tell 0.5 degrees from 1 degree without a lot of photons — so it is
    left in rather than smoothed away.
    """
    return 2.0 * np.arcsin(np.sqrt(np.clip(qber, 0.0, 1.0)))


def generate_polarization_dataset(
        num_episodes: int = 8,
        duration_s: float = 600.0,
        dt_s: float = 0.1,
        probe_photons: int = 2000,
        temperature_noise_c: float = 0.05,
        vibration_noise: float = 2.0e-4,
        model: PolarizationDriftModel | None = None,
        launch_stokes: Sequence[float] = (1.0, 0.0, 0.0),
        seed: int = 0,
) -> Dataset:
    """Simulate several independent fibre spans and record what a receiver sees.

    Args:
        num_episodes: independent runs.  Each gets its own drift realisation,
            its own fibre birefringence axis and its own jump times, so a model
            cannot memorise one trajectory.
        duration_s: length of each run.  Make this several thermal correlation
            times or the slow component will not be represented.
        dt_s: sampling interval — the compensator's update period.
        probe_photons: monitoring photons behind each QBER estimate.  This is
            the single most important knob in the whole dataset: it sets the
            noise floor on the only direct observation the model gets, and it
            is a real engineering trade, since every probe photon is a photon
            not spent on key.
        temperature_noise_c: thermometer noise, standard deviation.
        vibration_noise: accelerometer noise, standard deviation.
        model: drift model; defaults to :class:`PolarizationDriftModel`.
        launch_stokes: transmitted polarisation.
        seed: base seed; episode *i* uses ``seed + i``.

    Returns:
        Dataset: one row per sample per episode.

    Raises:
        ValueError: for a non-positive episode count or probe count.
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if probe_photons < 1:
        raise ValueError("probe_photons must be at least 1")

    model = model or PolarizationDriftModel()
    frames = []

    for episode in range(num_episodes):
        rng = np.random.default_rng(seed + episode)
        trajectory = model.simulate(duration_s, dt_s, launch_stokes=launch_stokes, rng=rng)

        # Monitoring: `probe_photons` Bernoulli trials at the true error rate.
        errors = rng.binomial(probe_photons, np.clip(trajectory.qber, 0.0, 1.0))
        measured_qber = errors / probe_photons

        temperature = trajectory.temperature_c + rng.normal(
            0.0, temperature_noise_c, size=len(trajectory))
        vibration = np.abs(trajectory.vibration_rms
                           + rng.normal(0.0, vibration_noise, size=len(trajectory)))

        seconds_since_jump = _seconds_since_jump(trajectory.time_s, trajectory.jump_times_s)

        frames.append(pd.DataFrame({
            "episode": episode,
            "time_s": trajectory.time_s,
            "true_angle_rad": trajectory.angle_rad,
            "true_qber": trajectory.qber,
            "true_stokes_1": trajectory.stokes[:, 0],
            "true_stokes_2": trajectory.stokes[:, 1],
            "true_stokes_3": trajectory.stokes[:, 2],
            "measured_qber": measured_qber,
            "measured_angle_rad": _infer_angle_from_qber(measured_qber),
            "probe_photons": probe_photons,
            "temperature_c": temperature,
            "temperature_rate_c_per_s": np.gradient(temperature, dt_s),
            "vibration_rms": vibration,
            "seconds_since_jump": seconds_since_jump,
        }))

    frame = pd.concat(frames, ignore_index=True)

    metadata = DatasetMetadata(
        name="polarization_drift",
        description=("One row per compensator update: what the receiver observes "
                     "(noisy QBER estimate and environmental sensors) alongside the "
                     "true fibre state, which is for evaluation only."),
        parameters={
            "num_episodes": num_episodes,
            "duration_s": duration_s,
            "dt_s": dt_s,
            "probe_photons": probe_photons,
            "temperature_noise_c": temperature_noise_c,
            "vibration_noise": vibration_noise,
            "launch_stokes": list(launch_stokes),
            "drift_model": asdict(model),
        },
        seed=seed,
        columns=describe_columns(COLUMNS, frame),
    )
    return Dataset(frame=frame, metadata=metadata)


def _seconds_since_jump(time_s: NDArray[np.float64],
                        jump_times_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Time since the most recent jump, or since the start if there was none."""
    if jump_times_s.size == 0:
        return time_s.copy()
    indices = np.searchsorted(jump_times_s, time_s, side="right") - 1
    last_jump = np.where(indices >= 0, jump_times_s[np.clip(indices, 0, None)], 0.0)
    return time_s - last_jump


# ---------------------------------------------------------------------------
# supervised framing
# ---------------------------------------------------------------------------

#: Columns a compensator is allowed to see. Everything ``true_*`` is excluded.
DEFAULT_FEATURES = (
    "measured_qber",
    "measured_angle_rad",
    "temperature_c",
    "temperature_rate_c_per_s",
    "vibration_rms",
)


def make_supervised_windows(dataset: Dataset,
                            window: int = 50,
                            horizon: int = 10,
                            features: Sequence[str] = DEFAULT_FEATURES,
                            target: str = "true_angle_rad",
                            ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                       NDArray[np.int64], NDArray[np.float64]]:
    """Frame the dataset as "predict the misalignment ``horizon`` steps ahead".

    Windows never cross an episode boundary, and the target is strictly in the
    future of every feature in its window, so there is no leakage by
    construction.

    Args:
        dataset: output of :func:`generate_polarization_dataset`.
        window: how many past samples the model sees.
        horizon: how far ahead to predict, in samples.
        features: observable columns to use.  Passing a ``true_*`` column
            raises, because that is the mistake this function exists to
            prevent.
        target: what to predict.  A ``true_*`` column is correct *here* — it is
            the label, not an input.

    Returns:
        ``(X, y, episode, time_s)`` where ``X`` has shape
        ``(n, window, len(features))``, ``y`` shape ``(n,)``, and ``episode``
        and ``time_s`` identify each window's prediction point for splitting
        and plotting.

    Raises:
        ValueError: on a leaking feature, an unknown column, or a window and
            horizon that do not fit in any episode.
    """
    leaking = [name for name in features if name.startswith("true_")]
    if leaking:
        raise ValueError(
            f"{leaking} are ground-truth columns and must not be model inputs; "
            f"a receiver cannot measure them")

    frame = dataset.frame
    unknown = [name for name in list(features) + [target] if name not in frame.columns]
    if unknown:
        raise ValueError(f"columns not in the dataset: {unknown}")
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must both be at least 1")

    windows, targets, episodes, times = [], [], [], []
    for episode, group in frame.groupby("episode", sort=True):
        values = group[list(features)].to_numpy(dtype=float)
        labels = group[target].to_numpy(dtype=float)
        clock = group["time_s"].to_numpy(dtype=float)
        last_start = len(group) - window - horizon
        for start in range(last_start + 1):
            windows.append(values[start:start + window])
            targets.append(labels[start + window + horizon - 1])
            episodes.append(episode)
            times.append(clock[start + window - 1])

    if not windows:
        raise ValueError(
            f"window={window} + horizon={horizon} does not fit in any episode "
            f"(shortest has {frame.groupby('episode').size().min()} samples)")

    return (np.asarray(windows), np.asarray(targets),
            np.asarray(episodes, dtype=np.int64), np.asarray(times))


def chronological_split(episode: NDArray[np.int64], time_s: NDArray[np.float64],
                        train_fraction: float = 0.7,
                        gap_fraction: float = 0.02,
                        ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split each episode by time, with a gap, and no shuffling.

    Overlapping windows make a random split meaningless: neighbouring rows are
    near-duplicates, so the test set is effectively in the training set.  This
    cuts each episode at ``train_fraction`` of its span and discards a
    ``gap_fraction`` band around the cut so that no training window's horizon
    reaches into the test period.

    Returns:
        ``(train_mask, test_mask)`` — boolean masks over the window arrays.
        Windows inside the gap are in neither.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    train = np.zeros(episode.shape, dtype=bool)
    test = np.zeros(episode.shape, dtype=bool)

    for value in np.unique(episode):
        selector = episode == value
        clock = time_s[selector]
        span = clock.max() - clock.min()
        cut = clock.min() + train_fraction * span
        gap = gap_fraction * span
        train[selector] = clock <= cut - gap
        test[selector] = clock > cut + gap

    return train, test
