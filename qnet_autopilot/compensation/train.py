"""Fitting a predictor, on data it could actually have collected.

The target is the **future measurement**, not the future truth
------------------------------------------------------------
The simulator knows where the fibre really is, so it is tempting to regress on
that.  A deployed system has no such thing: all it will ever have is the
monitoring channel's noisy estimate, at the time and at the horizon.  Training
against ground truth therefore fits a model that cannot be retrained in the
field, and — worse — quietly credits the model for denoising it could only do
because it was shown the noise-free answer.

Regressing on the future *measurement* is what a deployed system can do, and
under a squared loss it has the same minimiser, because the measurement is an
unbiased estimate of the truth.  ``target="true"`` is available for diagnosis:
comparing the two says how much of the remaining error is measurement noise and
how much is the fibre being genuinely unpredictable.

The split is chronological, with a gap
--------------------------------------
Windows overlap, so a shuffled split puts near-duplicates on both sides and
reports a number that means nothing.  Each trajectory is cut in time, with a
guard band wider than the window plus the horizon so no training window can see
into the validation period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..agents.numpy_nn import MLP, Adam
from ..physics.fiber import PolarizationTrajectory
from .closed_loop import ControlConfig
from .monitoring import measure_rotation
from .predictors import (FeatureSpec, LearnedCompensator, Standardiser, _log_map,
                         build_features)

__all__ = ["Dataset", "build_dataset", "chronological_split", "fit_ridge", "fit_mlp",
           "MlpRegressor"]


@dataclass
class Dataset:
    """Windowed features and the rotation increment to predict.

    Attributes:
        features: shape ``(n, spec.size)``.
        targets: shape ``(n, 3)`` — the rotation vector from now to the horizon.
        episode: which trajectory each row came from.
        time_s: the time of the window's last sample, for splitting.
        spec: the feature layout, carried so a fitted model cannot be used with
            a different one by accident.
    """

    features: NDArray[np.float64]
    targets: NDArray[np.float64]
    episode: NDArray[np.int64]
    time_s: NDArray[np.float64]
    spec: FeatureSpec

    def __len__(self) -> int:
        return len(self.features)


def build_dataset(trajectories: Sequence[PolarizationTrajectory],
                  control: ControlConfig,
                  spec: FeatureSpec,
                  target: str = "measured",
                  rng: np.random.Generator | int | None = None) -> Dataset:
    """Turn simulated runs into supervised examples, exactly as inference sees them.

    The measurement, windowing and framing here must match
    :class:`~.predictors.LearnedCompensator` step for step — a mismatch between
    training and inference features is the single easiest way to produce a model
    that validates well and fails in the loop.

    Args:
        trajectories: runs simulated with ``track_rotation=True``.
        control: the loop timing; sets the control period and the horizon.
        spec: feature layout.
        target: ``"measured"`` (deployable) or ``"true"`` (diagnostic).
        rng: seed for the measurement noise.

    Returns:
        Dataset: ready to fit.

    Raises:
        ValueError: on an unknown target, or a trajectory without rotations.
    """
    if target not in ("measured", "true"):
        raise ValueError(f"unknown target {target!r}; expected 'measured' or 'true'")

    generator = np.random.default_rng(rng)
    features: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    episodes: list[int] = []
    times: list[float] = []

    for index, trajectory in enumerate(trajectories):
        if trajectory.rotation is None:
            raise ValueError("trajectories must be simulated with track_rotation=True")

        time_s = trajectory.time_s
        dt_s = float(time_s[1] - time_s[0])
        stride = max(int(round(control.control_period_s / dt_s)), 1)
        horizon_steps = max(int(round(control.latency_s / control.control_period_s)), 1)

        update_indices = np.arange(0, len(time_s), stride)
        measured = measure_rotation(trajectory.rotation[update_indices],
                                    control.monitoring, rng=generator)
        reference = (measured if target == "measured"
                     else trajectory.rotation[update_indices])

        covariates = np.stack([
            trajectory.temperature_c[update_indices],
            np.gradient(trajectory.temperature_c, dt_s)[update_indices],
            trajectory.vibration_rms[update_indices],
        ], axis=-1)

        last = len(update_indices) - horizon_steps
        for step in range(spec.window - 1, last):
            history = measured[step - spec.window + 1: step + 1]
            features.append(build_features(list(history), covariates[step], spec))
            increment = reference[step + horizon_steps] @ measured[step].T
            targets.append(_log_map(increment) / spec.horizon_scale)
            episodes.append(index)
            times.append(float(time_s[update_indices[step]]))

    return Dataset(features=np.asarray(features), targets=np.asarray(targets),
                   episode=np.asarray(episodes, dtype=np.int64),
                   time_s=np.asarray(times), spec=spec)


def chronological_split(dataset: Dataset, train_fraction: float = 0.7,
                        gap_fraction: float = 0.05
                        ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split each trajectory by time, discarding a guard band at the cut.

    Returns:
        ``(train_mask, validation_mask)``.  Rows in the gap are in neither.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    train = np.zeros(len(dataset), dtype=bool)
    validation = np.zeros(len(dataset), dtype=bool)

    for value in np.unique(dataset.episode):
        selector = dataset.episode == value
        clock = dataset.time_s[selector]
        span = clock.max() - clock.min()
        cut = clock.min() + train_fraction * span
        gap = gap_fraction * span
        train[selector] = clock <= cut - gap
        validation[selector] = clock > cut + gap

    return train, validation


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def fit_ridge(dataset: Dataset, train_mask: NDArray[np.bool_],
              alpha: float = 1.0, name: str = "ridge (Wiener)"
              ) -> LearnedCompensator:
    """Ridge regression on the windowed features.

    This is not a token linear baseline.  A linear predictor over a window of
    past samples **is** the Wiener filter — the optimal linear predictor for a
    stationary process — so it is the strongest classical thing that can be
    said about this signal, and it is the bar the network has to clear.  If the
    drift were a pure Gaussian process, nothing nonlinear could beat it.

    Args:
        dataset: from :func:`build_dataset`.
        train_mask: which rows to fit on.
        alpha: ridge penalty.
        name: display name.

    Returns:
        LearnedCompensator: ready to run in the loop.
    """
    from sklearn.linear_model import Ridge

    standardiser = Standardiser().fit(dataset.features[train_mask])
    model = Ridge(alpha=alpha)
    model.fit(standardiser.transform(dataset.features[train_mask]),
              dataset.targets[train_mask])
    return LearnedCompensator(model=model, spec=dataset.spec,
                              standardiser=standardiser, name=name)


@dataclass
class MlpRegressor:
    """A numpy MLP with a ``predict`` method, so it drops into the same wrapper.

    Trained with Adam on the mean squared error.  The same network and the same
    hand-written backward pass as the phase-2 agent, checked against finite
    differences in ``tests/test_agents.py``.
    """

    network: MLP

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.network.forward(np.atleast_2d(features))


def fit_mlp(dataset: Dataset, train_mask: NDArray[np.bool_],
            validation_mask: NDArray[np.bool_] | None = None,
            hidden: Sequence[int] = (96, 96),
            learning_rate: float = 3e-3,
            learning_rate_end: float = 3e-4,
            epochs: int = 60,
            batch_size: int = 256,
            seed: int = 0,
            name: str = "MLP",
            progress=None) -> tuple[LearnedCompensator, list[dict]]:
    """Fit the numpy MLP, keeping the best epoch by validation error.

    Args:
        dataset: from :func:`build_dataset`.
        train_mask, validation_mask: row selectors from
            :func:`chronological_split`.
        hidden: hidden layer widths.
        learning_rate, learning_rate_end: linearly decayed Adam step size.
        epochs: passes over the training rows.
        batch_size: minibatch size.
        seed: initialisation and shuffling.
        name: display name.
        progress: called with a dict per epoch.

    Returns:
        ``(compensator, history)``.
    """
    rng = np.random.default_rng(seed)
    standardiser = Standardiser().fit(dataset.features[train_mask])

    train_features = standardiser.transform(dataset.features[train_mask])
    train_targets = dataset.targets[train_mask]
    if validation_mask is not None and validation_mask.any():
        validation_features = standardiser.transform(dataset.features[validation_mask])
        validation_targets = dataset.targets[validation_mask]
    else:
        validation_features = validation_targets = None

    network = MLP([dataset.spec.size, *hidden, 3], rng=seed)
    optimiser = Adam(learning_rate=learning_rate, clip_norm=5.0)

    history: list[dict] = []
    best_error = np.inf
    best_weights = None
    count = len(train_features)

    for epoch in range(epochs):
        fraction = epoch / max(epochs - 1, 1)
        optimiser.learning_rate = (learning_rate
                                   + fraction * (learning_rate_end - learning_rate))

        order = rng.permutation(count)
        losses = []
        for start in range(0, count, batch_size):
            batch = order[start:start + batch_size]
            inputs = train_features[batch]
            targets = train_targets[batch]
            predicted = network.forward(inputs)
            residual = predicted - targets
            losses.append(float(np.mean(residual ** 2)))
            optimiser.step(network.parameters,
                           network.backward(2.0 * residual / len(batch)))

        record = {"epoch": epoch, "train_mse": float(np.mean(losses))}
        if validation_features is not None:
            error = float(np.mean(
                (network.forward(validation_features) - validation_targets) ** 2))
            record["validation_mse"] = error
            if error < best_error:
                best_error = error
                best_weights = [array.copy() for array in network.parameters]
        history.append(record)
        if progress is not None:
            progress(record)

    if best_weights is not None:
        for array, saved in zip(network.parameters, best_weights):
            array[...] = saved

    compensator = LearnedCompensator(model=MlpRegressor(network), spec=dataset.spec,
                                     standardiser=standardiser, name=name)
    return compensator, history
