"""Forecasting where each link's alignment error is going, from what was seen.

What is being forecast, and what is not
---------------------------------------
Only the **drift** component of the QBER. The total error a link shows is
alignment drift plus the Raman background the allocator itself just caused, and
those two must not be forecast together: the second is a consequence of the
controller's own action, and a model fitted to the sum learns to predict its own
control signal. Feed that model back into the controller and it chases itself.

So :func:`drift_estimate` strips the part the controller is responsible for
before anything is fitted or predicted. A deployed system can do this — it knows
its own transmit power and its own fibre's Raman coefficient — and it must.

Why a linear predictor is the right tool and not a placeholder
--------------------------------------------------------------
The drift here is an Ornstein-Uhlenbeck process with Poisson jumps, sampled
every few seconds. For the OU part the conditional mean is exactly
:math:`\\mu + (x_t - \\mu) e^{-h/\\tau}` — a linear function of the current
value — and no amount of model capacity improves on it. The jumps are
memoryless by construction and therefore unpredictable by anything.

Phase 3 reached the same conclusion for the same reason and measured it against
a neural network. That measurement is not repeated here; what is measured is
whether *using* a forecast changes the allocation, which is a different
question and the one this phase is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Forecaster", "PersistenceForecaster", "RidgeForecaster",
           "OracleForecaster", "drift_estimate", "build_windows", "fit_ridge_forecaster"]


def drift_estimate(measured_qber: ArrayLike, raman_qber: ArrayLike,
                   optical_error_rate: float = 0.005) -> NDArray[np.float64]:
    """Recover the fibre's own alignment error from a contaminated measurement.

    Args:
        measured_qber: what the link reported this step.
        raman_qber: the background the controller's own admission added, which
            it can compute from the power it chose.
        optical_error_rate: the encoder/decoder's fixed error, which is not
            drift and does not move.

    Returns:
        Non-negative drift estimate. Clipped at zero because measurement noise
        on a quiet link routinely puts the difference slightly negative, and a
        negative alignment error is not a thing.
    """
    return np.maximum(np.asarray(measured_qber, dtype=float)
                      - np.asarray(raman_qber, dtype=float)
                      - optical_error_rate, 0.0)


@runtime_checkable
class Forecaster(Protocol):
    """Predicts a link's drift over the next few control periods."""

    name: str
    horizon: int

    def reset(self, env) -> None:
        """Clear history at the start of an episode.

        Takes the environment rather than a link count so that a forecaster can
        bind to episode-specific state — which in practice only the oracle
        needs, and which is exactly the kind of access every other forecaster
        here must not have.
        """

    def predict(self, drift: NDArray[np.float64], step: int) -> NDArray[np.float64]:
        """Forecast from this step's drift estimate.

        Args:
            drift: current per-link drift estimate, shape ``(links,)``.
            step: control step index, so an oracle can look itself up.

        Returns:
            Shape ``(links, horizon)`` — drift at each of the next ``horizon``
            control periods.
        """


@dataclass
class PersistenceForecaster:
    """Assume the fibre stays where it is.

    The honest null hypothesis, and what every reactive controller implicitly
    believes. On a slow OU process over a short horizon it is very nearly right,
    which is exactly why the comparison has to be made rather than assumed: if
    persistence were enough, the whole predictive apparatus would be dead
    weight.
    """

    horizon: int = 12
    name: str = "persistência"

    def reset(self, env) -> None:
        pass

    def predict(self, drift: NDArray[np.float64], step: int) -> NDArray[np.float64]:
        return np.repeat(np.asarray(drift, dtype=float)[:, None], self.horizon, axis=1)


@dataclass
class OracleForecaster:
    """Knows the future drift exactly. The ceiling, not a policy.

    Present so that the gap between a real forecaster and perfect foresight is
    visible. A predictive policy that captures most of the oracle's advantage is
    done; one that captures a tenth of it has a forecasting problem, not an
    allocation problem, and the two are worth telling apart.

    It reads the environment's drift trace directly at ``reset``. That is
    cheating by construction and it is the only thing in phase 4 allowed to do
    it — nothing fitted, tuned or reported as deployable touches that array.
    """

    horizon: int = 12
    name: str = "oráculo"
    trace: NDArray[np.float64] | None = None

    def reset(self, env) -> None:
        self.trace = np.asarray(env._polarisation, dtype=float)

    def predict(self, drift: NDArray[np.float64], step: int) -> NDArray[np.float64]:
        if self.trace is None:
            raise RuntimeError("call reset(env) before predicting")
        steps = len(self.trace)
        indices = np.minimum(step + 1 + np.arange(self.horizon), steps - 1)
        return self.trace[indices].T


@dataclass
class RidgeForecaster:
    """Ridge regression on a window of past drift estimates.

    One model shared by every link. Links differ in how noisy their estimates
    are — a 145 km span measures itself from a hundredth of the sifted bits a
    30 km span does — but they share the same underlying drift process, so a
    per-link model would fit that noise rather than the physics, on a tenth of
    the data.

    Attributes:
        coefficients: shape ``(window + 1, horizon)``, including an intercept
            row. The intercept is what lets the model regress towards the
            process mean rather than towards zero.
        window: how many past estimates to use.
        horizon: how many control periods ahead to predict.
    """

    coefficients: NDArray[np.float64]
    window: int = 6
    horizon: int = 12
    name: str = "ridge"
    _history: list = field(default_factory=list, repr=False)

    def reset(self, env) -> None:
        self._history = []

    def predict(self, drift: NDArray[np.float64], step: int) -> NDArray[np.float64]:
        drift = np.asarray(drift, dtype=float)
        self._history.append(drift)
        if len(self._history) > self.window:
            self._history.pop(0)

        # Until the window fills, fall back to persistence rather than padding
        # with zeros: a fabricated history produces a confident wrong forecast,
        # and the opening steps of every episode would be corrupted.
        if len(self._history) < self.window:
            return np.repeat(drift[:, None], self.horizon, axis=1)

        features = np.stack(self._history, axis=1)              # (links, window)
        design = np.hstack([features, np.ones((len(drift), 1))])
        return np.maximum(design @ self.coefficients, 0.0)


def build_windows(traces: Sequence[NDArray[np.float64]], window: int = 6,
                  horizon: int = 12) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Windowed features and multi-horizon targets from per-episode drift traces.

    Args:
        traces: one array per episode, shape ``(steps, links)``.
        window: past estimates per feature vector.
        horizon: control periods to predict ahead.

    Returns:
        ``(features, targets)`` of shapes ``(n, window)`` and ``(n, horizon)``.

    Note:
        Every link of every episode contributes rows to the same pool, which is
        the point: one model, fitted on the process rather than on a link.
    """
    features: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []

    for trace in traces:
        trace = np.asarray(trace, dtype=float)
        steps, links = trace.shape
        last = steps - horizon
        for step in range(window - 1, last):
            past = trace[step - window + 1: step + 1]            # (window, links)
            future = trace[step + 1: step + 1 + horizon]         # (horizon, links)
            for link in range(links):
                features.append(past[:, link])
                targets.append(future[:, link])

    if not features:
        raise ValueError("traces are too short for this window and horizon")
    return np.asarray(features), np.asarray(targets)


def fit_ridge_forecaster(features: NDArray[np.float64], targets: NDArray[np.float64],
                         window: int = 6, horizon: int = 12,
                         alpha: float = 1.0e-3) -> RidgeForecaster:
    """Fit the shared drift model.

    Args:
        features, targets: from :func:`build_windows`.
        window, horizon: must match how the windows were built.
        alpha: ridge penalty, **relative to the features' own scale** (see the
            note). The point of the penalty here is numerical, not statistical:
            the design matrix is well conditioned and there are tens of
            thousands of rows behind a handful of coefficients.

    Returns:
        RidgeForecaster: ready to run in the loop.

    Raises:
        ValueError: if the shapes disagree with the declared window or horizon.

    Note:
        The penalty is scaled by the mean squared feature magnitude, which makes
        ``alpha`` dimensionless. Without that scaling it is measured in whatever
        units the features happen to be in, and these features are QBER values
        of order 0.005 — an absolute ``alpha`` of 1 then outweighs the entire
        data term by five orders of magnitude and shrinks the model to its
        intercept. That failure is silent: the fit still runs, still returns
        coefficients, and is simply worse than persistence at every horizon.
    """
    if features.shape[1] != window:
        raise ValueError(f"features have width {features.shape[1]}, expected {window}")
    if targets.shape[1] != horizon:
        raise ValueError(f"targets have width {targets.shape[1]}, expected {horizon}")

    design = np.hstack([features, np.ones((len(features), 1))])
    scale = float(np.mean(features ** 2)) or 1.0
    penalty = alpha * scale * len(features) * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0                       # never shrink the intercept
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return RidgeForecaster(coefficients=coefficients, window=window, horizon=horizon)
