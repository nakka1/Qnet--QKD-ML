"""Running feature normalisation, fitted on what the agent actually sees.

Why this is not a detail
-----------------------
The state and action features of phase 2 mix quantities that live on very
different scales: fidelities near 0.9, times in milliseconds that run to tens,
counts of memory slots, and booleans. The first layer of an MLP sees their sum
weighted by initialisation, so before any learning happens the largest-magnitude
feature dominates the representation and the small ones contribute almost
nothing to the gradient. He initialisation fixes the *layer-to-layer* scale; it
does nothing about the input scale.

Why a running estimate rather than a fixed one
----------------------------------------------
There is no dataset to fit on. The distribution of states an agent visits is a
consequence of the policy, and the policy changes throughout training — an agent
that has learned to purify visits states an agent that has not never reaches.
Fitting statistics once on the initial random policy would normalise for a
distribution the agent leaves behind.

So the statistics are accumulated online, with Welford's algorithm, over exactly
the features the agent encounters. They are frozen when training ends and saved
with the weights, because at evaluation time the normalisation has to be a fixed
function — an agent whose preprocessing keeps moving is not a policy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["FeatureNormaliser"]


class FeatureNormaliser:
    """Zero-mean, unit-variance scaling with online statistics.

    Attributes:
        count: how many rows have been seen.
        mean: per-feature mean.
        m2: per-feature sum of squared deviations, which is what Welford
            accumulates; the variance is ``m2 / count``.
        frozen: when True, :meth:`update` does nothing.

    Note:
        Welford rather than accumulating sums of squares, because the sum of
        squares of a feature with a large mean loses precision to cancellation
        exactly where the variance is small — which here is the boolean and
        near-constant features, the ones normalisation matters most for.
    """

    def __init__(self, size: int, epsilon: float = 1e-8):
        self.size = int(size)
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = np.zeros(self.size)
        self.m2 = np.zeros(self.size)
        self.frozen = False

    # -- statistics ---------------------------------------------------------

    @property
    def variance(self) -> NDArray[np.float64]:
        if self.count < 2:
            return np.ones(self.size)
        return self.m2 / self.count

    @property
    def scale(self) -> NDArray[np.float64]:
        """Standard deviation, floored so a constant feature is passed through.

        A feature that never varies has zero variance, and dividing by it would
        turn a harmless constant into an infinity. Flooring at one leaves such a
        feature at its raw magnitude, where the bias term can absorb it.
        """
        return np.sqrt(np.maximum(self.variance, self.epsilon))

    def update(self, rows: NDArray[np.float64]) -> "FeatureNormaliser":
        """Fold a batch of observed feature rows into the statistics.

        Args:
            rows: shape ``(n, size)`` or ``(size,)``.

        Returns:
            self, for chaining.

        Raises:
            ValueError: if the row width does not match.
        """
        if self.frozen:
            return self

        batch = np.atleast_2d(np.asarray(rows, dtype=float))
        if batch.shape[1] != self.size:
            raise ValueError(
                f"expected rows of width {self.size}, got {batch.shape[1]}")

        # Chan's parallel form of Welford: fold the whole batch at once rather
        # than row by row, which is both faster and numerically identical.
        batch_count = len(batch)
        if batch_count == 0:
            return self
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + (delta ** 2) * (self.count * batch_count / total)
        self.count = total
        return self

    def freeze(self) -> "FeatureNormaliser":
        """Stop updating. Called when training ends."""
        self.frozen = True
        return self

    # -- use ----------------------------------------------------------------

    def transform(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardise, passing the input straight through until fitted."""
        values = np.asarray(rows, dtype=float)
        if self.count < 2:
            return values
        return (values - self.mean) / self.scale

    # -- persistence --------------------------------------------------------

    def state(self) -> dict[str, NDArray[np.float64]]:
        return {"count": np.asarray(float(self.count)), "mean": self.mean,
                "m2": self.m2}

    def load_state(self, data) -> "FeatureNormaliser":
        self.count = int(float(np.asarray(data["count"])))
        self.mean = np.asarray(data["mean"], dtype=float).copy()
        self.m2 = np.asarray(data["m2"], dtype=float).copy()
        return self
