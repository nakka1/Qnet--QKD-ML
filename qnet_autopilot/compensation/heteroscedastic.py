"""A compensator that predicts its own uncertainty, and acts on it.

The hypothesis
--------------
Phase 3 measured that a neural network does not beat the Wiener filter at
predicting this fibre, and explained why: the drift is a sum of a Gaussian
process and narrowband sinusoids, for which linear prediction is optimal. That
result is about the *mean*. It says nothing about the **variance**, and the
variance here is not constant:

* right after a cable jump the recent history is unrepresentative and any
  predictor is briefly wrong by a lot;
* on a long span the monitoring estimate is built from a hundredth of the
  photons a short span gets, so the same model is far less certain about it;
* during quiet thermal stretches almost any predictor is nearly exact.

A model that knows *when* it is unreliable can do something a better point
estimate cannot: stand down. The second head predicts a log-variance per axis,
and the loop uses it to shrink the correction towards the last measurement —
which is exactly persistence, the thing that is optimal when you know nothing
about the future. When the model is confident the correction is applied in
full; when it is not, the loop degrades gracefully towards the baseline instead
of confidently injecting the wrong phase.

The shrinkage is not a free parameter dressed up as Bayes
---------------------------------------------------------
Treating the predicted increment as a noisy observation of the true increment,
with the model's own variance :math:`\\sigma^2`, and persistence as a prior
centred on zero increment with variance :math:`\\tau^2`, the posterior mean is
the prediction scaled by :math:`\\tau^2 / (\\tau^2 + \\sigma^2)`. That is the
whole rule. :math:`\\tau^2` is the variance of the true increments, estimated
once from the training set, so the only thing moving at inference is the model's
own confidence.

Training this is where it goes wrong
------------------------------------
Plain Gaussian NLL on a shared body is unstable, and it failed here in the exact
way the literature describes. The gradient on the mean is
:math:`(\mu - y)/\sigma^2`, so as soon as the variance head inflates on a row,
that row stops training the mean; the residual stays large, which is evidence
for still more variance. The first attempt ran the variance out to a billion
times the target scale, the mean stopped moving, and the shrinkage gain went to
zero — a compensator that had learned to switch itself off.

The fix is :math:`\beta`-NLL (Seitzer et al., ICLR 2022): weight each sample's
loss by :math:`\sigma^{2\beta}`, treated as a constant for the gradient. At
:math:`\beta = 0` it is ordinary NLL; at :math:`\beta = 1` the mean's gradient
is exactly the MSE one and the variance head can no longer starve it. The 0.5
used here is the value that paper recommends, and the run below reports the
fraction of variance outputs pinned against their clamp so a repeat of the
original failure is visible rather than silent.

What would falsify it
---------------------
If the predicted variance carries no information — if it is roughly constant, or
uncorrelated with the actual squared error — the shrinkage becomes a constant
multiplier and the whole apparatus reduces to a badly-tuned gain. So the
variance head is scored on its own terms (does predicted variance track squared
error?) before the closed loop is run at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..agents.numpy_nn import MLP, Adam
from .predictors import FeatureSpec, Standardiser, _exp_map, build_features
from .train import Dataset

__all__ = ["HeteroscedasticRegressor", "ShrinkingCompensator", "fit_heteroscedastic",
           "gaussian_nll"]

#: How far, in natural logs, the predicted variance may sit either side of the
#: training targets' own variance.
#:
#: The clamp exists because Gaussian NLL is unbounded below: a network can drive
#: the variance towards zero on rows it happens to fit and take the loss to
#: minus infinity. But the bound has to be **relative to the data**. These
#: targets are rotation increments divided by the control period, so their
#: variance is of order 10^3; a clamp written as an absolute number sat an order
#: of magnitude below that, every row saturated against it, and the variance
#: head emitted one constant — which looks exactly like "the variance carries no
#: information" and is really "the variance was never allowed to move".
LOG_VARIANCE_MARGIN = 12.0

#: Fallback bounds for a regressor built without fitted bounds.
LOG_VARIANCE_BOUNDS = (-30.0, 30.0)


def gaussian_nll(target: NDArray[np.float64], mean: NDArray[np.float64],
                 log_variance: NDArray[np.float64]) -> float:
    """Mean negative log-likelihood of a diagonal Gaussian, up to a constant.

    .. math:: \\tfrac{1}{2}\\bigl[\\log \\sigma^2
              + (y - \\mu)^2 / \\sigma^2\\bigr]

    Args:
        target, mean, log_variance: shape ``(n, 3)``.

    Returns:
        Mean over rows and axes.
    """
    residual = mean - target
    return float(np.mean(0.5 * (log_variance + residual ** 2 * np.exp(-log_variance))))


@dataclass
class HeteroscedasticRegressor:
    """One network, six outputs: three means and three log-variances.

    A single trunk with two heads in the simplest possible arrangement — the
    output layer is split rather than duplicated — because the two quantities
    are functions of the same features and the useful part of the hypothesis is
    that predicting the variance *at all* changes what the loop does, not that
    it needs its own tower to do it.

    Attributes:
        network: the shared network, ``spec.size -> ... -> 6``.
        log_variance_bounds: clamp on the variance head, set from the training
            targets' scale at fit time.
    """

    network: MLP
    log_variance_bounds: tuple[float, float] = LOG_VARIANCE_BOUNDS

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean only, so this drops into the existing compensator wrapper."""
        return self.forward(features)[0]

    def forward(self, features: NDArray[np.float64]
                ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns ``(mean, log_variance)``, each shape ``(n, 3)``."""
        outputs = self.network.forward(np.atleast_2d(features))
        mean = outputs[:, :3]
        log_variance = np.clip(outputs[:, 3:], *self.log_variance_bounds)
        return mean, log_variance

    def saturated_fraction(self, features: NDArray[np.float64]) -> float:
        """Share of variance outputs pinned against the clamp.

        Worth checking rather than assuming: a saturated head emits a constant,
        which is indistinguishable from an uninformative one in every downstream
        metric and means something entirely different.
        """
        raw = self.network.forward(np.atleast_2d(features))[:, 3:]
        low, high = self.log_variance_bounds
        return float(np.mean((raw <= low) | (raw >= high)))


@dataclass
class ShrinkingCompensator:
    """Applies the predicted increment, scaled by how much the model trusts it.

    Attributes:
        model: a :class:`HeteroscedasticRegressor`.
        spec: the feature layout it was fitted with.
        standardiser: the scaling it was fitted with.
        prior_variance: variance of the true increments, per axis, from the
            training set. The :math:`\\tau^2` of the shrinkage rule.
        shrink: when False, ignore the variance head and apply the mean in full.
            The ablation: same weights, same means, shrinkage off.
        name: display name.
    """

    model: HeteroscedasticRegressor
    spec: FeatureSpec
    standardiser: Standardiser
    prior_variance: NDArray[np.float64]
    shrink: bool = True
    name: str = "dual-head (média + variância)"

    def reset(self) -> None:
        from collections import deque

        from .predictors import Persistence

        self._history: deque = deque(maxlen=self.spec.window)
        self._fallback = Persistence()
        self.last_gain: NDArray[np.float64] = np.ones(3)

    def __post_init__(self) -> None:
        self.reset()

    def predict(self, time_s, measured, covariates, horizon_s):
        measured = np.asarray(measured, dtype=float)
        self._history.append(measured)
        if len(self._history) < self.spec.window:
            return self._fallback.predict(time_s, measured, covariates, horizon_s)

        features = build_features(list(self._history), covariates, self.spec)
        scaled = self.standardiser.transform(features[None, :])
        mean, log_variance = self.model.forward(scaled)
        increment = mean.reshape(3)

        if self.shrink:
            # Posterior mean under a zero-centred prior of variance tau^2 and a
            # likelihood of variance sigma^2: the prediction, shrunk towards no
            # movement in proportion to how unsure the model is.
            variance = np.exp(log_variance.reshape(3))
            gain = self.prior_variance / (self.prior_variance + variance)
            self.last_gain = gain
            increment = increment * gain

        return _exp_map(increment * self.spec.horizon_scale) @ measured


def fit_heteroscedastic(dataset: Dataset, train_mask: NDArray[np.bool_],
                        validation_mask: NDArray[np.bool_] | None = None,
                        hidden: Sequence[int] = (192, 192),
                        learning_rate: float = 3e-3,
                        learning_rate_end: float = 3e-4,
                        epochs: int = 60,
                        batch_size: int = 256,
                        warmup_epochs: int = 5,
                        beta: float = 0.5,
                        seed: int = 0,
                        name: str = "dual-head (média + variância)",
                        progress=None) -> tuple[ShrinkingCompensator, list[dict]]:
    """Fit mean and log-variance heads under a Gaussian likelihood.

    Args:
        dataset: from :func:`~.train.build_dataset`.
        train_mask, validation_mask: row selectors from
            :func:`~.train.chronological_split`.
        hidden: hidden layer widths of the shared body.
        learning_rate, learning_rate_end: linearly decayed Adam step size.
        epochs: passes over the training rows.
        batch_size: minibatch size.
        warmup_epochs: train the mean head on plain squared error for this many
            epochs before switching on the likelihood, so the variance head is
            not asked to explain a residual the mean has not had a chance to
            reduce.
        beta: the exponent of the :math:`\sigma^{2\beta}` weighting. 0 is
            ordinary NLL, which is unstable here; 1 leaves the mean trained
            exactly as by MSE. 0.5 is the recommended compromise.
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

    # Bounds and starting point both come from the targets' own scale, so the
    # variance head begins calibrated and has room to move in either direction.
    target_variance = float(np.mean(np.var(train_targets, axis=0)))
    centre = float(np.log(max(target_variance, 1e-30)))
    bounds = (centre - LOG_VARIANCE_MARGIN, centre + LOG_VARIANCE_MARGIN)

    network = MLP([dataset.spec.size, *hidden, 6], rng=seed)
    network.biases[-1][3:] = centre
    optimiser = Adam(learning_rate=learning_rate, clip_norm=5.0)
    model = HeteroscedasticRegressor(network=network, log_variance_bounds=bounds)

    history: list[dict] = []
    best_error = np.inf
    best_weights = None
    count = len(train_features)

    for epoch in range(epochs):
        fraction = epoch / max(epochs - 1, 1)
        optimiser.learning_rate = (learning_rate
                                   + fraction * (learning_rate_end - learning_rate))
        likelihood = epoch >= warmup_epochs

        order = rng.permutation(count)
        losses = []
        for start in range(0, count, batch_size):
            batch = order[start:start + batch_size]
            inputs = train_features[batch]
            targets = train_targets[batch]

            outputs = network.forward(inputs)
            mean = outputs[:, :3]
            raw_log_variance = outputs[:, 3:]
            log_variance = np.clip(raw_log_variance, *bounds)
            residual = mean - targets

            # Both losses average over rows *and* the three axes, so the
            # gradient is divided by the element count, not the batch size.
            # (``fit_mlp`` divides by the batch size, which is the same
            # objective at three times the step; matching conventions here is
            # what lets the mean-only control below be a like-for-like one.)
            elements = residual.size

            gradient = np.zeros_like(outputs)
            if likelihood:
                variance = np.exp(log_variance)
                precision = 1.0 / variance
                # beta-NLL: every per-sample term is scaled by sigma^(2 beta),
                # held constant for the gradient. At beta = 1 the mean's
                # gradient below reduces to the MSE one, which is the point —
                # the variance head can no longer switch off the mean's
                # learning signal on the rows it finds hard.
                weight = variance ** beta
                gradient[:, :3] = weight * residual * precision / elements
                # d/d(log var) of 0.5*(log v + r^2/v) = 0.5*(1 - r^2/v).
                #
                # At the clamp the gradient passes through only if it points
                # back into the feasible range. Zeroing it outright — the
                # obvious guard — is a one-way trap: the first batch that drives
                # an output past a bound removes its gradient entirely and it
                # can never come back, so the head emits a constant for the rest
                # of training. That is not hypothetical; it is what the first
                # version of this did, and the symptom is indistinguishable from
                # a variance head that has learned nothing.
                variance_gradient = weight * 0.5 * (1.0 - residual ** 2 * precision)
                low, high = bounds
                blocked = (((raw_log_variance <= low) & (variance_gradient > 0))
                           | ((raw_log_variance >= high) & (variance_gradient < 0)))
                gradient[:, 3:] = np.where(blocked, 0.0,
                                           variance_gradient) / elements
                losses.append(gaussian_nll(targets, mean, log_variance))
            else:
                gradient[:, :3] = 2.0 * residual / elements
                losses.append(float(np.mean(residual ** 2)))

            optimiser.step(network.parameters, network.backward(gradient))

        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "likelihood": bool(likelihood)}
        if validation_features is not None:
            record["saturated"] = model.saturated_fraction(validation_features)
        if validation_features is not None:
            mean, log_variance = model.forward(validation_features)
            record["validation_mse"] = float(np.mean((mean - validation_targets) ** 2))
            record["validation_nll"] = gaussian_nll(validation_targets, mean,
                                                    log_variance)
            # Selected on the mean's error, not the likelihood: the loop is
            # scored on QBER, and a checkpoint that wins on calibration while
            # losing on the mean is not the one to keep.
            if record["validation_mse"] < best_error:
                best_error = record["validation_mse"]
                best_weights = [array.copy() for array in network.parameters]
        history.append(record)
        if progress is not None:
            progress(record)

    if best_weights is not None:
        for array, saved in zip(network.parameters, best_weights):
            array[...] = saved

    prior_variance = np.var(dataset.targets[train_mask], axis=0)
    compensator = ShrinkingCompensator(model=model, spec=dataset.spec,
                                       standardiser=standardiser,
                                       prior_variance=prior_variance, name=name)
    return compensator, history
