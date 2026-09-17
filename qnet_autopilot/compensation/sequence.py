"""A temporal convolutional predictor for the polarisation loop.

What this adds to phase 3
-------------------------
The phase-3 learned compensator is an MLP over a *flattened* window: the
sixteen past rotation increments arrive as a single 45-dimensional vector, and
the first layer has to learn from scratch that entries 0-2, 3-5 and 6-8 are the
same three quantities at successive lags.  It does learn it, and the result is
in the phase-3 table.  But nothing in the architecture knows the input is a time
series, so every temporal pattern has to be relearned separately at every lag,
which costs parameters and — more to the point — costs data.

A causal convolution knows.  The same three-channel filter is applied at every
lag, so a pattern learned at one position is available at all of them, and the
receptive field is grown by stacking dilated layers rather than by widening the
first layer.  For the drift in phase 3 that matters specifically because of the
**vibration band**: an oscillation is the same shape wherever it sits in the
window, and a shift-equivariant model is the right prior for it, while a dense
layer over a flattened window is close to the worst one.

Why a convolution and not an LSTM
---------------------------------
A recurrent model is the other obvious choice and would fit the same data.  It
is not the one here, for three reasons that are specific to this loop:

* **Causality is structural, not learned.**  A causal convolution cannot see the
  future because of how it is indexed.  Nothing has to enforce it, and no bug
  can quietly break it — which, in a control loop whose whole subject is a
  latency, is worth more than it sounds.
* **The receptive field is a design parameter.**  Two dilated layers with kernel
  3 reach 7 steps; three reach 15.  With a recurrent model the effective memory
  is a property of the trained weights, and "how far back does it actually look"
  becomes a question you answer with an experiment instead of with arithmetic.
* **It runs inside a kilohertz control loop.**  The forward pass here is two
  small matrix multiplies on a window that is already in memory; a recurrent
  pass is a Python loop over time steps, per control step, in the loop.

The interface is deliberately unchanged
---------------------------------------
:class:`SequenceRegressor` exposes ``predict(X) -> (n, 3)`` on exactly the
feature matrix :func:`qnet_autopilot.compensation.train.build_dataset` already
produces — the flattened window plus covariates — and reshapes it internally.
So it drops into :class:`~qnet_autopilot.compensation.predictors.LearnedCompensator`,
runs in the same closed loop, and is scored by the same
:func:`~qnet_autopilot.compensation.closed_loop.run_closed_loop` against the same
baselines.  Nothing about the comparison changes except the model.

The gradients are hand-written, like everything else in this package, and
``tests/test_qkd.py`` checks them against finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..agents.numpy_nn import Adam
from .predictors import FeatureSpec, LearnedCompensator, Standardiser

__all__ = ["CausalConv1d", "SequenceSpec", "TemporalConvNet", "SequenceRegressor",
           "fit_sequence"]


class CausalConv1d:
    """One dilated causal convolution over a ``(batch, time, channels)`` tensor.

    Output step *t* depends on input steps :math:`t - d(k-1), \\ldots, t` and on
    nothing later, with the left edge zero-padded.  Implemented as a single
    matrix multiply over gathered patches rather than a loop over time, because
    the whole point of preferring this to a recurrent model is that it is one
    matrix multiply.

    Attributes:
        weight: shape ``(kernel * in_channels, out_channels)``.
        bias: shape ``(out_channels,)``.
        kernel: taps per filter.
        dilation: spacing between taps.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3,
                 dilation: int = 1, rng: np.random.Generator | int | None = None,
                 zero_output: bool = False):
        """
        Args:
            in_channels, out_channels: channel counts.
            kernel: taps per filter.
            dilation: spacing between taps; 1, 2, 4 ... over stacked layers grows
                the receptive field geometrically.
            rng: seed or generator.
            zero_output: initialise the weights to zero.  Used for the last
                layer of a predictor so it starts by predicting no increment,
                which is persistence — the right thing to be before training.

        Raises:
            ValueError: on a non-positive kernel or dilation.
        """
        if kernel < 1 or dilation < 1:
            raise ValueError("kernel and dilation must be at least 1")

        generator = np.random.default_rng(rng)
        fan_in = kernel * in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.dilation = dilation
        self.weight = (np.zeros((fan_in, out_channels)) if zero_output
                       else generator.normal(0.0, np.sqrt(2.0 / fan_in),
                                             size=(fan_in, out_channels)))
        self.bias = np.zeros(out_channels)
        self._patches: NDArray[np.float64] | None = None
        self._input_shape: tuple[int, int, int] | None = None

    @property
    def parameters(self) -> list[NDArray[np.float64]]:
        return [self.weight, self.bias]

    @property
    def receptive_field(self) -> int:
        """How many input steps one output step can see."""
        return self.dilation * (self.kernel - 1) + 1

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run the convolution.

        Args:
            inputs: shape ``(batch, time, in_channels)``.

        Returns:
            Shape ``(batch, time, out_channels)``.
        """
        inputs = np.asarray(inputs, dtype=float)
        batch, time, channels = inputs.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")

        pad = self.dilation * (self.kernel - 1)
        padded = np.zeros((batch, time + pad, channels))
        padded[:, pad:, :] = inputs

        taps = [padded[:, tap * self.dilation: tap * self.dilation + time, :]
                for tap in range(self.kernel)]
        patches = np.concatenate(taps, axis=-1).reshape(batch * time, -1)

        self._patches = patches
        self._input_shape = (batch, time, channels)
        return (patches @ self.weight + self.bias).reshape(batch, time, self.out_channels)

    def backward(self, output_gradient: NDArray[np.float64]
                 ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Parameter gradients and the gradient with respect to the input.

        Args:
            output_gradient: dL/d(output), shape ``(batch, time, out_channels)``.

        Returns:
            ``([weight_grad, bias_grad], input_grad)``.

        Raises:
            RuntimeError: if called before :meth:`forward`.
        """
        if self._patches is None or self._input_shape is None:
            raise RuntimeError("call forward() before backward()")

        batch, time, channels = self._input_shape
        gradient = np.asarray(output_gradient, dtype=float).reshape(batch * time, -1)

        weight_gradient = self._patches.T @ gradient
        bias_gradient = gradient.sum(axis=0)

        # Scatter the patch gradients back to the padded input: every tap of
        # every output step wrote into one input step, and several of them wrote
        # into the same one, so the adjoint accumulates rather than assigns.
        patch_gradient = (gradient @ self.weight.T).reshape(batch, time, self.kernel,
                                                            channels)
        pad = self.dilation * (self.kernel - 1)
        padded_gradient = np.zeros((batch, time + pad, channels))
        for tap in range(self.kernel):
            start = tap * self.dilation
            padded_gradient[:, start:start + time, :] += patch_gradient[:, :, tap, :]

        return [weight_gradient, bias_gradient], padded_gradient[:, pad:, :]


@dataclass(frozen=True)
class SequenceSpec:
    """The shape of the temporal model, and how it reads the feature vector.

    Attributes:
        feature_spec: the phase-3 feature layout the matrix was built with.  The
            first ``3 * (window - 1)`` entries are the lagged rotation
            increments, three channels per lag, and the rest are the covariates.
        channels: width of the convolutional layers.
        kernel: taps per filter.
        dilations: one entry per layer.  ``(1, 2)`` with kernel 3 reaches seven
            lags, which spans a period of the vibration band at the phase-3
            operating point; ``(1, 2, 4)`` reaches fifteen.
        hidden: width of the dense head that sits on the last time step.

    Note:
        The lags run oldest-last in the feature vector — element *i* is where the
        fibre was *i* steps ago — so the sequence is reversed before the
        convolution sees it.  A causal model fed a time-reversed series is not
        merely worse, it is answering a different question, and this is the kind
        of mistake that produces a model which validates well and fails in the
        loop.
    """

    feature_spec: FeatureSpec
    channels: int = 32
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2)
    hidden: int = 64

    @property
    def steps(self) -> int:
        """Time steps in the sequence part of the feature vector."""
        return self.feature_spec.window - 1

    @property
    def covariates(self) -> int:
        """How many non-sequence features are appended."""
        return 3 if self.feature_spec.use_covariates else 0

    @property
    def receptive_field(self) -> int:
        """How many lags the stack can see."""
        return 1 + sum(dilation * (self.kernel - 1) for dilation in self.dilations)


class TemporalConvNet:
    """Stacked causal convolutions, then a dense head on the last step.

    Attributes:
        spec: the layout.
        layers: the convolutional stack.
        head_weight, head_bias: the dense layer mapping the last time step's
            features, concatenated with the covariates, to a rotation increment.
        output_weight, output_bias: the linear output layer, zero-initialised so
            an untrained model predicts no increment — that is persistence, and
            it is the right prior.
    """

    def __init__(self, spec: SequenceSpec, rng: np.random.Generator | int | None = None):
        generator = np.random.default_rng(rng)
        self.spec = spec

        self.layers: list[CausalConv1d] = []
        in_channels = 3
        for dilation in spec.dilations:
            self.layers.append(CausalConv1d(in_channels, spec.channels,
                                            kernel=spec.kernel, dilation=dilation,
                                            rng=generator))
            in_channels = spec.channels

        fan_in = spec.channels + spec.covariates
        self.head_weight = generator.normal(0.0, np.sqrt(2.0 / fan_in),
                                            size=(fan_in, spec.hidden))
        self.head_bias = np.zeros(spec.hidden)
        self.output_weight = np.zeros((spec.hidden, 3))
        self.output_bias = np.zeros(3)

        self._cache: dict[str, NDArray[np.float64]] = {}

    @property
    def parameters(self) -> list[NDArray[np.float64]]:
        """Every array, in a fixed order, for the optimiser to update in place."""
        arrays: list[NDArray[np.float64]] = []
        for layer in self.layers:
            arrays.extend(layer.parameters)
        arrays.extend([self.head_weight, self.head_bias,
                       self.output_weight, self.output_bias])
        return arrays

    def parameter_count(self) -> int:
        return sum(array.size for array in self.parameters)

    def get_flat(self) -> NDArray[np.float64]:
        return np.concatenate([array.ravel() for array in self.parameters])

    def set_flat(self, flat: NDArray[np.float64]) -> None:
        """Load parameters from a flat vector — used by the gradient check."""
        offset = 0
        for array in self.parameters:
            size = array.size
            array[...] = flat[offset:offset + size].reshape(array.shape)
            offset += size
        if offset != flat.size:
            raise ValueError(f"expected {offset} parameters, got {flat.size}")

    def _split(self, features: NDArray[np.float64]
               ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Feature matrix to ``(sequence, covariates)``, oldest first."""
        spec = self.spec
        rows = np.atleast_2d(np.asarray(features, dtype=float))
        expected = 3 * spec.steps + spec.covariates
        if rows.shape[1] != expected:
            raise ValueError(f"expected {expected} features, got {rows.shape[1]}")

        sequence = rows[:, :3 * spec.steps].reshape(len(rows), spec.steps, 3)
        sequence = sequence[:, ::-1, :]  # oldest first; see SequenceSpec's note
        covariates = rows[:, 3 * spec.steps:]
        return np.ascontiguousarray(sequence), covariates

    def forward(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict a rotation increment per row.

        Args:
            features: shape ``(batch, spec.feature_spec.size)``.

        Returns:
            Shape ``(batch, 3)``.
        """
        sequence, covariates = self._split(features)

        activations = [sequence]
        activation = sequence
        for layer in self.layers:
            activation = np.maximum(layer.forward(activation), 0.0)
            activations.append(activation)

        last = activation[:, -1, :]
        head_input = (np.concatenate([last, covariates], axis=1)
                      if self.spec.covariates else last)
        hidden_pre = head_input @ self.head_weight + self.head_bias
        hidden = np.maximum(hidden_pre, 0.0)
        output = hidden @ self.output_weight + self.output_bias

        self._cache = {"activations": activations, "head_input": head_input,
                       "hidden": hidden, "covariates": covariates}
        return output

    def backward(self, output_gradient: NDArray[np.float64]
                 ) -> list[NDArray[np.float64]]:
        """Gradients of the loss with respect to every parameter.

        Args:
            output_gradient: dL/d(output), shape ``(batch, 3)``.

        Returns:
            One array per entry of :attr:`parameters`, same shapes and order.

        Raises:
            RuntimeError: if called before :meth:`forward`.
        """
        if not self._cache:
            raise RuntimeError("call forward() before backward()")

        gradient = np.asarray(output_gradient, dtype=float)
        hidden = self._cache["hidden"]
        head_input = self._cache["head_input"]
        activations = self._cache["activations"]

        output_weight_gradient = hidden.T @ gradient
        output_bias_gradient = gradient.sum(axis=0)

        hidden_gradient = (gradient @ self.output_weight.T) * (hidden > 0)
        head_weight_gradient = head_input.T @ hidden_gradient
        head_bias_gradient = hidden_gradient.sum(axis=0)

        input_gradient = hidden_gradient @ self.head_weight.T
        last_gradient = input_gradient[:, :self.spec.channels]

        # Only the last time step fed the head, so every earlier step's gradient
        # arrives through the convolutions and starts at zero here.
        sequence_gradient = np.zeros_like(activations[-1])
        sequence_gradient[:, -1, :] = last_gradient

        convolution_gradients: list[NDArray[np.float64]] = []
        for index in range(len(self.layers) - 1, -1, -1):
            sequence_gradient = sequence_gradient * (activations[index + 1] > 0)
            parameters, sequence_gradient = self.layers[index].backward(sequence_gradient)
            convolution_gradients = parameters + convolution_gradients

        return convolution_gradients + [head_weight_gradient, head_bias_gradient,
                                        output_weight_gradient, output_bias_gradient]


@dataclass
class SequenceRegressor:
    """A :class:`TemporalConvNet` with a ``predict`` method, for the phase-3 wrapper.

    Attributes:
        network: the fitted model.
    """

    network: TemporalConvNet

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.network.forward(np.atleast_2d(features))


def fit_sequence(dataset, train_mask: NDArray[np.bool_],
                 validation_mask: NDArray[np.bool_] | None = None,
                 channels: int = 32, kernel: int = 3,
                 dilations: Sequence[int] = (1, 2), hidden: int = 64,
                 learning_rate: float = 3e-3, learning_rate_end: float = 3e-4,
                 epochs: int = 60, batch_size: int = 256, seed: int = 0,
                 name: str = "TCN", progress=None
                 ) -> tuple[LearnedCompensator, list[dict]]:
    """Fit the temporal model, keeping the best epoch by validation error.

    Deliberately the same signature, the same optimiser, the same schedule and
    the same best-epoch rule as
    :func:`qnet_autopilot.compensation.train.fit_mlp`, on the same dataset
    object.  The only difference between the two results is the architecture,
    which is the only way the comparison in the phase-5 report means anything.

    Args:
        dataset: from :func:`qnet_autopilot.compensation.train.build_dataset`.
        train_mask, validation_mask: row selectors from
            :func:`qnet_autopilot.compensation.train.chronological_split`.
        channels, kernel, dilations, hidden: the architecture.
        learning_rate, learning_rate_end: linearly decayed Adam step size.
        epochs: passes over the training rows.
        batch_size: minibatch size.
        seed: initialisation and shuffling.
        name: display name.
        progress: called with a dict per epoch.

    Returns:
        ``(compensator, history)``.

    Raises:
        ValueError: if the window is too short for the dilation stack to be
            causal — a receptive field wider than the window would read padding
            as data.
    """
    spec = SequenceSpec(feature_spec=dataset.spec, channels=channels, kernel=kernel,
                        dilations=tuple(dilations), hidden=hidden)
    if spec.receptive_field > spec.steps:
        raise ValueError(
            f"receptive field {spec.receptive_field} exceeds the {spec.steps} "
            f"available lags; widen FeatureSpec.window or shorten dilations")

    rng = np.random.default_rng(seed)
    standardiser = Standardiser().fit(dataset.features[train_mask])

    train_features = standardiser.transform(dataset.features[train_mask])
    train_targets = dataset.targets[train_mask]
    if validation_mask is not None and validation_mask.any():
        validation_features = standardiser.transform(dataset.features[validation_mask])
        validation_targets = dataset.targets[validation_mask]
    else:
        validation_features = validation_targets = None

    network = TemporalConvNet(spec, rng=seed)
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
            predicted = network.forward(train_features[batch])
            residual = predicted - train_targets[batch]
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

    compensator = LearnedCompensator(model=SequenceRegressor(network), spec=dataset.spec,
                                     standardiser=standardiser, name=name)
    return compensator, history
