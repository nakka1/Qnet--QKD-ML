"""A small neural network with hand-written backpropagation, and Adam.

Two hundred lines of numpy instead of a framework, because the container this
was developed in cannot install one and *an untrained agent is not a result*.
The scale makes it reasonable: the networks here are two hidden layers of 64
units over a 34-dimensional input, and an episode is a hundred steps.  numpy
runs that at a few thousand steps a second, which is enough to train to
convergence in minutes.

The parts that are easy to get wrong, and how they are handled:

* **Backprop.**  Written out per layer, and checked against finite differences
  in ``tests/test_agents.py``.  A hand-derived gradient that is subtly wrong
  still trains — slowly, to a worse optimum — and is nearly undetectable by
  watching a learning curve, so the numerical check is not optional.
* **Initialisation.**  He scaling for ReLU, and the policy head's last layer
  starts at zero so the initial policy is exactly uniform over legal actions.
  A random last layer means the agent starts with arbitrary strong preferences
  and spends its first thousand episodes unlearning them.
* **Adam.**  Including bias correction, which matters most in exactly the first
  few hundred updates where a policy-gradient run decides what to explore.

:class:`MLP` is deliberately not a general framework.  It does the one thing
the trainer needs — forward with cached activations, backward from an output
gradient — and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["MLP", "Adam", "softmax", "categorical_entropy"]


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def categorical_entropy(probabilities: NDArray[np.float64]) -> float:
    """Shannon entropy in nats, guarding the log at zero."""
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


class MLP:
    """Fully connected network with ReLU hidden layers and a linear output.

    Attributes:
        sizes: layer widths, input first and output last.
        weights, biases: the parameters, in layer order.
    """

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator | int | None = None,
                 zero_output: bool = False):
        """
        Args:
            sizes: e.g. ``[34, 64, 64, 1]``.
            rng: seed or generator.
            zero_output: initialise the final layer to zero.  For a policy head
                this makes the starting policy exactly uniform, which is the
                right prior when every action is legal by construction and the
                agent has no reason yet to prefer any of them.
        """
        if len(sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output layer")

        generator = np.random.default_rng(rng)
        self.sizes = list(sizes)
        self.weights: list[NDArray[np.float64]] = []
        self.biases: list[NDArray[np.float64]] = []

        for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = index == len(sizes) - 2
            if last and zero_output:
                weight = np.zeros((fan_in, fan_out))
            else:
                # He initialisation: variance 2/fan_in keeps activation scale
                # stable through ReLU layers.
                weight = generator.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            self.weights.append(weight)
            self.biases.append(np.zeros(fan_out))

        self._cache: list[NDArray[np.float64]] = []

    # -- parameters ---------------------------------------------------------

    @property
    def parameters(self) -> list[NDArray[np.float64]]:
        """Every array, in a fixed order, for the optimiser to update in place."""
        return [array for pair in zip(self.weights, self.biases) for array in pair]

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

    # -- forward and backward ----------------------------------------------

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run the network, caching what backward needs.

        Args:
            inputs: shape ``(batch, sizes[0])``.

        Returns:
            Shape ``(batch, sizes[-1])``.
        """
        activation = np.asarray(inputs, dtype=float)
        if activation.ndim == 1:
            activation = activation[None, :]

        self._cache = [activation]
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_activation = activation @ weight + bias
            activation = pre_activation if index == last else np.maximum(pre_activation, 0.0)
            self._cache.append(activation)
        return activation

    def backward(self, output_gradient: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Gradients of the loss with respect to every parameter.

        Args:
            output_gradient: dL/d(output), shape ``(batch, sizes[-1])``.

        Returns:
            One array per entry of :attr:`parameters`, same shapes.

        Raises:
            RuntimeError: if called before :meth:`forward`.
        """
        if not self._cache:
            raise RuntimeError("call forward() before backward()")

        gradient = np.asarray(output_gradient, dtype=float)
        if gradient.ndim == 1:
            gradient = gradient[:, None]

        weight_grads: list[NDArray[np.float64]] = [None] * len(self.weights)  # type: ignore
        bias_grads: list[NDArray[np.float64]] = [None] * len(self.biases)  # type: ignore

        for index in range(len(self.weights) - 1, -1, -1):
            activation_in = self._cache[index]
            weight_grads[index] = activation_in.T @ gradient
            bias_grads[index] = gradient.sum(axis=0)
            if index > 0:
                gradient = gradient @ self.weights[index].T
                # ReLU derivative, read off the cached post-activation.
                gradient = gradient * (self._cache[index] > 0)

        return [array for pair in zip(weight_grads, bias_grads) for array in pair]

    def backward_to_input(self, output_gradient: NDArray[np.float64]
                          ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        """Parameter gradients **and** the gradient with respect to the input.

        :meth:`backward` stops at the first layer's weights, which is all a
        standalone network needs. A network stacked on top of another one needs
        one more step: the gradient that flows *into* its input, so the network
        below can continue the chain.

        Args:
            output_gradient: dL/d(output), shape ``(batch, sizes[-1])``.

        Returns:
            ``(parameter_gradients, input_gradient)``. The parameter list is
            exactly what :meth:`backward` returns, in the same order; the input
            gradient has the shape of the forward pass's input.

        Raises:
            RuntimeError: if called before :meth:`forward`.

        Note:
            Kept separate from :meth:`backward` rather than folded into it with
            a flag: every existing caller passes a gradient and unpacks a flat
            list, and changing that return type would break them silently
            rather than loudly.
        """
        if not self._cache:
            raise RuntimeError("call forward() before backward_to_input()")

        gradient = np.asarray(output_gradient, dtype=float)
        if gradient.ndim == 1:
            gradient = gradient[:, None]

        weight_grads: list[NDArray[np.float64]] = [None] * len(self.weights)  # type: ignore
        bias_grads: list[NDArray[np.float64]] = [None] * len(self.biases)  # type: ignore

        for index in range(len(self.weights) - 1, -1, -1):
            activation_in = self._cache[index]
            weight_grads[index] = activation_in.T @ gradient
            bias_grads[index] = gradient.sum(axis=0)
            gradient = gradient @ self.weights[index].T
            if index > 0:
                # ReLU derivative, read off the cached post-activation. The
                # input layer has no activation to differentiate, which is why
                # this is inside the branch and the matmul above is not.
                gradient = gradient * (self._cache[index] > 0)

        parameters = [array for pair in zip(weight_grads, bias_grads) for array in pair]
        return parameters, gradient

    # -- serialisation ------------------------------------------------------

    def __getstate__(self) -> dict:
        """Pickle the weights, never the activation cache.

        ``forward`` keeps every layer's activations so ``backward`` can use
        them, and after one pass over a validation set that cache is the whole
        dataset times the hidden width — megabytes to gigabytes, none of it part
        of the model. Saved checkpoints were 200x larger than their weights
        before this existed, and a restored network would have carried someone
        else's activations into its first ``backward`` call.
        """
        state = dict(self.__dict__)
        state["_cache"] = []
        return state


@dataclass
class Adam:
    """Adam, with bias correction and optional global gradient-norm clipping.

    Attributes:
        learning_rate: step size.
        beta1, beta2: moment decay rates.
        epsilon: denominator floor.
        clip_norm: clip the global gradient norm to this before stepping.
            Policy gradients on sparse-reward tasks produce occasional enormous
            updates when a rare success lands; without clipping one of them can
            destroy a policy that took thousands of episodes to build.
    """

    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    clip_norm: float | None = 1.0

    _first: list[NDArray[np.float64]] = field(default_factory=list, repr=False)
    _second: list[NDArray[np.float64]] = field(default_factory=list, repr=False)
    _step: int = field(default=0, repr=False)

    def step(self, parameters: Sequence[NDArray[np.float64]],
             gradients: Sequence[NDArray[np.float64]]) -> float:
        """Update ``parameters`` in place.

        Returns:
            The global gradient norm before clipping — worth logging, because a
            norm that grows without bound is the first sign a policy-gradient
            run is about to collapse.
        """
        if not self._first:
            self._first = [np.zeros_like(array) for array in parameters]
            self._second = [np.zeros_like(array) for array in parameters]

        total_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in gradients)))
        scale = 1.0
        if self.clip_norm is not None and total_norm > self.clip_norm:
            scale = self.clip_norm / (total_norm + 1e-12)

        self._step += 1
        bias_correction1 = 1.0 - self.beta1 ** self._step
        bias_correction2 = 1.0 - self.beta2 ** self._step

        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            gradient = gradient * scale
            self._first[index] = self.beta1 * self._first[index] + (1 - self.beta1) * gradient
            self._second[index] = (self.beta2 * self._second[index]
                                   + (1 - self.beta2) * gradient * gradient)
            corrected_first = self._first[index] / bias_correction1
            corrected_second = self._second[index] / bias_correction2
            parameter -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + self.epsilon)

        return total_norm
