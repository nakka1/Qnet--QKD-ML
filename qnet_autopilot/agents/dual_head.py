"""A shared state encoder with a policy head and a value head.

The hypothesis
--------------
Phase 2 ships two entirely separate networks: an actor over
``concat(state, action)`` and a critic over ``state``. They share no weights, so
each learns its own representation of the state from its own loss. The standard
alternative is one **trunk** that encodes the state, read by two heads — and the
argument for it is that the critic's loss is a much denser, better-conditioned
learning signal than the policy gradient (a regression on returns at every
visited state, against an advantage-weighted score at one chosen action), so the
representation it shapes should help the actor too.

The argument against it is equally standard and equally plausible: the two
losses have different scales and different curvature, and a trunk pulled by both
can end up serving neither. Which of the two happens is an empirical question
about this task, which is why it is measured rather than assumed.

What makes it non-trivial here
------------------------------
The two heads do not read the same thing. The critic needs an embedding of the
state; the actor needs one score *per legal action*, from the state and that
action together. So the trunk encodes the state once, and its embedding is
broadcast across the action rows before the actor head sees it — which means the
gradient flowing back into the trunk from the actor is the **sum** over those
rows, and the trunk's total gradient is that sum plus the critic's. Getting that
wrong produces a network that trains, slowly, to something worse, with no
visible symptom; :func:`~qnet_autopilot.agents.numpy_nn.MLP.backward_to_input`
and the finite-difference check in ``tests/test_agents.py`` are what keep it
honest.

One deliberate departure
------------------------
The separate-network policy carries one Adam state per network. This one carries
a single optimiser over trunk and both heads, because the trunk's parameters
receive gradient from both losses and two optimisers stepping the same arrays
would apply the update twice with two different moment estimates. That is a
consequence of sharing, not a free choice, and it is part of what is being
compared.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..env.features import ACTION_FEATURES, STATE_FEATURES
from .normalise import FeatureNormaliser
from .numpy_nn import MLP

__all__ = ["DualHeadPolicy"]


class DualHeadPolicy:
    """Shared state trunk, policy head and value head.

    Implements the same interface as
    :class:`~qnet_autopilot.agents.actor_critic.ActorCriticPolicy`, so
    :func:`~qnet_autopilot.agents.actor_critic.train` drives both without
    knowing which it has — the point being that a difference in results is then
    a difference in architecture and nothing else.

    Attributes:
        trunk: state features to embedding.
        actor_head: ``concat(embedding, action)`` to a scalar score.
        critic_head: embedding to a value estimate.
        embedding: width of the shared representation.
        normaliser, action_normaliser: running feature statistics, or ``None``.
    """

    def __init__(self, trunk: Sequence[int] = (64,), head: Sequence[int] = (64,),
                 embedding: int = 64,
                 rng: np.random.Generator | int | None = None,
                 name: str = "dual_head", normalise: bool = True):
        """
        Args:
            trunk: hidden widths of the shared encoder.
            head: hidden widths of each head.
            embedding: width of the shared representation.
            rng: seed or generator.
            name: display name.
            normalise: fit running feature statistics. On by default here,
                unlike the separate-network policy, because this architecture is
                the refinement rather than the shipped baseline.
        """
        if embedding < 1:
            raise ValueError("embedding must be at least 1 wide")

        generator = np.random.default_rng(rng)
        self.name = name
        self.rng = generator
        self.embedding = int(embedding)

        self.trunk = MLP([STATE_FEATURES, *trunk, embedding], rng=generator)
        self.actor_head = MLP([embedding + ACTION_FEATURES, *head, 1],
                              rng=generator, zero_output=True)
        self.critic_head = MLP([embedding, *head, 1], rng=generator)

        self.greedy = False
        self.temperature = 1.0
        self.normaliser = FeatureNormaliser(STATE_FEATURES) if normalise else None
        self.action_normaliser = (FeatureNormaliser(ACTION_FEATURES) if normalise
                                  else None)

        self._embedding_pre: NDArray[np.float64] | None = None
        self._action_rows: int = 0

    # -- encoding -----------------------------------------------------------

    def _encode_state(self, state_features: NDArray[np.float64]
                      ) -> NDArray[np.float64]:
        if self.normaliser is None:
            return np.asarray(state_features, dtype=float)
        return self.normaliser.transform(state_features)

    def _encode_actions(self, action_features: NDArray[np.float64]
                        ) -> NDArray[np.float64]:
        if self.action_normaliser is None:
            return np.asarray(action_features, dtype=float)
        return self.action_normaliser.transform(action_features)

    def _embed(self, state_batch: NDArray[np.float64]) -> NDArray[np.float64]:
        """Trunk output with a ReLU on top, remembering the pre-activation.

        The ReLU is applied here rather than inside the trunk because
        :class:`~qnet_autopilot.agents.numpy_nn.MLP` ends linear by design. It
        matters: without a nonlinearity at the junction, the trunk's last layer
        and each head's first layer compose to a single linear map, and the
        trunk adds no expressive power over the separate networks it replaces —
        the comparison would then be measuring parameter count, not sharing.
        """
        self._embedding_pre = self.trunk.forward(state_batch)
        return np.maximum(self._embedding_pre, 0.0)

    def _embedding_gradient(self, gradient: NDArray[np.float64]
                            ) -> NDArray[np.float64]:
        """Push a gradient on the embedding back through the ReLU."""
        return gradient * (self._embedding_pre > 0)

    # -- inference ----------------------------------------------------------

    def logits(self, state_features: NDArray[np.float64],
               action_features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score every legal action, caching what :meth:`actor_backward` needs."""
        actions = self._encode_actions(action_features)
        embedding = self._embed(self._encode_state(state_features)[None, :])
        self._action_rows = len(actions)
        repeated = np.broadcast_to(embedding, (self._action_rows, self.embedding))
        return self.actor_head.forward(
            np.concatenate([repeated, actions], axis=1))[:, 0]

    def action_probabilities(self, observation, temperature: float | None = None
                             ) -> NDArray[np.float64]:
        """Softmax over the legal actions at the given temperature."""
        from .numpy_nn import softmax

        state_features, action_features, _ = observation
        logits = self.logits(state_features, action_features)
        scale = self.temperature if temperature is None else temperature
        if scale <= 0:
            probabilities = np.zeros_like(logits)
            probabilities[int(np.argmax(logits))] = 1.0
            return probabilities
        return softmax(logits / scale)

    def act(self, observation) -> int:
        if self.greedy or self.temperature <= 0:
            state_features, action_features, _ = observation
            return int(np.argmax(self.logits(state_features, action_features)))
        probabilities = self.action_probabilities(observation)
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def value(self, state_features: NDArray[np.float64]) -> float:
        embedding = self._embed(self._encode_state(state_features)[None, :])
        return float(self.critic_head.forward(embedding)[0, 0])

    def reset(self) -> None:
        pass

    # -- training protocol --------------------------------------------------

    def observe(self, state_features: NDArray[np.float64],
                action_features: NDArray[np.float64]) -> None:
        if self.normaliser is not None:
            self.normaliser.update(np.atleast_2d(state_features))
        if self.action_normaliser is not None:
            self.action_normaliser.update(action_features)

    def freeze_normalisers(self) -> None:
        for normaliser in (self.normaliser, self.action_normaliser):
            if normaliser is not None:
                normaliser.freeze()

    def parameter_groups(self) -> dict[str, list[NDArray[np.float64]]]:
        """One group, because the trunk is written by both losses.

        Splitting them would put the trunk's arrays in two groups and step them
        twice per update, with two independent Adam moment estimates — which is
        not a shared trunk, it is a race.
        """
        return {"shared": (self.trunk.parameters + self.actor_head.parameters
                           + self.critic_head.parameters)}

    def inference_parameters(self) -> list[NDArray[np.float64]]:
        """Every array that changes what the policy does.

        The trunk is in it, unlike the separate-network case: here the actor
        reads the state *through* the trunk, so restoring a checkpoint's actor
        head onto a different iteration's trunk would restore half a policy.
        """
        return self.trunk.parameters + self.actor_head.parameters

    def _pack(self, trunk_grads, actor_grads, critic_grads
              ) -> dict[str, list[NDArray[np.float64]]]:
        """Lay gradients out in exactly the order :meth:`parameter_groups` uses."""
        zeros = lambda net: [np.zeros_like(array) for array in net.parameters]
        return {"shared": (list(trunk_grads if trunk_grads is not None
                                else zeros(self.trunk))
                           + list(actor_grads if actor_grads is not None
                                  else zeros(self.actor_head))
                           + list(critic_grads if critic_grads is not None
                                  else zeros(self.critic_head)))}

    def actor_backward(self, logit_gradient: NDArray[np.float64]
                       ) -> dict[str, list[NDArray[np.float64]]]:
        """Policy-gradient contribution to the head *and* the trunk.

        The embedding was broadcast across the action rows, so the gradient
        arriving back at it is the sum down those rows — the same embedding
        influenced every score.
        """
        head_grads, input_gradient = self.actor_head.backward_to_input(
            logit_gradient[:, None])
        embedding_gradient = input_gradient[:, :self.embedding].sum(
            axis=0, keepdims=True)
        trunk_grads = self.trunk.backward(
            self._embedding_gradient(embedding_gradient))
        return self._pack(trunk_grads, head_grads, None)

    def critic_forward(self, state_batch: NDArray[np.float64]) -> NDArray[np.float64]:
        encoded = np.stack([self._encode_state(row) for row in state_batch])
        return self.critic_head.forward(self._embed(encoded))

    def critic_backward(self, value_gradient: NDArray[np.float64]
                        ) -> dict[str, list[NDArray[np.float64]]]:
        head_grads, input_gradient = self.critic_head.backward_to_input(value_gradient)
        trunk_grads = self.trunk.backward(self._embedding_gradient(input_gradient))
        return self._pack(trunk_grads, None, head_grads)

    # -- persistence --------------------------------------------------------

    def save(self, path) -> None:
        arrays = {}
        for prefix, net in (("trunk", self.trunk), ("actor", self.actor_head),
                            ("critic", self.critic_head)):
            for index, array in enumerate(net.parameters):
                arrays[f"{prefix}_{index}"] = array
        for prefix, normaliser in (("state_norm", self.normaliser),
                                   ("action_norm", self.action_normaliser)):
            if normaliser is not None:
                for key, value in normaliser.state().items():
                    arrays[f"{prefix}_{key}"] = value
        np.savez(path, **arrays)

    def load(self, path) -> None:
        data = np.load(path)
        for prefix, net in (("trunk", self.trunk), ("actor", self.actor_head),
                            ("critic", self.critic_head)):
            for index, array in enumerate(net.parameters):
                array[...] = data[f"{prefix}_{index}"]
        for prefix, normaliser in (("state_norm", self.normaliser),
                                   ("action_norm", self.action_normaliser)):
            if normaliser is not None and f"{prefix}_count" in data:
                normaliser.load_state({key: data[f"{prefix}_{key}"]
                                       for key in ("count", "mean", "m2")})
                normaliser.freeze()

    def view(self, temperature: float = 1.0, rng=None,
             name: str | None = None) -> "DualHeadPolicy":
        """Another handle on the same weights, at a different temperature."""
        clone = DualHeadPolicy.__new__(DualHeadPolicy)
        clone.name = name or self.name
        clone.rng = self.rng if rng is None else np.random.default_rng(rng)
        clone.trunk = self.trunk
        clone.actor_head = self.actor_head
        clone.critic_head = self.critic_head
        clone.embedding = self.embedding
        clone.normaliser = self.normaliser
        clone.action_normaliser = self.action_normaliser
        clone.temperature = temperature
        clone.greedy = temperature <= 0
        clone._embedding_pre = None
        clone._action_rows = 0
        return clone
