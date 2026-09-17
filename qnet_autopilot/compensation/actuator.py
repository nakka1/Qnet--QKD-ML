"""The device that applies the correction, and the ways it refuses to.

Why an ideal actuator is the wrong model
----------------------------------------
Phase 3 assumed the compensator's requested rotation appears on the fibre
exactly, one control latency later. Real polarisation controllers do not work
that way, and the differences are not second-order:

* **Finite range.** A controller is a cascade of retarders, each with a bounded
  retardance. It cannot follow a fibre that drifts in one direction forever; an
  element walks towards its limit and has to be brought back.
* **Endless reset.** Retardance is periodic: changing an element by a full
  :math:`2\\pi` leaves the device optically identical. That is what makes
  "endless" tracking possible — the element unwinds and carries on. But the
  unwinding takes time, and the states it passes through on the way are not the
  correction the loop asked for. During a reset the link is misaligned.
* **Finite slew rate.** Elements move at a bounded speed, so a large requested
  change arrives late and partially, and a controller that demands large changes
  gets less of what it asked for than one that demands small ones.

Why this is the experiment and not a detail
-------------------------------------------
Phase 3's headline negative was that a neural network does not beat the Wiener
filter, and the explanation was that the drift is a stationary Gaussian process
for which linear prediction is optimal. That argument is about the *mean squared
error of the forecast*. A bounded actuator breaks the link between that error
and the outcome: two predictors with identical MSE are no longer equivalent,
because the one whose commands wander less keeps the elements away from their
limits and resets less often.

Phase 3's own limits section said the reset "is exactly the nonlinearity that
would change the conclusion". This module is that nonlinearity, built so the
claim can be tested rather than repeated.

The parametrisation, and why it is not a rotation vector
--------------------------------------------------------
The state is three **retarder angles**, not the three components of a rotation
vector, and the difference is the whole model. A physical controller — three
fibre squeezers at 0°, 45°, 0°, or an equivalent waveplate cascade — applies

.. math:: R(\\alpha, \\beta, \\gamma)
          = R_x(\\alpha)\\, R_y(\\beta)\\, R_x(\\gamma)

on the Poincare sphere, each element rotating about its own **fixed** axis. In
that parametrisation adding :math:`2\\pi` to any single angle is exactly the
identity, which is what makes an endless reset possible and correct. In a
rotation-vector parametrisation it is not: changing one component by
:math:`2\\pi` changes both the magnitude and the direction of the vector, and a
"reset" written that way would silently move the fibre. Getting this wrong
produces a model that looks reasonable and quietly cheats.

Extracting angles from a requested rotation is Euler decomposition, which has
two branches and a :math:`2\\pi` family within each. A real controller picks the
one nearest where it already is — that is the tracking behaviour — and the cost
of always moving the short way is that the angles random-walk outwards until
they hit the range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ActuatorConfig", "PolarisationActuator", "IdealActuator",
           "retarder_rotation", "retarder_angles"]


def _rx(angle: float) -> NDArray[np.float64]:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _ry(angle: float) -> NDArray[np.float64]:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def retarder_rotation(angles: ArrayLike) -> NDArray[np.float64]:
    """The rotation a three-element cascade applies, from its retardances.

    Args:
        angles: ``(alpha, beta, gamma)`` in radians.

    Returns:
        The Poincare-sphere rotation :math:`R_x(\\alpha) R_y(\\beta)
        R_x(\\gamma)`, shape ``(3, 3)``.
    """
    alpha, beta, gamma = np.asarray(angles, dtype=float)
    return _rx(alpha) @ _ry(beta) @ _rx(gamma)


def retarder_angles(rotation: ArrayLike, near: ArrayLike | None = None
                    ) -> NDArray[np.float64]:
    """Retardances that produce a rotation, chosen nearest the current setting.

    Args:
        rotation: the target rotation, shape ``(3, 3)``.
        near: the controller's current angles. The solution closest to these is
            returned, including across the two Euler branches and the
            :math:`2\\pi` families. ``None`` returns the principal solution.

    Returns:
        ``(alpha, beta, gamma)``.

    Note:
        Near ``beta = 0`` the decomposition is degenerate — only
        ``alpha + gamma`` is determined, the polarisation-controller equivalent
        of gimbal lock. The whole difference is assigned to ``alpha`` and
        ``gamma`` is left where it is, which is what a real controller does:
        with a redundant degree of freedom it does not move the element it does
        not need to move.
    """
    matrix = np.asarray(rotation, dtype=float)
    current = (np.zeros(3) if near is None
               else np.asarray(near, dtype=float).copy())

    cos_beta = float(np.clip(matrix[0, 0], -1.0, 1.0))
    beta = float(np.arccos(cos_beta))

    if abs(np.sin(beta)) < 1e-8:
        # Degenerate: R reduces to a rotation about x alone (beta = 0) or to one
        # composed with a half-turn (beta = pi). Only the sum or difference of
        # alpha and gamma is determined.
        gamma = float(current[2])
        total = float(np.arctan2(matrix[2, 1], matrix[1, 1]))
        alpha = total - gamma if cos_beta > 0 else total + gamma
        candidates = [np.array([alpha, beta, gamma])]
    else:
        alpha = float(np.arctan2(matrix[1, 0], -matrix[2, 0]))
        gamma = float(np.arctan2(matrix[0, 1], matrix[0, 2]))
        candidates = [np.array([alpha, beta, gamma]),
                      np.array([alpha + np.pi, -beta, gamma + np.pi])]

    # Within each branch, slide every angle by whole turns to sit nearest the
    # current setting. Both branches then compete on total travel.
    best, best_cost = None, np.inf
    for candidate in candidates:
        unwrapped = candidate + 2.0 * np.pi * np.round(
            (current - candidate) / (2.0 * np.pi))
        cost = float(np.sum(np.abs(unwrapped - current)))
        if cost < best_cost:
            best, best_cost = unwrapped, cost
    return best


@dataclass(frozen=True)
class ActuatorConfig:
    """Bounds and speeds of a three-element polarisation controller.

    Attributes:
        range_rad: how far each retardance may travel from zero before it must
            reset. ``inf`` gives an unbounded controller, which is phase 3's
            original assumption and the control condition for the experiment.
            A multi-element endless controller has a few turns of travel; the
            default is four.
        slew_rate_rad_per_s: how fast an element moves. This is the parameter
            that separates device classes, and it spans four orders of
            magnitude in practice: a motorised waveplate manages tens of
            radians per second, a piezo fibre squeezer hundreds to thousands,
            and a lithium-niobate controller tens of thousands. The default is
            a fast piezo device.

            It matters here more than the range does, and the arithmetic says
            why: tracking a vibration of amplitude *A* at frequency *f* needs a
            peak rate of :math:`2 \pi f A`, which for phase 3's operating point
            — 0.30 rad at 40 Hz — is about 75 rad/s. A device below that cannot
            follow the fibre however much travel it has.
        reset_slew_multiplier: resets usually run faster than tracking, because
            nothing is being tracked during one.
    """

    range_rad: float = 4.0 * np.pi
    slew_rate_rad_per_s: float = 2000.0
    reset_slew_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.range_rad <= 0:
            raise ValueError("range_rad must be positive")
        if self.slew_rate_rad_per_s <= 0:
            raise ValueError("slew_rate_rad_per_s must be positive")
        if self.reset_slew_multiplier <= 0:
            raise ValueError("reset_slew_multiplier must be positive")
        if np.isfinite(self.range_rad) and self.range_rad < 2.0 * np.pi:
            raise ValueError(
                "range_rad below one full turn: the device could never complete "
                "a reset, so it is not an endless controller")


class IdealActuator:
    """Applies exactly what it is asked for. Phase 3's original assumption.

    Present so the bounded actuator has a control that runs through the same
    code path, rather than being compared against a different branch of the loop.
    """

    name = "atuador ideal"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.resets = 0
        self.resetting_steps = 0
        self.travel_rad = 0.0

    def apply(self, requested: ArrayLike, dt_s: float) -> NDArray[np.float64]:
        return np.asarray(requested, dtype=float)


class PolarisationActuator:
    """Three bounded, rate-limited retarders, with endless reset.

    Args:
        config: bounds and speeds.

    Attributes:
        angles: current retardances, shape ``(3,)``.
        resets: resets started this run.
        resetting_steps: control steps spent with at least one element
            unwinding — the time the link spent misaligned by the device rather
            than by the fibre.
        travel_rad: total distance every element has moved. This is the
            mechanism behind everything else: a compensator whose commands
            wander demands more travel, gets less of what it asked for from a
            rate-limited device, and reaches its range sooner.
    """

    name = "atuador com faixa finita"

    def __init__(self, config: ActuatorConfig | None = None) -> None:
        self.config = config or ActuatorConfig()
        self.reset()

    def reset(self) -> None:
        self.angles = np.zeros(3)
        self._resetting = np.zeros(3, dtype=bool)
        self._reset_target = np.zeros(3)
        self.resets = 0
        self.resetting_steps = 0
        self.travel_rad = 0.0

    def apply(self, requested: ArrayLike, dt_s: float) -> NDArray[np.float64]:
        """Move towards the requested rotation and return what is on the fibre.

        Args:
            requested: the rotation the compensator asked for, shape ``(3, 3)``.
            dt_s: how long the elements have to move.

        Returns:
            The rotation actually applied after this step, shape ``(3, 3)``.
        """
        config = self.config
        target = retarder_angles(requested, near=self.angles)

        step = config.slew_rate_rad_per_s * dt_s
        reset_step = step * config.reset_slew_multiplier
        before = self.angles.copy()

        for axis in range(3):
            if self._resetting[axis]:
                remaining = self._reset_target[axis] - self.angles[axis]
                self.angles[axis] += np.clip(remaining, -reset_step, reset_step)
                if abs(self.angles[axis] - self._reset_target[axis]) < 1e-9:
                    self.angles[axis] = self._reset_target[axis]
                    self._resetting[axis] = False
                continue

            self.angles[axis] += np.clip(target[axis] - self.angles[axis],
                                         -step, step)

            if abs(self.angles[axis]) >= config.range_rad:
                # Out of travel. Unwind a full turn: optically identical at the
                # end, and wrong all the way there.
                self._resetting[axis] = True
                self._reset_target[axis] = (self.angles[axis]
                                            - np.sign(self.angles[axis])
                                            * 2.0 * np.pi)
                self.resets += 1

        if self._resetting.any():
            self.resetting_steps += 1
        self.travel_rad += float(np.sum(np.abs(self.angles - before)))

        return retarder_rotation(self.angles)
