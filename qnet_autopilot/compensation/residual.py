"""What a closed polarisation loop leaves behind, as a process a planner can use.

Why not just reuse the phase-1 drift model
------------------------------------------
:class:`~qnet_autopilot.physics.fiber.PolarizationDriftModel` is a model of an
**uncompensated** fibre, and it random-walks: the state of polarisation wanders
over the whole Poincare sphere and the QBER it implies climbs to a coin flip
and stays there. That is correct for a bare fibre and wrong for a deployed one,
because a deployed link runs the compensator of phase 3 underneath. Point-
sampling that model every few seconds — the cadence a traffic allocator runs at
— also aliases the acoustic band into something that looks like slow structure
and is not, which is a mistake phase 1 already had to fix once.

What a closed loop leaves is a different process with different properties, and
the differences are exactly the ones an allocator cares about:

* it is **mean-reverting**, not diffusive — the loop pulls the residual back
  towards zero, so a bad minute is followed by a good one rather than by a
  worse one;
* its QBER distribution is **skewed with a long right tail**, because the
  residual angle has two independent perpendicular components and the error
  goes as their square — most of the time it is small, and occasionally it is
  several times its own mean;
* it has **transients**: a cable jump displaces the polarisation instantly and
  the loop takes time to catch it, which is the one excursion no forecaster can
  anticipate and every allocator has to survive.

The model
---------
Two perpendicular components of the residual angle, each an Ornstein-Uhlenbeck
process, plus a jump term that decays back with the loop's recovery time:

.. math::

    \\theta_i(t) &\\sim \\text{OU}(0,\\ \\sigma_\\theta,\\ \\tau), \\quad i = 1, 2 \\\\
    e(t) &= \\sin^2\\!\\bigl(\\lVert\\theta(t)\\rVert / 2\\bigr)

The square is what produces the tail, and it is not a modelling flourish: it is
the same geometry that makes :func:`~qnet_autopilot.physics.fiber.stokes_qber`
quadratic in the misalignment for small angles.

Calibration
-----------
The defaults are set from phase 3's own measurements rather than from the
literature. :data:`COMPENSATED` reproduces the residual the ridge compensator
achieved at the phase-3 operating point — mean QBER near 0.8%, 95th percentile
near 2% — and :data:`DEGRADED` represents a link whose loop is running but
struggling, which on a real plant is the aerial span, the one with the bad
splice, or the one whose actuator is near its reset. Both are arguments, and
phase 4 sweeps between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ResidualDriftModel", "COMPENSATED", "DEGRADED"]


@dataclass(frozen=True)
class ResidualDriftModel:
    """Residual alignment error of a link running a polarisation compensator.

    Attributes:
        rms_angle_rad: steady-state RMS of each of the two perpendicular
            residual components. The QBER's mean is roughly
            ``rms_angle_rad**2 / 2`` for small angles.
        correlation_time_s: how long the residual stays correlated with itself.
            This is the quantity that decides whether forecasting is possible
            at all: much shorter than the control period and the process is
            white noise to the allocator; much longer and persistence already
            answers it.
        jump_rate_per_s: Poisson rate of cable disturbances the loop has to
            chase.
        jump_scale_rad: mean angular size of a jump, exponentially distributed.
        recovery_time_s: how fast the loop pulls a jump back out. Phase 3
            measured a median of about a millisecond for the learned
            compensator and eleven for the tuned classical one, but a jump
            large enough to drive the actuator off its operating point takes
            far longer, and it is those that an allocator sees at all.
    """

    rms_angle_rad: float = 0.12
    correlation_time_s: float = 120.0
    jump_rate_per_s: float = 1.0 / 600.0
    jump_scale_rad: float = 0.30
    recovery_time_s: float = 25.0

    def __post_init__(self) -> None:
        if self.rms_angle_rad < 0:
            raise ValueError("rms_angle_rad cannot be negative")
        if self.correlation_time_s <= 0:
            raise ValueError("correlation_time_s must be positive")
        if self.recovery_time_s <= 0:
            raise ValueError("recovery_time_s must be positive")
        if self.jump_rate_per_s < 0:
            raise ValueError("jump_rate_per_s cannot be negative")

    def simulate(self, duration_s: float, dt_s: float,
                 rng: np.random.Generator | int | None = None
                 ) -> NDArray[np.float64]:
        """Residual QBER over time.

        Args:
            duration_s: total simulated time.
            dt_s: sampling interval — the allocator's control period.
            rng: seed or generator.

        Returns:
            QBER at each sample, shape ``(steps,)``, in ``[0, 1]``.

        Note:
            The OU part is discretised **exactly** (the stationary
            autoregressive form, not an Euler step), so the sampled process has
            the right variance and the right autocorrelation at any ``dt_s``.
            An Euler discretisation at a 5-second step against a 120-second
            correlation time would be close, but "close" here would quietly
            change how forecastable the process is, which is the thing being
            measured.
        """
        generator = np.random.default_rng(rng)
        steps = int(round(duration_s / dt_s)) + 1

        decay = float(np.exp(-dt_s / self.correlation_time_s))
        innovation = self.rms_angle_rad * np.sqrt(max(1.0 - decay ** 2, 0.0))

        components = np.empty((steps, 2))
        components[0] = generator.normal(0.0, self.rms_angle_rad, size=2)
        noise = generator.normal(0.0, innovation, size=(steps - 1, 2))
        for step in range(1, steps):
            components[step] = decay * components[step - 1] + noise[step - 1]

        angle = np.linalg.norm(components, axis=1) + self._jump_series(
            steps, dt_s, generator)
        return np.clip(np.sin(angle / 2.0) ** 2, 0.0, 1.0)

    def _jump_series(self, steps: int, dt_s: float,
                     generator: np.random.Generator) -> NDArray[np.float64]:
        """Displacements from cable jumps, each decaying at the loop's rate."""
        series = np.zeros(steps)
        if self.jump_rate_per_s <= 0 or self.jump_scale_rad <= 0:
            return series

        expected = self.jump_rate_per_s * dt_s * steps
        count = generator.poisson(expected)
        if count == 0:
            return series

        times = generator.integers(0, steps, size=count)
        sizes = generator.exponential(self.jump_scale_rad, size=count)
        decay = np.exp(-np.arange(steps) * dt_s / self.recovery_time_s)
        for start, size in zip(times, sizes):
            series[start:] += size * decay[:steps - start]
        return series


#: A link whose compensator is working, calibrated to phase 3's measured
#: residual at its main operating point (mean QBER ~0.8%, p95 ~2%).
COMPENSATED = ResidualDriftModel(rms_angle_rad=0.112, correlation_time_s=150.0,
                                 jump_rate_per_s=1.0 / 900.0, jump_scale_rad=0.20,
                                 recovery_time_s=20.0)

#: A link whose loop is running but losing — an aerial span, a bad splice, an
#: actuator near its reset. Same process, several times the residual.
DEGRADED = ResidualDriftModel(rms_angle_rad=0.20, correlation_time_s=90.0,
                              jump_rate_per_s=1.0 / 400.0, jump_scale_rad=0.35,
                              recovery_time_s=40.0)
