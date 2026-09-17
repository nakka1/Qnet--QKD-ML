"""Running a compensator against a drifting fibre, and scoring it honestly.

Why this is a closed loop and not a regression
----------------------------------------------
It is tempting to score a polarisation predictor by the RMSE of its forecast.
That number is easy to compute and answers the wrong question.  What a link
operator cares about is the QBER *after* correction and the secret key that
survives it, and those are not monotone in RMSE: a predictor that is slightly
worse on average but recovers in two control periods after a cable jump beats
one that is smoother and takes twenty, because BB84 produces **no key at all**
above roughly 11% QBER and a long excursion past that threshold costs the whole
window.

So the compensator is run in the loop: it sees only what the monitoring channel
gives it, its correction is applied after the control latency, and the score is
the residual rotation, the QBER it implies, and the key rate that follows.

The latency is the whole problem
--------------------------------
With zero control latency, "correct to the last measurement" is optimal and
there is nothing to predict.  Real polarisation controllers have a loop delay —
measurement integration, computation, the electro-optic actuator's own response
— and during it the fibre keeps moving.  Prediction earns its keep exactly in
proportion to that delay, which is why :func:`run_closed_loop` takes it as a
parameter and why the report sweeps it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.fiber import PolarizationTrajectory, rotation_angle
from ..physics.link import bb84_finite_key_rate, bb84_secret_key_rate
from .actuator import IdealActuator
from .monitoring import MonitoringConfig, measure_rotation

__all__ = ["ControlConfig", "Compensator", "LoopResult", "run_closed_loop",
           "bb84_qber", "BB84_BASES"]

#: The two mutually unbiased Stokes axes BB84 encodes on.
BB84_BASES = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@runtime_checkable
class Compensator(Protocol):
    """Anything that turns monitoring history into a correction.

    Implementations are stateful across a run and must be reset between runs.
    """

    name: str

    def reset(self) -> None:
        """Clear any history at the start of a run."""

    def predict(self, time_s: float, measured: NDArray[np.float64],
                covariates: NDArray[np.float64], horizon_s: float
                ) -> NDArray[np.float64]:
        """Predict the fibre rotation ``horizon_s`` into the future.

        Args:
            time_s: now.
            measured: the current noisy rotation estimate, shape ``(3, 3)``.
            covariates: environmental sensors — temperature, its rate, and the
                accelerometer reading.
            horizon_s: how far ahead the correction has to be valid.

        Returns:
            The predicted rotation, shape ``(3, 3)``.  The loop applies its
            transpose as the correction.
        """


@dataclass(frozen=True)
class ControlConfig:
    """The control loop's timing and its monitoring channel.

    Attributes:
        control_period_s: how often the compensator updates. Must be a multiple
            of the trajectory's sampling interval.
        latency_s: delay between a measurement and the correction derived from
            it taking effect. Zero makes prediction pointless, which is the
            right sanity check and the wrong operating point. Must be an
            integer multiple of ``control_period_s`` — see the note below.
        monitoring: the probe measurement setup.
        sifted_rate_hz: sifted key rate at zero QBER, for converting the
            residual error into a secret key rate.
    """

    control_period_s: float = 1.0e-3
    latency_s: float = 2.0e-3
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    sifted_rate_hz: float = 1.0e5

    def __post_init__(self) -> None:
        if self.control_period_s <= 0:
            raise ValueError("control_period_s must be positive")
        if self.latency_s < 0:
            raise ValueError("latency_s cannot be negative")

        # A predictor is trained to forecast a whole number of control steps
        # ahead, because those are the only times it has measurements for. If
        # the latency is not a multiple of the control period, the loop applies
        # the correction at a moment the model was never trained to predict,
        # and the mismatch shows up as a silent loss of accuracy that looks
        # like the predictor simply being worse at short latencies. Refusing
        # the configuration is better than absorbing the error.
        steps = self.latency_s / self.control_period_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"latency_s ({self.latency_s * 1e3:g} ms) must be an integer "
                f"multiple of control_period_s ({self.control_period_s * 1e3:g} ms); "
                f"got {steps:.3f} steps")


def bb84_qber(residual: ArrayLike, bases: ArrayLike = BB84_BASES) -> NDArray[np.float64]:
    """QBER from a residual rotation, averaged over BB84's two bases.

    For a basis whose Stokes axis is :math:`\\hat b`, a residual rotation
    :math:`E` sends it to :math:`E\\hat b` and the bit error probability is
    :math:`(1 - \\hat b \\cdot E\\hat b)/2`.  BB84 sifts over two bases, so the
    observed QBER is the average of the two.

    Args:
        residual: residual rotation(s), shape ``(..., 3, 3)``.
        bases: Stokes axes of the encoding bases, shape ``(k, 3)``.

    Returns:
        QBER in ``[0, 1]``, shape ``(...)``.
    """
    residual = np.asarray(residual, dtype=float)
    bases = np.asarray(bases, dtype=float)
    rotated = np.einsum("...ij,kj->...ki", residual, bases)
    alignment = np.einsum("...ki,ki->...k", rotated, bases)
    return np.mean((1.0 - alignment) / 2.0, axis=-1)


@dataclass
class LoopResult:
    """Per-sample outcome of one closed-loop run.

    Attributes:
        time_s: sample times.
        residual_angle_rad: rotation angle of the residual error — the bound on
            how badly *any* polarisation state is misaligned, which is what a
            compensator is accountable for. The misalignment of one chosen
            state can be zero while the residual rotation is large.
        qber: BB84 QBER implied by the residual.
        secret_key_rate_hz: what survives error correction and privacy
            amplification. Zero above the threshold, and that is the point.
        prediction_error_rad: angle between the predicted and true rotation —
            the open-loop regression metric, kept for comparison but not the
            score.
        jump_times_s: when the fibre jumped, for the recovery analysis.
        name: the compensator's name.
        actuator_resets: how many times a retarder ran out of travel and had to
            unwind. Zero for an ideal actuator.
        actuator_resetting_fraction: share of samples during which at least one
            element was unwinding, and the rotation on the fibre was therefore
            not the one the compensator asked for.
        actuator_travel_rad: total distance the elements moved. The mechanism
            behind the other two: wandering commands cost travel, and travel is
            what a rate-limited device runs out of.
    """

    time_s: NDArray[np.float64]
    residual_angle_rad: NDArray[np.float64]
    qber: NDArray[np.float64]
    secret_key_rate_hz: NDArray[np.float64]
    prediction_error_rad: NDArray[np.float64]
    jump_times_s: NDArray[np.float64]
    name: str = ""
    actuator_resets: int = 0
    actuator_resetting_fraction: float = 0.0
    actuator_travel_rad: float = 0.0

    def finite_key_rate_hz(self, sifted_rate_hz: float,
                           block_duration_s: float = 1.0) -> float:
        """Mean secret rate with **finite** blocks, not the asymptotic ceiling.

        The asymptotic formula assumes the error rate is known exactly and
        privacy amplification is free. A real link settles up on a schedule and
        pays for both. Here the run is cut into blocks of ``block_duration_s``,
        each block's own mean QBER is used, and the finite-key length is
        evaluated per block — which is what a deployed system would actually
        extract.

        Args:
            sifted_rate_hz: the link's sifted rate.
            block_duration_s: how long a block accumulates before processing.

        Returns:
            Mean secret bits per second over the run.

        Note:
            Blocking by QBER *within* the run matters and is not the same as
            applying the formula to the run's mean error. A block that spends
            its time above the threshold yields nothing, and averaging the error
            first would hide that inside a middling number — the same reason the
            phase-3 report plots QBER per sifting block rather than per sample.
        """
        if len(self.time_s) < 2:
            return 0.0
        dt_s = float(self.time_s[1] - self.time_s[0])
        per_block = max(int(round(block_duration_s / dt_s)), 1)
        usable = len(self.qber) // per_block * per_block
        if usable == 0:
            return 0.0

        blocked = self.qber[:usable].reshape(-1, per_block).mean(axis=1)
        rates = bb84_finite_key_rate(sifted_rate_hz, blocked,
                                     per_block * dt_s)
        return float(np.mean(rates))

    def summary(self, qber_threshold: float = 0.11) -> dict[str, float]:
        """The numbers worth quoting, in the order they matter."""
        return {
            "compensator": self.name,
            "mean_qber": float(np.mean(self.qber)),
            "median_qber": float(np.median(self.qber)),
            "p95_qber": float(np.percentile(self.qber, 95)),
            "uptime": float(np.mean(self.qber < qber_threshold)),
            "mean_key_rate_hz": float(np.mean(self.secret_key_rate_hz)),
            "mean_residual_mrad": float(np.mean(self.residual_angle_rad) * 1e3),
            "p95_residual_mrad": float(np.percentile(self.residual_angle_rad, 95) * 1e3),
            "mean_prediction_error_mrad": float(np.mean(self.prediction_error_rad) * 1e3),
            "actuator_resets": float(self.actuator_resets),
            "actuator_resetting_fraction": float(self.actuator_resetting_fraction),
            "actuator_travel_rad": float(self.actuator_travel_rad),
        }

    def recovery_times_s(self, qber_threshold: float = 0.11,
                         settle: float = 0.02) -> list[float]:
        """How long after each jump the link takes to come back below threshold.

        A jump is the one thing no predictor can anticipate, so this measures
        the property that actually separates compensators in the tail: not how
        well they track, but how fast they recover.

        Args:
            qber_threshold: the QBER a link must get back under.
            settle: also require the QBER to stay under the threshold for this
                many seconds, so a single lucky sample does not count as
                recovery.

        Returns:
            One recovery time per jump, in seconds.  ``inf`` where the link
            never recovered before the run ended.
        """
        recoveries: list[float] = []
        healthy = self.qber < qber_threshold
        for jump in self.jump_times_s:
            start = int(np.searchsorted(self.time_s, jump))
            if start >= len(self.time_s):
                continue
            recovered = float("inf")
            for index in range(start, len(self.time_s)):
                if not healthy[index]:
                    continue
                window_end = np.searchsorted(self.time_s, self.time_s[index] + settle)
                if bool(np.all(healthy[index:window_end])):
                    recovered = float(self.time_s[index] - jump)
                    break
            recoveries.append(recovered)
        return recoveries


def run_closed_loop(trajectory: PolarizationTrajectory,
                    compensator: Compensator,
                    control: ControlConfig | None = None,
                    rng: np.random.Generator | int | None = None,
                    actuator=None) -> LoopResult:
    """Drive a compensator against a simulated fibre and score the result.

    The loop, per control step *k* at time :math:`t_k`:

    1. the monitoring channel produces a noisy estimate of :math:`R(t_k)`;
    2. the compensator predicts :math:`R(t_k + \\tau)` from that and its history;
    3. the correction :math:`C_k = \\hat R^{\\mathsf T}` takes effect at
       :math:`t_k + \\tau` and stays in force until the next one lands;
    4. at every sample in between, the residual is :math:`C R(t)`.

    Args:
        trajectory: a run from
            :meth:`~qnet_autopilot.physics.fiber.PolarizationDriftModel.simulate`
            with ``track_rotation=True``.
        compensator: the policy under test.
        control: loop timing and monitoring; defaults to 1 kHz with 2 ms latency.
        rng: seed or generator for the measurement noise.
        actuator: the device that applies the correction. ``None`` keeps phase
            3's original assumption — the requested rotation appears exactly —
            and takes the vectorised path. Passing a
            :class:`~.actuator.PolarisationActuator` makes the correction
            bounded and rate-limited, and forces the sequential path, which is
            about ten times slower because the applied rotation then depends on
            every step before it.

    Returns:
        LoopResult: per-sample residuals, QBER and key rate.

    Raises:
        ValueError: if the trajectory has no rotation history — a compensator
            cannot be scored against a single tracked state (see
            :mod:`.monitoring`).
    """
    if trajectory.rotation is None:
        raise ValueError(
            "this trajectory tracked only one launch state, which does not "
            "determine the fibre's rotation. Re-simulate with track_rotation=True.")

    control = control or ControlConfig()
    generator = np.random.default_rng(rng)

    time_s = trajectory.time_s
    rotations = trajectory.rotation
    sample_count = len(time_s)
    dt_s = float(time_s[1] - time_s[0]) if sample_count > 1 else 1.0

    step_stride = max(int(round(control.control_period_s / dt_s)), 1)
    latency_steps = int(round(control.latency_s / dt_s))
    horizon_s = latency_steps * dt_s

    covariates = np.stack([
        trajectory.temperature_c,
        np.gradient(trajectory.temperature_c, dt_s),
        trajectory.vibration_rms,
    ], axis=-1)

    # Measure once for the whole run: the noise is independent per update, so
    # drawing it in a batch is equivalent to drawing it in the loop and far
    # faster. Only the indices the loop visits are actually consumed.
    update_indices = np.arange(0, sample_count, step_stride)
    measured = measure_rotation(rotations[update_indices], control.monitoring,
                                rng=generator)

    # The compensator is stateful, so its updates are a genuine sequential loop —
    # but only over control steps, not over samples. Which correction is in force
    # at each sample follows from the update times, and that part is vectorised:
    # a 30-second run at 1 kHz is 30,000 samples and 15,000 updates, and doing
    # the bookkeeping per sample doubled the cost of every experiment.
    effective = np.minimum(update_indices + latency_steps, sample_count - 1)
    predictions = np.empty((len(update_indices), 3, 3))

    compensator.reset()
    for step, index in enumerate(update_indices):
        predictions[step] = np.asarray(
            compensator.predict(float(time_s[index]), measured[step],
                                covariates[index], horizon_s), dtype=float)

    # Open-loop error, at the moment each prediction takes effect.
    prediction_error_at_update = rotation_angle(
        np.einsum("kji,kjl->kil", predictions, rotations[effective]))

    in_force = np.searchsorted(effective, np.arange(sample_count), side="right") - 1
    corrections = np.empty((sample_count, 3, 3))
    uncorrected = in_force < 0
    corrections[uncorrected] = np.eye(3)
    corrections[~uncorrected] = np.transpose(
        predictions[in_force[~uncorrected]], (0, 2, 1))

    resets = 0
    resetting_fraction = 0.0
    travel = 0.0
    if actuator is not None and not isinstance(actuator, IdealActuator):
        # A bounded device's output depends on where it already is, so the
        # correction in force at each sample has to be walked forward in order.
        # Only the *achieved* rotation changes here; what the compensator asked
        # for, and when, is untouched.
        actuator.reset()
        achieved = np.empty_like(corrections)
        for index in range(sample_count):
            achieved[index] = actuator.apply(corrections[index], dt_s)
        corrections = achieved
        resets = int(actuator.resets)
        resetting_fraction = actuator.resetting_steps / max(sample_count, 1)
        travel = float(actuator.travel_rad)

    prediction_error = np.zeros(sample_count)
    prediction_error[~uncorrected] = prediction_error_at_update[in_force[~uncorrected]]

    residual = np.einsum("nij,njk->nik", corrections, rotations)
    residual_angle = rotation_angle(residual)
    qber = bb84_qber(residual)
    key_rate = bb84_secret_key_rate(control.sifted_rate_hz, qber)

    return LoopResult(time_s=time_s, residual_angle_rad=residual_angle, qber=qber,
                      secret_key_rate_hz=np.asarray(key_rate, dtype=float),
                      prediction_error_rad=prediction_error,
                      jump_times_s=trajectory.jump_times_s,
                      name=getattr(compensator, "name", type(compensator).__name__),
                      actuator_resets=resets,
                      actuator_resetting_fraction=resetting_fraction,
                      actuator_travel_rad=travel)
