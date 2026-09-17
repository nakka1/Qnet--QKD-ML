"""Fibre channel: loss, latency, and the polarisation drift a compensator must track.

Loss and latency are one-liners.  The interesting part is the polarisation
model, because it is the thing the "compensate fibre fluctuations in real time"
problem is actually about, and because a model that gets its *time structure*
wrong will make a predictor look far better or far worse than it would be in
the field.

The model
---------
The fibre's effect on polarisation is a unitary on the qubit, i.e. a rotation
of the Stokes vector on the Poincare sphere.  We track that rotation
:math:`R(t) \\in SO(3)` directly, as the product of three physically distinct
contributions:

.. math:: R(t) = R_{\\text{jump}}(t)\\; R_{\\text{thermal}}(t)\\; R_{\\text{vib}}(t)

**Thermal** — the slow one.  Temperature changes the birefringence of the whole
span, so the rotation performs a random walk over the entire sphere.  The
angular velocity :math:`\\Omega_{\\text{th}}(t) \\in \\mathbb{R}^3` is an
Ornstein-Uhlenbeck process (exactly discretised, so the statistics do not
depend on the step size), optionally driven by a real temperature series
through ``dT/dt``.  This is the component a predictor *can* learn: it is
correlated over minutes and it is coupled to an observable covariate.

**Vibration** — the fast one.  Mechanical stress modulates birefringence in a
narrow acoustic band.  Synthesised as a sum of sinusoids with random phases and
frequencies drawn around the band centre (Rice's method), which gives a
stationary process with exactly the requested RMS amplitude and bandwidth, at
any step size, with no spin-up transient.  This component contributes a bounded
*angle*, not a drift.

**Jumps** — the rare one.  A cable is moved, a splice is disturbed, a patch
panel is touched.  Poisson-timed rotations of a few tenths of a radian.  These
matter out of all proportion to their rate: they are what puts the tail on the
QBER distribution, and no predictor can anticipate them.  A compensator is
judged on how fast it *recovers*, which is a different problem from tracking,
and a dataset without jumps will not teach that difference.

Splitting the three matters because they have different learnability.  Lumping
them into one Gaussian process — the usual shortcut — produces a signal that is
simultaneously too predictable in its tail and too unpredictable in its body.

What comes out
--------------
:meth:`PolarizationDriftModel.simulate` returns a
:class:`PolarizationTrajectory`: the resulting Stokes vector for a chosen launch
state, the misalignment angle, the induced QBER, the environmental covariates
that drove it — and, when asked, the **full rotation** :math:`R(t)`.

That last one is not a convenience.  Where a single Stokes vector went is enough
to compute QBER but **not enough to compensate**: one point on the sphere leaves
a whole U(1) of rotations that could have produced it.  BB84 needs two bases
aligned, so a real polarisation controller monitors two non-parallel probe
states and reconstructs the rotation from both.  A dataset that records only one
state can train a predictor of *misalignment*, which sounds useful and cannot
drive a controller.  ``track_rotation=True`` gives the compensator what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as C

__all__ = [
    "FiberParameters",
    "transmittance",
    "attenuation_db",
    "propagation_delay_s",
    "PolarizationDriftModel",
    "PolarizationTrajectory",
    "rotation_matrix",
    "rotation_vector",
    "rotation_angle",
    "stokes_fidelity",
    "stokes_qber",
    "polarization_fidelity_from_angle",
]


# ---------------------------------------------------------------------------
# loss and latency
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiberParameters:
    """Everything about one fibre span that the models need.

    Attributes:
        length_km: span length.
        attenuation_db_per_km: loss coefficient at the signal wavelength.
        group_index: fibre group index, sets propagation delay.
        wavelength_nm: signal wavelength, recorded for provenance.
    """

    length_km: float
    attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM
    group_index: float = C.FIBER_GROUP_INDEX
    wavelength_nm: float = C.DEFAULT_WAVELENGTH_NM

    @property
    def transmittance(self) -> float:
        """Fraction of photons surviving the span."""
        return transmittance(self.length_km, self.attenuation_db_per_km)

    @property
    def delay_s(self) -> float:
        """One-way propagation delay."""
        return propagation_delay_s(self.length_km, self.group_index)

    def with_length(self, length_km: float) -> "FiberParameters":
        """Same fibre, different length — for distance sweeps."""
        return replace(self, length_km=length_km)


def attenuation_db(length_km: ArrayLike,
                   attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM
                   ) -> NDArray[np.float64]:
    """Total loss of the span, in dB."""
    return np.asarray(length_km, dtype=float) * attenuation_db_per_km


def transmittance(length_km: ArrayLike,
                  attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM
                  ) -> NDArray[np.float64]:
    """Probability a photon survives the span.

    :math:`\\eta = 10^{-\\alpha L / 10}`.  This is the exponential wall that
    makes repeaters necessary: 0.2 dB/km costs a factor of 100 every 100 km.
    """
    return np.power(10.0, -attenuation_db(length_km, attenuation_db_per_km) / 10.0)


def propagation_delay_s(length_km: ArrayLike,
                        group_index: float = C.FIBER_GROUP_INDEX) -> NDArray[np.float64]:
    """One-way propagation delay through the span, in seconds.

    This is the quantity that bounds every heralded protocol: a link cannot
    confirm an entanglement attempt faster than the classical message can
    travel, so the attempt rate over an L-km span is capped near ``c/(n L)``
    regardless of how fast the hardware is.
    """
    return np.asarray(length_km, dtype=float) * 1e3 * group_index / C.SPEED_OF_LIGHT_VACUUM


# ---------------------------------------------------------------------------
# Poincare-sphere geometry
# ---------------------------------------------------------------------------

def rotation_matrix(axis: ArrayLike, angle: ArrayLike) -> NDArray[np.float64]:
    """Rotation of Stokes vectors about ``axis`` by ``angle`` (Rodrigues).

    Args:
        axis: 3-vector(s), need not be normalised; shape ``(..., 3)``.
        angle: rotation angle in radians, shape ``(...)``.

    Returns:
        Rotation matrices of shape ``(..., 3, 3)``.

    Note:
        A rotation by :math:`\\theta` on the Poincare sphere corresponds to a
        :math:`\\theta/2` rotation of the qubit state vector; all angles in
        this module are Poincare-sphere angles unless stated otherwise.
    """
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    unit = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 0)

    x, y, z = np.moveaxis(unit, -1, 0)
    zero = np.zeros_like(x)
    cross = np.stack([
        np.stack([zero, -z, y], axis=-1),
        np.stack([z, zero, -x], axis=-1),
        np.stack([-y, x, zero], axis=-1),
    ], axis=-2)

    outer = unit[..., :, None] * unit[..., None, :]
    identity = np.broadcast_to(np.eye(3), outer.shape)
    cos = np.cos(angle)[..., None, None]
    sin = np.sin(angle)[..., None, None]
    return cos * identity + sin * cross + (1 - cos) * outer


def rotation_vector(matrix: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`rotation_matrix`: the axis-times-angle of a rotation.

    The SO(3) logarithm.  Working in this three-vector representation is what
    makes extrapolating a rotation forward in time meaningful — averaging or
    differencing rotation *matrices* leaves the group, while rotation vectors
    compose additively to first order, which is exactly the regime a control
    loop operates in.

    Args:
        matrix: rotation matrices, shape ``(..., 3, 3)``.

    Returns:
        Rotation vectors, shape ``(..., 3)``, with magnitude in ``[0, pi]``.

    Note:
        Near ``pi`` the axis is recovered from the symmetric part rather than
        the antisymmetric one, because ``sin(theta)`` vanishes there and the
        usual formula loses all its precision. Near zero the small-angle limit
        is used for the same reason.
    """
    matrix = np.asarray(matrix, dtype=float)
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

    skew = np.stack([matrix[..., 2, 1] - matrix[..., 1, 2],
                     matrix[..., 0, 2] - matrix[..., 2, 0],
                     matrix[..., 1, 0] - matrix[..., 0, 1]], axis=-1)

    small = angle < 1e-8
    near_pi = angle > np.pi - 1e-5
    regular = ~(small | near_pi)

    result = np.zeros(matrix.shape[:-2] + (3,), dtype=float)
    # Small angle: sin(theta) ~ theta, so the skew part is already 2*vector.
    result[small] = 0.5 * skew[small]
    if np.any(regular):
        scale = (angle[regular] / (2.0 * np.sin(angle[regular])))[..., None]
        result[regular] = scale * skew[regular]
    if np.any(near_pi):
        # (R + I)/2 = n n^T at theta = pi; its largest column gives the axis.
        symmetric = (matrix[near_pi] + np.eye(3)) / 2.0
        diagonal = np.einsum("...ii->...i", symmetric)
        best = np.argmax(diagonal, axis=-1)
        axis = symmetric[np.arange(symmetric.shape[0]), :, best]
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 0)
        # The sign is ambiguous at exactly pi; take the skew part's hint.
        sign = np.sign(np.sum(axis * skew[near_pi], axis=-1, keepdims=True))
        sign = np.where(sign == 0, 1.0, sign)
        result[near_pi] = axis * sign * angle[near_pi][..., None]
    return result


def rotation_angle(matrix: ArrayLike) -> NDArray[np.float64]:
    """The rotation angle of a rotation matrix, in ``[0, pi]``.

    For a residual (uncorrected) rotation this is the single number that bounds
    how badly *any* polarisation state is misaligned — which is what a
    compensator is accountable for, unlike the misalignment of one chosen
    state, which can be zero while the rotation is large.
    """
    trace = np.trace(np.asarray(matrix, dtype=float), axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def stokes_fidelity(received: ArrayLike, reference: ArrayLike) -> NDArray[np.float64]:
    """Fidelity between two pure polarisation states given as Stokes vectors.

    For unit Stokes vectors, :math:`F = (1 + \\hat s \\cdot \\hat s_0)/2`.
    """
    received = np.asarray(received, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return 0.5 * (1.0 + np.sum(received * reference, axis=-1))


def stokes_qber(received: ArrayLike, reference: ArrayLike) -> NDArray[np.float64]:
    """Bit error probability from polarisation misalignment alone.

    The complement of :func:`stokes_fidelity`: a detector aligned to
    ``reference`` misidentifies a photon in state ``received`` with probability
    :math:`(1 - \\hat s \\cdot \\hat s_0)/2 = \\sin^2(\\Theta/2)`, where
    :math:`\\Theta` is the angle between them on the Poincare sphere.

    This is *only* the misalignment contribution.  Dark counts, detector
    crosstalk and multi-photon terms add to it; see
    :mod:`qnet_autopilot.physics.link`.
    """
    return 1.0 - stokes_fidelity(received, reference)


def polarization_fidelity_from_angle(angle: ArrayLike) -> NDArray[np.float64]:
    """SeQUeNCe's ``polarization_fidelity`` for a given misalignment angle.

    SeQUeNCe's ``QuantumChannel`` takes a scalar ``polarization_fidelity``: the
    probability a transmitted qubit suffers no polarisation error.  For a
    misalignment of :math:`\\Theta` on the Poincare sphere that probability is
    :math:`\\cos^2(\\Theta/2)`.  Use this to feed a drift trajectory into a
    SeQUeNCe simulation one time-slice at a time.
    """
    return np.cos(np.asarray(angle, dtype=float) / 2.0) ** 2


# ---------------------------------------------------------------------------
# drift model
# ---------------------------------------------------------------------------

@dataclass
class PolarizationTrajectory:
    """One simulated polarisation history, plus the covariates that caused it.

    Attributes:
        time_s: sample times, shape ``(n,)``.
        stokes: received Stokes vector at each sample, shape ``(n, 3)``.
        angle_rad: misalignment from the launch state on the Poincare sphere,
            shape ``(n,)``.  This is the regression target for a compensator.
        qber: bit error probability from that misalignment, shape ``(n,)``.
        temperature_c: driving temperature, shape ``(n,)``.
        vibration_rms: instantaneous vibration amplitude proxy — what an
            accelerometer on the cable would report, shape ``(n,)``.
        jump_times_s: times at which a discrete jump occurred.
        launch_stokes: the transmitted state everything is referenced to.
        rotation: the fibre's full Stokes rotation at each sample, shape
            ``(n, 3, 3)``, or ``None`` when it was not requested.  Column *i* is
            where the *i*-th basis state ended up, so ``rotation @ s`` sends any
            launch state to where it arrives.  This is what a compensator has to
            invert, and it is not recoverable from :attr:`stokes` alone.
    """

    time_s: NDArray[np.float64]
    stokes: NDArray[np.float64]
    angle_rad: NDArray[np.float64]
    qber: NDArray[np.float64]
    temperature_c: NDArray[np.float64]
    vibration_rms: NDArray[np.float64]
    jump_times_s: NDArray[np.float64]
    launch_stokes: NDArray[np.float64]
    rotation: NDArray[np.float64] | None = None

    def __len__(self) -> int:
        return int(self.time_s.size)

    @property
    def polarization_fidelity(self) -> NDArray[np.float64]:
        """The trajectory expressed as SeQUeNCe's channel parameter."""
        return polarization_fidelity_from_angle(self.angle_rad)

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Column-oriented view, ready for a DataFrame."""
        return {
            "time_s": self.time_s,
            "stokes_1": self.stokes[:, 0],
            "stokes_2": self.stokes[:, 1],
            "stokes_3": self.stokes[:, 2],
            "angle_rad": self.angle_rad,
            "qber": self.qber,
            "polarization_fidelity": self.polarization_fidelity,
            "temperature_c": self.temperature_c,
            "vibration_rms": self.vibration_rms,
        }


@dataclass
class PolarizationDriftModel:
    """Thermal drift + mechanical vibration + discrete jumps on the Poincare sphere.

    Attributes:
        thermal_correlation_time_s: OU correlation time of the drift rate.
            Long compared to the sampling interval means a predictable,
            slowly-varying drift; short means near-white and unlearnable.
        thermal_drift_rate_rad_per_s: steady-state RMS of the drift rate.
        thermal_per_kelvin_rad_per_s: extra drift rate per K/s of temperature
            change.  This is the coupling that makes a temperature sensor
            *useful* to a predictor; set it to zero for a purely stochastic
            model.
        vibration_frequency_hz: centre of the mechanical noise band.
        vibration_bandwidth_hz: width of that band.  Zero gives a pure tone.
        vibration_amplitude_rad: RMS angular amplitude of the vibration term.
        vibration_num_tones: how many sinusoids to synthesise the band with.
            Eight is enough for the sum to look Gaussian.
        vibration_harmonics: relative RMS weight of each harmonic of
            ``vibration_frequency_hz``, starting at the fundamental.  The
            default ``(1.0,)`` is a single band and reproduces the original
            model exactly.  Aerial fibre needs more than one: a polarimeter on
            220 km of optical-ground-wire sees a strong tone at the power-line
            frequency *with* its second and third harmonics, because the
            driver is the current in the conductor the fibre is strung inside
            (Peterson et al., NIST).  Weights are renormalised so the total RMS
            stays ``vibration_amplitude_rad`` — changing the harmonic content
            redistributes the jitter, it does not add any.
        thermal_components: optional ``(correlation_time_s, rate_rad_per_s)``
            pairs, summed, replacing the single OU term.  One OU process has a
            flat spectrum below its corner and :math:`f^{-2}` above it; measured
            drift on deployed fibre is closer to :math:`f^{-1}` over four
            decades (Ramaswamy et al., ORNL: :math:`\beta = 1.0 \pm 0.2`), which
            no single relaxation time can produce.  A handful of OU processes
            with logarithmically spaced correlation times can — see
            :func:`~qnet_autopilot.physics.field_calibration.flicker_components`.
        jump_rate_per_s: Poisson rate of discrete polarisation jumps.
        jump_scale_rad: mean angular size of a jump (exponentially distributed).
    """

    thermal_correlation_time_s: float = C.DEFAULT_THERMAL_CORRELATION_TIME_S
    thermal_drift_rate_rad_per_s: float = C.DEFAULT_THERMAL_DRIFT_RATE_RAD_PER_S
    thermal_per_kelvin_rad_per_s: float = 5.0e-2
    vibration_frequency_hz: float = C.DEFAULT_VIBRATION_FREQUENCY_HZ
    vibration_bandwidth_hz: float = 40.0
    vibration_amplitude_rad: float = C.DEFAULT_VIBRATION_AMPLITUDE_RAD
    vibration_num_tones: int = 8
    vibration_harmonics: tuple[float, ...] = (1.0,)
    thermal_components: tuple[tuple[float, float], ...] | None = None
    jump_rate_per_s: float = C.DEFAULT_JUMP_RATE_PER_S
    jump_scale_rad: float = C.DEFAULT_JUMP_SCALE_RAD

    def simulate(self,
                 duration_s: float,
                 dt_s: float,
                 launch_stokes: ArrayLike = (1.0, 0.0, 0.0),
                 temperature_c: ArrayLike | None = None,
                 rng: np.random.Generator | int | None = None,
                 track_rotation: bool = False,
                 ) -> PolarizationTrajectory:
        """Simulate a polarisation history.

        Args:
            duration_s: total simulated time.
            dt_s: sampling interval.  The thermal process is exactly
                discretised so this is free to choose; the vibration term is
                sampled, not integrated, so it aliases above the Nyquist rate
                of ``vibration_frequency_hz`` — sample at least twice the band
                edge if the vibration matters to you.
            launch_stokes: transmitted polarisation, as a Stokes 3-vector.
                Normalised internally.
            temperature_c: driving temperature at each sample.  If ``None``, a
                plausible diurnal-plus-noise series is synthesised.  Pass real
                sensor data here to drive the model from a measured
                environment.
            rng: seed or generator, for reproducibility.
            track_rotation: also return the full 3x3 rotation at each sample.
                Costs three vector rotations per step instead of one, and
                ``9n`` floats of memory. Required for anything that has to
                *undo* the drift rather than merely measure it.

        Returns:
            PolarizationTrajectory: the history and its covariates.

        Raises:
            ValueError: on a non-positive step, or a temperature series whose
                length does not match the number of samples.
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        generator = np.random.default_rng(rng)
        num_samples = int(round(duration_s / dt_s))
        time_s = np.arange(num_samples, dtype=float) * dt_s

        temperature = self._temperature_series(time_s, temperature_c, generator)
        thermal_rate = self._thermal_rate(time_s, dt_s, temperature, generator)
        vibration, vibration_rms = self._vibration(time_s, generator)
        jump_times, jump_quaternions = self._jumps(duration_s, generator)

        launch = np.asarray(launch_stokes, dtype=float)
        launch = launch / np.linalg.norm(launch)

        # Accumulating the slow rotation is inherently sequential, so it is the
        # one part that cannot be vectorised. Quaternions keep that loop cheap
        # (four floats and sixteen multiplies per step, in plain Python) and
        # everything after it — applying the rotation to the launch state, then
        # the vibration offset on top — is done on the whole array at once.
        thermal_quaternions = _accumulate_rotation(
            thermal_rate * dt_s, time_s, jump_times, jump_quaternions)

        drifted = _rotate_by_quaternion(thermal_quaternions, launch)
        stokes = _rotate_about_axis(drifted, vibration)

        rotation = None
        if track_rotation:
            # Rotating the three basis vectors gives the matrix's columns.
            columns = [_rotate_about_axis(
                _rotate_by_quaternion(thermal_quaternions, axis), vibration)
                for axis in np.eye(3)]
            rotation = np.stack(columns, axis=-1)

        angle_rad = np.arccos(np.clip(stokes @ launch, -1.0, 1.0))
        qber = stokes_qber(stokes, launch)

        return PolarizationTrajectory(
            time_s=time_s,
            stokes=stokes,
            angle_rad=angle_rad,
            qber=qber,
            temperature_c=temperature,
            vibration_rms=vibration_rms,
            jump_times_s=jump_times,
            launch_stokes=launch,
            rotation=rotation,
        )

    # -- components ---------------------------------------------------------

    def _temperature_series(self, time_s, temperature_c, generator) -> NDArray[np.float64]:
        """Use the caller's temperature, or synthesise a diurnal one."""
        if temperature_c is not None:
            temperature = np.asarray(temperature_c, dtype=float)
            if temperature.shape != time_s.shape:
                raise ValueError(
                    f"temperature_c has {temperature.size} samples but the run has {time_s.size}")
            return temperature

        # 24-hour sinusoid, 6 K peak-to-peak, plus slow noise. Buried fibre sees
        # a damped version of the surface swing; this is the damped version.
        diurnal = 3.0 * np.sin(2 * np.pi * time_s / 86400.0)
        noise = np.cumsum(generator.normal(0.0, 0.01, size=time_s.size))
        noise -= noise.mean()
        return 20.0 + diurnal + 0.1 * noise

    def _ou_rate(self, num_samples, dt_s, tau, sigma, generator
                 ) -> NDArray[np.float64]:
        """One exactly-discretised Ornstein-Uhlenbeck angular velocity."""
        decay = np.exp(-dt_s / tau) if tau > 0 else 0.0
        # Stationary increment std: keeps the marginal variance at sigma^2
        # whatever dt is, so results do not shift when you resample.
        increment_std = sigma * np.sqrt(max(1.0 - decay ** 2, 0.0))

        rate = np.empty((num_samples, 3), dtype=float)
        # Start in the stationary distribution: no spin-up transient.
        current = generator.normal(0.0, sigma, size=3)
        noise = generator.normal(0.0, increment_std, size=(num_samples, 3))
        for index in range(num_samples):
            current = decay * current + noise[index]
            rate[index] = current
        return rate

    def _thermal_rate(self, time_s, dt_s, temperature, generator) -> NDArray[np.float64]:
        """Exactly-discretised OU angular velocity, plus a dT/dt drive.

        With ``thermal_components`` the single OU process becomes a sum of
        them. Summing independent OU processes is what turns a spectrum with
        one corner into one that is straight over the decades the components
        span — the shape measured on deployed fibre.
        """
        num_samples = time_s.size

        if self.thermal_components:
            rate = np.zeros((num_samples, 3), dtype=float)
            for tau, sigma in self.thermal_components:
                rate += self._ou_rate(num_samples, dt_s, tau, sigma, generator)
        else:
            rate = self._ou_rate(num_samples, dt_s,
                                 self.thermal_correlation_time_s,
                                 self.thermal_drift_rate_rad_per_s, generator)

        if self.thermal_per_kelvin_rad_per_s and num_samples > 1:
            # dT/dt drives drift along a fixed (but arbitrary) fibre axis: the
            # birefringence axis of the span, which does not itself move.
            dtemp_dt = np.gradient(temperature, dt_s)
            axis = generator.normal(size=3)
            axis /= np.linalg.norm(axis)
            rate += self.thermal_per_kelvin_rad_per_s * dtemp_dt[:, None] * axis

        return rate

    def _vibration(self, time_s, generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Narrowband angular jitter by sum-of-sinusoids (Rice's method)."""
        num_tones = max(int(self.vibration_num_tones), 1)
        amplitude = self.vibration_amplitude_rad
        if amplitude <= 0:
            zeros = np.zeros((time_s.size, 3))
            return zeros, np.zeros(time_s.size)

        # Equal-power tones spread across the band; random phases decorrelate
        # the axes. Per-tone amplitude sqrt(2/N) makes the sum's RMS exactly
        # `amplitude` per axis.
        weights = np.asarray(self.vibration_harmonics or (1.0,), dtype=float)
        if np.any(weights < 0):
            raise ValueError("vibration_harmonics cannot be negative")
        norm = float(np.sqrt(np.sum(weights ** 2)))
        if norm <= 0:
            raise ValueError("vibration_harmonics cannot be all zero")
        weights = weights / norm

        vibration = np.zeros((3, time_s.size), dtype=float)
        for order, weight in enumerate(weights, start=1):
            if weight <= 0:
                continue
            centre = self.vibration_frequency_hz * order
            frequencies = (centre + self.vibration_bandwidth_hz
                           * (generator.random((3, num_tones)) - 0.5))
            phases = generator.uniform(0.0, 2 * np.pi, size=(3, num_tones))
            per_tone = amplitude * weight * np.sqrt(2.0 / num_tones)
            phase = (2 * np.pi * frequencies[:, :, None] * time_s[None, None, :]
                     + phases[:, :, None])
            vibration += per_tone * np.sin(phase).sum(axis=1)  # (3, n)
        vibration = vibration.T  # (n, 3)

        # What an accelerometer would report: the magnitude of the jitter.
        vibration_rms = np.linalg.norm(vibration, axis=-1)
        return vibration, vibration_rms

    def _jumps(self, duration_s, generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Poisson-timed rotations of exponentially distributed size, as quaternions."""
        if self.jump_rate_per_s <= 0:
            return np.empty(0), np.empty((0, 4))

        expected = self.jump_rate_per_s * duration_s
        count = generator.poisson(expected)
        if count == 0:
            return np.empty(0), np.empty((0, 4))

        times = np.sort(generator.uniform(0.0, duration_s, size=count))
        axes = generator.normal(size=(count, 3))
        angles = generator.exponential(self.jump_scale_rad, size=count)
        return times, _axis_angle_to_quaternion(axes, angles)


# ---------------------------------------------------------------------------
# rotation plumbing
# ---------------------------------------------------------------------------
#
# Rotations are composed as quaternions and applied as vectorised Rodrigues.
# Matrices would be clearer to read but 3x3 numpy products in a per-sample loop
# dominate the runtime of every long trajectory, and long trajectories are the
# point: a compensator sampled at 1 kHz over ten minutes is 600,000 steps.

def _axis_angle_to_quaternion(axis: ArrayLike, angle: ArrayLike) -> NDArray[np.float64]:
    """``(w, x, y, z)`` for a rotation of ``angle`` about ``axis``."""
    axis = np.atleast_2d(np.asarray(axis, dtype=float))
    angle = np.atleast_1d(np.asarray(angle, dtype=float))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    unit = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 0)
    half = angle / 2.0
    return np.concatenate([np.cos(half)[:, None], unit * np.sin(half)[:, None]], axis=-1)


def _accumulate_rotation(step_vectors: NDArray[np.float64],
                         time_s: NDArray[np.float64],
                         jump_times: NDArray[np.float64],
                         jump_quaternions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative rotation quaternion after each step, with jumps inserted.

    ``step_vectors[k]`` is the rotation vector (axis times angle) applied at
    step ``k``; the result is the running product, so element ``k`` is the
    total rotation the fibre has applied by time ``time_s[k]``.
    """
    angles = np.linalg.norm(step_vectors, axis=-1)
    increments = _axis_angle_to_quaternion(step_vectors, angles)

    cumulative = np.empty_like(increments)
    w, x, y, z = 1.0, 0.0, 0.0, 0.0
    next_jump = 0

    for index in range(increments.shape[0]):
        aw, ax, ay, az = increments[index]
        w, x, y, z = (aw * w - ax * x - ay * y - az * z,
                      aw * x + ax * w + ay * z - az * y,
                      aw * y - ax * z + ay * w + az * x,
                      aw * z + ax * y - ay * x + az * w)

        while next_jump < jump_times.size and jump_times[next_jump] <= time_s[index]:
            jw, jx, jy, jz = jump_quaternions[next_jump]
            w, x, y, z = (jw * w - jx * x - jy * y - jz * z,
                          jw * x + jx * w + jy * z - jz * y,
                          jw * y - jx * z + jy * w + jz * x,
                          jw * z + jx * y - jy * x + jz * w)
            next_jump += 1

        cumulative[index] = (w, x, y, z)

    return cumulative


def _rotate_by_quaternion(quaternions: NDArray[np.float64],
                          vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate one vector by each of ``quaternions``: ``v' = v + 2 q_v x (q_v x v + w v)``."""
    w = quaternions[:, 0:1]
    q_vector = quaternions[:, 1:4]
    temp = np.cross(q_vector, vector) + w * vector
    return vector + 2.0 * np.cross(q_vector, temp)


def _rotate_about_axis(vectors: NDArray[np.float64],
                       rotation_vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues rotation of each vector by its own rotation vector, vectorised."""
    angles = np.linalg.norm(rotation_vectors, axis=-1, keepdims=True)
    axes = np.divide(rotation_vectors, angles,
                     out=np.zeros_like(rotation_vectors), where=angles > 0)
    cos = np.cos(angles)
    sin = np.sin(angles)
    dot = np.sum(axes * vectors, axis=-1, keepdims=True)
    return (vectors * cos + np.cross(axes, vectors) * sin + axes * dot * (1.0 - cos))
