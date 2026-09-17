"""What the receiver can actually measure about the fibre's polarisation rotation.

A compensator has to *undo* the fibre's rotation, so it has to know the whole
rotation — and one probe state is not enough to determine one.  Sending a single
state and watching where it lands on the Poincare sphere leaves a full U(1) of
rotations consistent with the observation: every rotation that maps the launch
state to the same point differs by a spin about that point.  You can compute
QBER from that single state and never notice the gap, which is exactly how a
polarisation dataset ends up unable to train a controller.

BB84 needs two mutually unbiased bases aligned, so a real polarisation
controller monitors **two non-parallel probe states** and reconstructs the
rotation from both.  That is what :func:`measure_rotation` models:

* two probe states are launched (by default the two Stokes axes that carry the
  BB84 bases), each arriving as a column of the fibre's rotation matrix;
* each arriving state is measured on three analyser axes with a **finite number
  of photons**, so each component is a binomial estimate with variance
  :math:`(1 - v_i^2)/N`;
* the two noisy columns are orthonormalised back onto SO(3), and the third
  column follows from the cross product.

The photon count is the knob that matters, and it is a real engineering trade:
every probe photon is a photon not spent on key.  Too few and the controller
chases its own measurement noise; too many and the monitoring channel costs more
than the drift does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MonitoringConfig", "measure_rotation", "orthonormalise"]

#: The two Stokes axes carrying BB84's rectilinear and diagonal bases.
DEFAULT_PROBE_STATES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


@dataclass(frozen=True)
class MonitoringConfig:
    """How the receiver watches the polarisation.

    Attributes:
        probe_states: launch states used for monitoring, as Stokes vectors.
            Two non-parallel states are the minimum that determines a rotation;
            a third adds redundancy and costs a third more photons.
        photons_per_basis: probe photons behind each analyser-axis estimate.
            Total monitoring cost per update is
            ``len(probe_states) * 3 * photons_per_basis``.
        detector_efficiency: fraction of probe photons that actually register.
            Folded straight into the effective count, since a lost photon and a
            photon never sent are the same thing to the estimator.
    """

    probe_states: Sequence[Sequence[float]] = DEFAULT_PROBE_STATES
    photons_per_basis: int = 400
    detector_efficiency: float = 0.85

    def __post_init__(self) -> None:
        probes = np.asarray(self.probe_states, dtype=float)
        if probes.ndim != 2 or probes.shape[1] != 3:
            raise ValueError("probe_states must be a sequence of Stokes 3-vectors")
        if len(probes) < 2:
            raise ValueError(
                "one probe state cannot determine a rotation — it leaves a U(1) "
                "of rotations consistent with the observation")
        if self.photons_per_basis < 1:
            raise ValueError("photons_per_basis must be at least 1")
        if not 0 < self.detector_efficiency <= 1:
            raise ValueError("detector_efficiency must be in (0, 1]")

    @property
    def effective_photons(self) -> int:
        """Photons that actually register per analyser-axis estimate."""
        return max(int(round(self.photons_per_basis * self.detector_efficiency)), 1)

    @property
    def photons_per_update(self) -> int:
        """Total probe photons spent on one rotation estimate."""
        return len(self.probe_states) * 3 * self.photons_per_basis


def orthonormalise(columns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project noisy measured columns back onto SO(3).

    Args:
        columns: shape ``(..., 3, k)`` with ``k >= 2`` measured columns.  Only
            the first two are used; the third is the cross product, which both
            enforces a right-handed frame and costs nothing.

    Returns:
        Rotation matrices of shape ``(..., 3, 3)``.

    Note:
        Gram-Schmidt rather than the SVD (Kabsch) projection.  With only two
        measured columns they give the same answer up to the noise already
        present, and Gram-Schmidt is a great deal cheaper inside a control loop
        that runs at a kilohertz.
    """
    first = columns[..., 0]
    second = columns[..., 1]

    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    projection = np.sum(second * first, axis=-1, keepdims=True) * first
    second = second - projection
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)

    return np.stack([first, second, third], axis=-1)


def measure_rotation(rotation: ArrayLike, config: MonitoringConfig,
                     rng: np.random.Generator | int | None = None
                     ) -> NDArray[np.float64]:
    """Estimate the fibre rotation from finite-photon probe measurements.

    Args:
        rotation: the true rotation(s), shape ``(..., 3, 3)``.
        config: the monitoring setup.
        rng: seed or generator.

    Returns:
        Estimated rotations, same shape, each a genuine element of SO(3).

    Note:
        The estimator is biased towards the identity in the same way any
        normalised estimate of a unit vector is: shrinking a noisy vector back
        to the sphere pulls components towards zero.  At a few hundred photons
        the bias is far below the drift being tracked; at a few dozen it is not,
        which is one of the things the photon sweep in the report measures.
    """
    generator = np.random.default_rng(rng)
    rotation = np.asarray(rotation, dtype=float)
    probes = np.asarray(config.probe_states, dtype=float)
    photons = config.effective_photons

    # Column j of R is where probe state j arrives (for the default axis probes);
    # in general it is R @ probe_j.
    arriving = np.einsum("...ij,kj->...ki", rotation, probes)

    # Each component is measured by an analyser aligned to that axis: the
    # detection probability is (1 + v_i)/2, so k successes out of N give an
    # unbiased estimate 2k/N - 1 with variance (1 - v_i^2)/N.
    probabilities = np.clip((1.0 + arriving) / 2.0, 0.0, 1.0)
    counts = generator.binomial(photons, probabilities)
    measured = 2.0 * counts / photons - 1.0

    return orthonormalise(np.moveaxis(measured, -2, -1))
