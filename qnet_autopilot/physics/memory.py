"""Quantum memory: how stored entanglement decays while it waits.

Every repeater decision is a trade against this clock.  Purifying improves
fidelity but costs two pairs and a round trip, during which everything else in
the memory bank decoheres.  Waiting for a better path costs fidelity on the
pairs already held.  A scheduler that does not model the decay accurately will
systematically over- or under-value patience, so this module makes the decay
explicit and cheap to evaluate.

The model is SeQUeNCe's: a memory is a qubit subject to a Pauli channel whose
total rate is ``1 / coherence_time``, split across X, Y and Z by the
``decoherence_errors`` distribution.  Acting on a Bell-diagonal pair, that
channel has the closed form in
:func:`qnet_autopilot.physics.bell_diagonal.decohere`, so a pair's whole
lifetime can be evaluated in one array operation instead of stepped.

Two distinct times govern a stored pair and they are easy to confuse:

* **coherence time** — the physical decay constant of the memory.
* **cutoff time** — a *policy*: the age at which the network gives up on a pair
  and discards it, ``cutoff_ratio x coherence_time`` in SeQUeNCe.  Raising it
  keeps low-fidelity pairs alive and occupying memory slots; lowering it throws
  away pairs that a purification round could still have rescued.  It is one of
  the knobs an agent can hold, which is why it is a parameter here and not a
  constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as C
from .bell_diagonal import decohere, fidelity, werner

__all__ = ["MemoryParameters", "pauli_rates", "fidelity_after", "time_to_fidelity"]

#: Equal split across X, Y and Z — a depolarising memory.  SeQUeNCe's default.
DEPOLARIZING_ERRORS = (1 / 3, 1 / 3, 1 / 3)

#: Pure dephasing: only Z errors.  Typical of spin qubits with long T1 and
#: short T2, where the dominant process is phase noise rather than relaxation.
DEPHASING_ERRORS = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class MemoryParameters:
    """Hardware description of one quantum memory.

    Attributes:
        coherence_time_s: the memory's decay constant; total Pauli error rate
            is its reciprocal.
        raw_fidelity: fidelity of a freshly heralded pair, before any idling.
        efficiency: probability that reading the memory out produces a photon.
        decoherence_errors: how the total error rate splits across X, Y, Z.
            Must sum to 1.
        cutoff_ratio: discard a pair once its age exceeds this many coherence
            times.  A policy, not a property of the hardware.
        frequency_hz: maximum excitation rate of the memory.
        wavelength_nm: emitted photon wavelength.
    """

    coherence_time_s: float = C.DEFAULT_MEMORY_COHERENCE_TIME_S
    raw_fidelity: float = C.DEFAULT_RAW_FIDELITY
    efficiency: float = 0.75
    decoherence_errors: Sequence[float] = DEPOLARIZING_ERRORS
    cutoff_ratio: float = 1.0
    frequency_hz: float = 2.0e3
    wavelength_nm: float = C.DEFAULT_WAVELENGTH_NM

    def __post_init__(self) -> None:
        errors = np.asarray(self.decoherence_errors, dtype=float)
        if errors.size != 3:
            raise ValueError("decoherence_errors must have three entries (X, Y, Z)")
        if not np.isclose(errors.sum(), 1.0):
            raise ValueError(f"decoherence_errors must sum to 1, got {errors.sum()}")
        if self.coherence_time_s <= 0:
            raise ValueError("coherence_time_s must be positive")

    @property
    def decoherence_rate(self) -> float:
        """Total Pauli error rate, s\\ :sup:`-1` — SeQUeNCe's ``decoherence_rate``."""
        return 1.0 / self.coherence_time_s

    @property
    def cutoff_time_s(self) -> float:
        """Age at which a stored pair is discarded."""
        return self.cutoff_ratio * self.coherence_time_s

    @property
    def pauli_rates(self) -> tuple[float, float, float]:
        """``(x_rate, y_rate, z_rate)`` in s\\ :sup:`-1`."""
        return pauli_rates(self.coherence_time_s, self.decoherence_errors)

    def fresh_state(self) -> NDArray[np.float64]:
        """Bell-diagonal state of a newly heralded pair."""
        return werner(self.raw_fidelity)


def pauli_rates(coherence_time_s: float,
                decoherence_errors: Sequence[float] = DEPOLARIZING_ERRORS
                ) -> tuple[float, float, float]:
    """Split a coherence time into per-axis Pauli error rates.

    Mirrors ``Memory.bds_decohere``: ``rate_i = decoherence_errors[i] / coherence_time``.
    """
    rate = 1.0 / coherence_time_s
    x, y, z = (float(v) for v in decoherence_errors)
    return rate * x, rate * y, rate * z


def fidelity_after(state: ArrayLike, elapsed_s: ArrayLike,
                   parameters: MemoryParameters | None = None,
                   both_qubits: bool = True) -> NDArray[np.float64]:
    """Fidelity of a stored pair after idling for ``elapsed_s``.

    Args:
        state: Bell-diagonal state when stored, shape ``(..., 4)``.
        elapsed_s: idle duration(s) in seconds; broadcasts against ``state``.
        parameters: memory hardware; defaults to :class:`MemoryParameters`.
        both_qubits: apply the channel to both halves of the pair.  True is
            right when both ends idle for the same time — the usual case for a
            pair sitting in a buffer.  Set False to model one end only, which
            is what SeQUeNCe does per ``bds_decohere`` call.

    Returns:
        Fidelity after the wait, broadcast over ``state`` and ``elapsed_s``.
    """
    parameters = parameters or MemoryParameters()
    x_rate, y_rate, z_rate = parameters.pauli_rates
    elapsed = np.asarray(elapsed_s, dtype=float)
    decohered = decohere(state, x_rate, y_rate, z_rate,
                         elapsed[..., None] if elapsed.ndim else elapsed,
                         both_qubits=both_qubits)
    return fidelity(decohered)


def time_to_fidelity(target_fidelity: float,
                     state: ArrayLike | None = None,
                     parameters: MemoryParameters | None = None,
                     both_qubits: bool = True,
                     max_time_s: float | None = None) -> float:
    """How long a pair stays above ``target_fidelity``.

    Solved by bisection rather than in closed form, because the closed form
    depends on the X/Y/Z split and stops being invertible for anisotropic
    channels.  Bisection is exact to floating point here and costs nothing at
    the rate this is called (once per scheduling decision, not once per step).

    Args:
        target_fidelity: the threshold to stay above.
        state: starting Bell-diagonal state; defaults to a fresh pair.
        parameters: memory hardware.
        both_qubits: as in :func:`fidelity_after`.
        max_time_s: search ceiling; defaults to 100 coherence times.

    Returns:
        Seconds until fidelity first drops to ``target_fidelity``.  ``0.0`` if
        the pair already starts below it, ``inf`` if it never gets there within
        ``max_time_s`` (a memory with no decoherence, or a target below the
        1/4 depolarised floor).
    """
    parameters = parameters or MemoryParameters()
    state = parameters.fresh_state() if state is None else np.asarray(state, dtype=float)
    ceiling = max_time_s if max_time_s is not None else 100.0 * parameters.coherence_time_s

    if float(fidelity(state)) <= target_fidelity:
        return 0.0
    if float(fidelity_after(state, ceiling, parameters, both_qubits)) > target_fidelity:
        return float("inf")

    low, high = 0.0, ceiling
    for _ in range(200):
        mid = 0.5 * (low + high)
        if float(fidelity_after(state, mid, parameters, both_qubits)) > target_fidelity:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
