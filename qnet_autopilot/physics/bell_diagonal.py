"""Bell-diagonal state algebra: decoherence, purification and swapping.

Why this module exists
----------------------
SeQUeNCe already implements every operation below, and implements them
correctly — but it implements them as *protocol objects bound to nodes on a
discrete-event timeline*.  Evaluating one purification step means constructing
a ``BBPSSW_BDS`` protocol, attaching it to two ``Memory`` instances owned by
two ``Node`` instances on a running ``Timeline``, and pumping events.  That is
the right design for simulating a network and the wrong one for the inner loop
of a reinforcement-learning agent, which needs to ask "what happens to fidelity
if I purify twice and then swap?" a few million times.

So this module provides the same operations as pure, stateless, vectorised
functions over numpy arrays.  They are not an approximation of SeQUeNCe and
not an independent re-derivation: they are transcriptions of
``sequence.entanglement_management.purification.{bbpssw_bds,dejmps_bds}``,
``sequence.entanglement_management.swapping.swapping_bds`` and
``sequence.components.memory.bds_decohere``, keeping the upstream variable
names (``gate_fid``, ``meas_fid``, ``p_succ``, ``c_I``/``c_X``/``c_Y``/``c_Z``,
``same_report`` …) so the two can be read side by side.
``tests/test_bds_matches_sequence.py`` asserts they agree with SeQUeNCe to
machine precision; if upstream changes a formula, that test fails.

Conventions (SeQUeNCe's, exactly)
---------------------------------
A Bell-diagonal state is the 4-vector of diagonal weights in the Bell basis,
in the order

.. math:: [\\,\\Phi^+,\\ \\Phi^-,\\ \\Psi^+,\\ \\Psi^-\\,]

so ``state[0]`` is the fidelity with respect to the target state
:math:`\\Phi^+`, and index 1, 2, 3 are reached from index 0 by a single
:math:`Z`, :math:`X` and :math:`Y` error respectively.  Note that DEJMPS is
conventionally written in the order :math:`[\\Phi^+, \\Psi^-, \\Psi^+, \\Phi^-]`;
:func:`dejmps` performs the index permutation ``[0, 3, 2, 1]`` internally, as
upstream does.

Every function accepts arrays of shape ``(..., 4)`` and broadcasts over the
leading axes, so a whole batch of links — or a whole batch of RL environments —
is one call.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PHI_PLUS", "PHI_MINUS", "PSI_PLUS", "PSI_MINUS",
    "werner", "fidelity", "twirl", "is_valid",
    "pauli_channel", "decohere",
    "bbpssw", "dejmps", "swap",
]

#: Index of each Bell state in the 4-vector.
PHI_PLUS, PHI_MINUS, PSI_PLUS, PSI_MINUS = 0, 1, 2, 3

#: Permutation between SeQUeNCe storage order and the DEJMPS recurrence order.
_DEJMPS_ORDER = [0, 3, 2, 1]

_EPS = 1e-12


def _as_bds(state: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(state, dtype=float)
    if array.shape[-1] != 4:
        raise ValueError(f"a Bell-diagonal state has 4 components, got shape {array.shape}")
    return array


def werner(fidelity_value: ArrayLike) -> NDArray[np.float64]:
    """Build the Werner state of the given :math:`\\Phi^+` fidelity.

    A Werner state spreads the remaining weight evenly over the other three
    Bell states, so ``[F, (1-F)/3, (1-F)/3, (1-F)/3]``.  This is the state
    BBPSSW assumes on input and produces on output.

    Args:
        fidelity_value: scalar or array of fidelities in [0, 1].

    Returns:
        Array of shape ``(..., 4)``.
    """
    F = np.asarray(fidelity_value, dtype=float)
    rest = (1.0 - F) / 3.0
    return np.stack([F, rest, rest, rest], axis=-1)


def fidelity(state: ArrayLike) -> NDArray[np.float64]:
    """:math:`\\Phi^+` fidelity, i.e. component 0."""
    return _as_bds(state)[..., PHI_PLUS]


def twirl(state: ArrayLike) -> NDArray[np.float64]:
    """Depolarise to the Werner state of the same fidelity.

    Twirling is what makes BBPSSW's closed form valid and what SeQUeNCe's
    swapping does when ``is_twirled`` is set.  It never increases fidelity and
    always destroys information, so apply it only where the protocol says to.
    """
    return werner(fidelity(state))


def is_valid(state: ArrayLike, tol: float = 1e-9) -> NDArray[np.bool_]:
    """True where the vector is a probability distribution over the Bell basis."""
    array = _as_bds(state)
    return (np.all(array >= -tol, axis=-1)
            & (np.abs(array.sum(axis=-1) - 1.0) <= tol))


# ---------------------------------------------------------------------------
# decoherence
# ---------------------------------------------------------------------------

def pauli_channel(x_rate: ArrayLike, y_rate: ArrayLike, z_rate: ArrayLike,
                  time: ArrayLike) -> NDArray[np.float64]:
    """Single-qubit Pauli-channel probabilities after idling for ``time``.

    Transcribes ``_p_id`` / ``_p_xerr`` / ``_p_yerr`` / ``_p_zerr`` from
    ``sequence.components.memory``.

    Args:
        x_rate, y_rate, z_rate: Pauli error rates in s\\ :sup:`-1`, i.e. the
            memory's decoherence rate split across X, Y and Z (see
            :func:`qnet_autopilot.physics.memory.pauli_rates`).
        time: idling duration **in seconds**.

    Returns:
        Array of shape ``(..., 4)`` holding ``[p_I, p_X, p_Y, p_Z]``.
    """
    x_rate, y_rate, z_rate, t = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x_rate, y_rate, z_rate, time)))

    exp_xy = np.exp(-2 * (x_rate + y_rate) * t)
    exp_xz = np.exp(-2 * (x_rate + z_rate) * t)
    exp_zy = np.exp(-2 * (z_rate + y_rate) * t)

    p_I = (1 + exp_xy + exp_xz + exp_zy) / 4
    p_X = (1 - exp_xy - exp_xz + exp_zy) / 4
    p_Y = (1 - exp_xy + exp_xz - exp_zy) / 4
    p_Z = (1 + exp_xy - exp_xz - exp_zy) / 4
    return np.stack([p_I, p_X, p_Y, p_Z], axis=-1)


def decohere(state: ArrayLike, x_rate: ArrayLike, y_rate: ArrayLike,
             z_rate: ArrayLike, time: ArrayLike, both_qubits: bool = False
             ) -> NDArray[np.float64]:
    """Apply memory idling decoherence to a Bell-diagonal state.

    This is ``Memory.bds_decohere`` without the timeline: the same
    ``transform_mtx`` acting on the diagonal elements.

    Args:
        state: Bell-diagonal state(s), shape ``(..., 4)``.
        x_rate, y_rate, z_rate: Pauli error rates in s\\ :sup:`-1`.
        time: idling duration in seconds.
        both_qubits: SeQUeNCe decoheres **one** memory per call, because the
            two halves of a pair idle for different durations and are updated
            independently.  Set this to True to apply the same channel to both
            halves in one step, which is what you want when the pair is held
            symmetrically (both ends idle equally) — it is exactly the
            single-sided transform applied twice.

    Returns:
        The decohered state, same shape as ``state``.
    """
    array = _as_bds(state)
    p_I, p_X, p_Y, p_Z = np.moveaxis(pauli_channel(x_rate, y_rate, z_rate, time), -1, 0)

    # Row/column layout copied from Memory.bds_decohere's transform_mtx.
    transform_mtx = np.stack([
        np.stack([p_I, p_Z, p_X, p_Y], axis=-1),
        np.stack([p_Z, p_I, p_Y, p_X], axis=-1),
        np.stack([p_X, p_Y, p_I, p_Z], axis=-1),
        np.stack([p_Y, p_X, p_Z, p_I], axis=-1),
    ], axis=-2)

    state_new = np.einsum("...ij,...j->...i", transform_mtx, array)
    if both_qubits:
        state_new = np.einsum("...ij,...j->...i", transform_mtx, state_new)
    return state_new


# ---------------------------------------------------------------------------
# purification
# ---------------------------------------------------------------------------

def bbpssw(kept_state: ArrayLike, meas_state: ArrayLike,
           gate_fid: ArrayLike = 1.0, meas_fid: ArrayLike = 1.0,
           remote_gate_fid: ArrayLike | None = None,
           remote_meas_fid: ArrayLike | None = None,
           ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """BBPSSW (recurrence) purification of two pairs, in closed form.

    Transcribes ``BBPSSW_BDS.purification_res``, which implements Eqs. (8) and
    (9) of *Entanglement Distribution in Quantum Repeater with Purification and
    Optimized Buffer Time* (arXiv:2305.14573).

    BBPSSW is defined on Werner states: only ``state[0]`` of each input is
    read, and the output is a Werner state.  Feed it a non-Werner input and you
    get the twirled answer — the same thing SeQUeNCe gives you, and a real
    approximation, not a rounding error.  Use :func:`dejmps` to keep the full
    Bell-diagonal structure.

    Args:
        kept_state: the pair that survives on success, shape ``(..., 4)``.
        meas_state: the pair consumed by the measurement, shape ``(..., 4)``.
        gate_fid: probability the local CNOT at the owner node is ideal (``p1``).
        meas_fid: probability the owner node reports the correct result (``eta1``).
        remote_gate_fid: same for the remote node (``p2``); defaults to ``gate_fid``.
        remote_meas_fid: same for the remote node (``eta2``); defaults to ``meas_fid``.

    Returns:
        ``(p_s_w, bds_elems)`` — success probability, and the Werner state of
        the retained pair conditioned on success.

    Raises:
        ValueError: if ``gate_fid * remote_gate_fid`` is zero, which makes
            Eq. (9) singular (upstream asserts the same thing).
    """
    p1 = np.asarray(gate_fid, dtype=float)
    p2 = p1 if remote_gate_fid is None else np.asarray(remote_gate_fid, dtype=float)
    eta1 = np.asarray(meas_fid, dtype=float)
    eta2 = eta1 if remote_meas_fid is None else np.asarray(remote_meas_fid, dtype=float)

    if np.any(p1 * p2 <= 0):
        raise ValueError("the uncancelled expression in Eq. (9) requires nonzero gate fidelities")

    F1 = fidelity(kept_state)
    F2 = fidelity(meas_state)
    e1 = (1 - F1) / 3
    e2 = (1 - F2) / 3

    same_report = eta1 * eta2 + (1 - eta1) * (1 - eta2)
    opposite_report = eta1 * (1 - eta2) + (1 - eta1) * eta2

    p_s_w = (p1 * p2 * same_report * (F1 * F2 + F1 * e2 + e1 * F2 + 5 * e1 * e2)
             + p1 * p2 * opposite_report * (2 * F1 * e2 + 2 * e1 * F2 + 4 * e1 * e2)
             + (1 - p1 * p2) / 2)

    F_s_w_numerator = (same_report * (F1 * F2 + e1 * e2)
                       + opposite_report * (F1 * e2 + e1 * e2)
                       + (1 - p1 * p2) / (8 * p1 * p2))
    F_s_w = F_s_w_numerator / (p_s_w / (p1 * p2))

    return p_s_w, werner(F_s_w)


def dejmps(kept_state: ArrayLike, meas_state: ArrayLike,
           gate_fid: ArrayLike = 1.0, meas_fid: ArrayLike = 1.0,
           remote_gate_fid: ArrayLike | None = None,
           remote_meas_fid: ArrayLike | None = None,
           ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """DEJMPS purification of two pairs, in closed form.

    Transcribes ``DEJMPS_BDS.purification_res``.  Unlike :func:`bbpssw`, DEJMPS
    uses the whole Bell-diagonal vector and returns a general Bell-diagonal
    state, so it outperforms BBPSSW on states that are not Werner — which is
    most states produced by an asymmetric link.

    Args and returns mirror :func:`bbpssw`; the output state is **not**
    twirled.
    """
    p1 = np.asarray(gate_fid, dtype=float)
    p2 = p1 if remote_gate_fid is None else np.asarray(remote_gate_fid, dtype=float)
    eta1 = np.asarray(meas_fid, dtype=float)
    eta2 = eta1 if remote_meas_fid is None else np.asarray(remote_meas_fid, dtype=float)

    kept = _as_bds(kept_state)[..., _DEJMPS_ORDER]
    meas = _as_bds(meas_state)[..., _DEJMPS_ORDER]

    kept_elem_1, kept_elem_2, kept_elem_3, kept_elem_4 = np.moveaxis(kept, -1, 0)
    meas_elem_1, meas_elem_2, meas_elem_3, meas_elem_4 = np.moveaxis(meas, -1, 0)

    a = kept_elem_1 + kept_elem_2
    b = meas_elem_1 + meas_elem_2

    same_report = eta1 * eta2 + (1 - eta1) * (1 - eta2)
    opposite_report = eta1 * (1 - eta2) + (1 - eta1) * eta2
    joint_gate_fid = p1 * p2

    p_succ = (1 / 2
              + joint_gate_fid * opposite_report
              + joint_gate_fid * (a * b + (1 - a) * (1 - b)) * (same_report - opposite_report)
              - joint_gate_fid / 2)

    new_elem_1 = (joint_gate_fid * (same_report * (kept_elem_1 * meas_elem_1 + kept_elem_2 * meas_elem_2)
                                    + opposite_report * (kept_elem_1 * meas_elem_3 + kept_elem_2 * meas_elem_4))
                  + (1 - joint_gate_fid) / 8)
    new_elem_2 = (joint_gate_fid * (same_report * (kept_elem_1 * meas_elem_2 + kept_elem_2 * meas_elem_1)
                                    + opposite_report * (kept_elem_1 * meas_elem_4 + kept_elem_2 * meas_elem_3))
                  + (1 - joint_gate_fid) / 8)
    new_elem_3 = (joint_gate_fid * (same_report * (kept_elem_3 * meas_elem_3 + kept_elem_4 * meas_elem_4)
                                    + opposite_report * (kept_elem_3 * meas_elem_1 + kept_elem_4 * meas_elem_2))
                  + (1 - joint_gate_fid) / 8)
    new_elem_4 = (joint_gate_fid * (same_report * (kept_elem_3 * meas_elem_4 + kept_elem_4 * meas_elem_3)
                                    + opposite_report * (kept_elem_3 * meas_elem_2 + kept_elem_4 * meas_elem_1))
                  + (1 - joint_gate_fid) / 8)

    bds_elems = np.stack([new_elem_1, new_elem_2, new_elem_3, new_elem_4], axis=-1)
    bds_elems = bds_elems / np.maximum(p_succ, _EPS)[..., None]
    return p_succ, bds_elems


# ---------------------------------------------------------------------------
# entanglement swapping
# ---------------------------------------------------------------------------

def swap(left_state: ArrayLike, right_state: ArrayLike,
         gate_fid: ArrayLike = 1.0, meas_fid: ArrayLike = 1.0,
         is_twirled: bool = False) -> NDArray[np.float64]:
    """Entanglement swapping of two adjacent pairs, in closed form.

    Transcribes ``EntanglementSwappingBDS.swapping_res``: a Bell-state
    measurement at the middle node fuses ``left_state`` (A–B) and
    ``right_state`` (B–C) into one pair (A–C).

    The ``c_I``/``c_X``/``c_Y``/``c_Z`` intermediates are the probabilities that
    the two input errors compose to each Pauli on the output pair; imperfect
    gates and measurements then mix those channels.

    Args:
        left_state: Bell-diagonal state of the left pair, shape ``(..., 4)``.
        right_state: Bell-diagonal state of the right pair, shape ``(..., 4)``.
        gate_fid: probability the swapping node's gate is ideal.
        meas_fid: probability each of its two single-qubit measurements is correct.
        is_twirled: twirl both inputs before, and the output after, as upstream
            does when the protocol is configured for Werner states.

    Returns:
        The Bell-diagonal state of the resulting end-to-end pair, shape
        ``(..., 4)``.

    Note:
        This is the state **conditioned on the BSM succeeding**.  The
        probability of that (detector efficiency, the 1/2 ceiling of a
        linear-optics BSM) belongs to the link model, not here — see
        :func:`qnet_autopilot.physics.link.bsm_success_probability`.
    """
    left = _as_bds(left_state)
    right = _as_bds(right_state)
    if is_twirled:
        left = twirl(left)
        right = twirl(right)

    left_elem_1, left_elem_2, left_elem_3, left_elem_4 = np.moveaxis(left, -1, 0)
    right_elem_1, right_elem_2, right_elem_3, right_elem_4 = np.moveaxis(right, -1, 0)

    gate_fid = np.asarray(gate_fid, dtype=float)
    meas_fid = np.asarray(meas_fid, dtype=float)

    c_I = (left_elem_1 * right_elem_1 + left_elem_2 * right_elem_2
           + left_elem_3 * right_elem_3 + left_elem_4 * right_elem_4)
    c_X = (left_elem_1 * right_elem_2 + left_elem_2 * right_elem_1
           + left_elem_3 * right_elem_4 + left_elem_4 * right_elem_3)
    c_Y = (left_elem_1 * right_elem_4 + left_elem_4 * right_elem_1
           + left_elem_2 * right_elem_3 + left_elem_3 * right_elem_2)
    c_Z = (left_elem_1 * right_elem_3 + left_elem_3 * right_elem_1
           + left_elem_2 * right_elem_4 + left_elem_4 * right_elem_2)

    new_elem_1 = gate_fid * (meas_fid ** 2 * c_I + meas_fid * (1 - meas_fid) * (c_X + c_Z)
                             + (1 - meas_fid) ** 2 * c_Y) + (1 - gate_fid) / 4
    new_elem_2 = gate_fid * (meas_fid ** 2 * c_X + meas_fid * (1 - meas_fid) * (c_I + c_Y)
                             + (1 - meas_fid) ** 2 * c_Z) + (1 - gate_fid) / 4
    new_elem_3 = gate_fid * (meas_fid ** 2 * c_Z + meas_fid * (1 - meas_fid) * (c_I + c_Y)
                             + (1 - meas_fid) ** 2 * c_X) + (1 - gate_fid) / 4
    new_elem_4 = gate_fid * (meas_fid ** 2 * c_Y + meas_fid * (1 - meas_fid) * (c_X + c_Z)
                             + (1 - meas_fid) ** 2 * c_I) + (1 - gate_fid) / 4

    if is_twirled:
        return werner(new_elem_1)
    return np.stack([new_elem_1, new_elem_2, new_elem_3, new_elem_4], axis=-1)
