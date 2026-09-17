"""Pure-numpy stand-in for the slice of ``qutip`` / ``qutip_qip`` SeQUeNCe uses.

SeQUeNCe uses qutip for one job: turning a list of gates into a single
:math:`2^N \\times 2^N` unitary.  Two call sites need it —
``sequence.components.circuit.Circuit.get_unitary_matrix`` and
``sequence.kernel.quantum_manager.utils.swap_qubits`` — and between them they
touch only:

* ``qutip.Qobj`` — wrapping a matrix, and ``.full()`` to get it back
* ``qutip_qip.circuit.QubitCircuit`` — ``user_gates``, ``add_gate``, ``propagators``
* ``qutip_qip.operations.Gate`` — a named gate with ``targets`` / ``controls``
* ``qutip_qip.operations.gate_sequence_product`` — the ordered product

Everything else in qutip (solvers, superoperators, the Qobj algebra) is out of
scope and deliberately absent.

**Conventions**, matched to qutip so the produced unitaries are identical:

* Qubit 0 is the *leftmost* factor of the Kronecker product, i.e. the most
  significant bit of the state index.
* ``gate_sequence_product([U1, U2, ..., Un])`` returns ``Un @ ... @ U2 @ U1``:
  the list is in circuit (time) order, the product is in operator order.
* For controlled gates the ``targets`` of the underlying matrix are ordered
  ``controls + targets``, so the control occupies the high bits.

Install the real ``qutip`` and ``qutip-qip`` whenever the environment allows;
:mod:`qnet_autopilot.compat` prefers them automatically.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["Qobj", "QubitCircuit", "Gate", "gate_sequence_product", "expand_gate"]

_SQRT_HALF = 1.0 / np.sqrt(2.0)

_BUILTIN_GATES: dict[str, NDArray[np.complex128]] = {
    "SNOT": _SQRT_HALF * np.array([[1, 1], [1, -1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "CNOT": np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype=complex),
    "CZ": np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]], dtype=complex),
    "SWAP": np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=complex),
    "TOFFOLI": np.eye(8, dtype=complex),
}
_BUILTIN_GATES["TOFFOLI"][6:8, 6:8] = np.array([[0, 1], [1, 0]], dtype=complex)

#: Gates whose matrix is built from a parameter rather than looked up.
_PARAMETERIZED = {"PHASEGATE"}


def _phase_gate(theta: float) -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * float(theta))]], dtype=complex)


class Qobj:
    """Minimal quantum object: a dense matrix that remembers its dims."""

    __slots__ = ("_data", "dims")

    def __init__(self, data, dims=None) -> None:
        if isinstance(data, Qobj):
            data = data.full()
        self._data: NDArray[np.complex128] = np.asarray(data, dtype=complex)
        if self._data.ndim == 1:
            self._data = self._data.reshape(-1, 1)
        if dims is None:
            rows, cols = self._data.shape
            dims = [[rows], [cols]]
        self.dims = dims

    def full(self) -> NDArray[np.complex128]:
        """Return the dense matrix as a numpy array (qutip API)."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape  # type: ignore[return-value]

    def __mul__(self, other):
        if isinstance(other, Qobj):
            return Qobj(self._data @ other._data, dims=self.dims)
        if np.isscalar(other):
            return Qobj(self._data * other, dims=self.dims)
        return Qobj(self._data @ np.asarray(other, dtype=complex), dims=self.dims)

    def __rmul__(self, other):
        if np.isscalar(other):
            return Qobj(self._data * other, dims=self.dims)
        return Qobj(np.asarray(other, dtype=complex) @ self._data, dims=self.dims)

    __matmul__ = __mul__

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Qobj(shape={self._data.shape}, dims={self.dims})"


def expand_gate(matrix, num_qubits: int, targets: Sequence[int]) -> NDArray[np.complex128]:
    """Lift a k-qubit gate acting on ``targets`` to the full ``num_qubits`` space.

    ``targets[i]`` is the circuit qubit that plays the role of qubit ``i`` of
    ``matrix``.  Qubit 0 is the most significant bit, matching qutip.

    The implementation contracts the gate's input legs against an identity
    tensor rather than assembling Kronecker products, so it handles
    non-adjacent and out-of-order targets (a control below its target, a SWAP
    across the register) without special cases.
    """
    matrix = np.asarray(matrix, dtype=complex)
    targets = list(targets)
    k = len(targets)
    if matrix.shape != (2 ** k, 2 ** k):
        raise ValueError(f"gate of shape {matrix.shape} cannot act on {k} qubits")
    if len(set(targets)) != k:
        raise ValueError(f"repeated qubit index in targets={targets}")
    if any(not 0 <= t < num_qubits for t in targets):
        raise ValueError(f"targets={targets} out of range for {num_qubits} qubits")
    if k == num_qubits and targets == list(range(num_qubits)):
        return matrix

    n = num_qubits
    identity = np.eye(2 ** n, dtype=complex).reshape([2] * (2 * n))
    gate_tensor = matrix.reshape([2] * (2 * k))

    # Contract the gate's input legs with the identity's output legs at `targets`.
    contracted = np.tensordot(gate_tensor, identity,
                              axes=(list(range(k, 2 * k)), targets))

    # tensordot puts the gate's output legs first; restore the canonical
    # ordering (out_0..out_{n-1}, in_0..in_{n-1}).
    remaining = [axis for axis in range(2 * n) if axis not in targets]
    permutation = [0] * (2 * n)
    for i, target in enumerate(targets):
        permutation[target] = i
    for j, axis in enumerate(remaining):
        permutation[axis] = k + j

    return np.transpose(contracted, permutation).reshape(2 ** n, 2 ** n)


class Gate:
    """A named gate with its target and control qubits (qutip_qip API)."""

    def __init__(self, name: str, targets=None, controls=None, arg_value=None) -> None:
        self.name = name
        self.targets = _as_list(targets)
        self.controls = _as_list(controls)
        self.arg_value = arg_value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Gate({self.name!r}, targets={self.targets}, "
                f"controls={self.controls}, arg_value={self.arg_value})")


def _as_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


class QubitCircuit:
    """An ordered list of gates over ``N`` qubits (qutip_qip API subset)."""

    def __init__(self, N: int, **_ignored) -> None:
        self.N = int(N)
        self.gates: list[Gate] = []
        self.user_gates: dict[str, Callable[..., Qobj]] = {}

    def add_gate(self, gate, targets=None, controls=None, arg_value=None) -> None:
        """Append a gate.

        ``gate`` is either a :class:`Gate` instance or a gate name, in which
        case ``targets`` may also be given positionally as an int or a list.
        """
        if isinstance(gate, Gate):
            self.gates.append(gate)
        else:
            self.gates.append(Gate(gate, targets=targets, controls=controls,
                                   arg_value=arg_value))

    def resolve_gate(self, gate: Gate) -> tuple[NDArray[np.complex128], list[int]]:
        """Return ``(matrix, qubit_order)`` for a single gate."""
        name = gate.name
        qubits = gate.controls + gate.targets

        if name in self.user_gates:
            factory = self.user_gates[name]
            matrix = factory() if gate.arg_value is None else factory(gate.arg_value)
            return np.asarray(Qobj(matrix).full(), dtype=complex), qubits

        if name in _PARAMETERIZED:
            if name == "PHASEGATE":
                return _phase_gate(gate.arg_value), qubits
            raise NotImplementedError(f"parameterized gate {name!r} not supported")

        if name in _BUILTIN_GATES:
            return _BUILTIN_GATES[name], qubits

        raise NotImplementedError(
            f"gate {name!r} is not in the compatibility shim's gate set; "
            f"install the real qutip-qip if you need it"
        )

    def propagators(self) -> list[Qobj]:
        """Full-register unitary for each gate, in circuit order."""
        propagators: list[Qobj] = []
        for gate in self.gates:
            matrix, qubits = self.resolve_gate(gate)
            propagators.append(Qobj(expand_gate(matrix, self.N, qubits)))
        return propagators


def gate_sequence_product(unitaries: Iterable[Qobj]) -> Qobj:
    """Compose a time-ordered list of unitaries into one.

    ``[U1, U2, ..., Un]`` (circuit order) becomes ``Un @ ... @ U2 @ U1``.
    """
    result: Qobj | None = None
    for unitary in unitaries:
        unitary = unitary if isinstance(unitary, Qobj) else Qobj(unitary)
        result = unitary if result is None else Qobj(unitary.full() @ result.full())
    if result is None:
        raise ValueError("gate_sequence_product() needs at least one unitary")
    return result
