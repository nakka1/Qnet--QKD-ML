"""The compatibility layer must be transparent, or it is worse than useless.

A shim that quietly changes a number is far more dangerous than a missing
dependency: the missing dependency stops you, the wrong number does not.  These
tests pin the two shims to the exact semantics of what they replace —
gmpy2's precision arithmetic and qutip's gate composition, including its
qubit-ordering and multiplication-order conventions, which are the two things
easiest to get backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.compat import BootstrapStatus, bootstrap
from qnet_autopilot.compat._gmpy2_shim import ceil, floor, get_context, mpfr, mpz, rint
from qnet_autopilot.compat._qutip_shim import (Gate, Qobj, QubitCircuit, expand_gate,
                                               gate_sequence_product)

SQRT_HALF = 1 / np.sqrt(2)
H = SQRT_HALF * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


class TestBootstrap:

    def test_bootstrap_is_idempotent_and_finds_sequence(self):
        first = bootstrap()
        second = bootstrap()
        assert isinstance(first, BootstrapStatus)
        assert first is second
        assert first.sequence_path is not None

    def test_status_summary_names_what_was_shimmed(self):
        status = bootstrap()
        summary = status.summary()
        assert "SeQUeNCe" in summary
        for name in status.shimmed:
            assert name in summary

    def test_sequence_actually_imports(self):
        bootstrap()
        import sequence.components.memory  # noqa: F401
        import sequence.entanglement_management.purification  # noqa: F401
        import sequence.kernel.timeline  # noqa: F401


class TestGmpy2Shim:

    def test_precision_is_settable_in_bits(self):
        get_context().precision = 80
        assert get_context().precision == 80

    def test_mpfr_accepts_the_types_sequence_passes_it(self):
        assert float(mpfr(3)) == 3.0
        assert float(mpfr(3.5)) == 3.5
        assert float(mpfr("1e12")) == 1e12
        assert float(mpfr(np.float64(2.25))) == 2.25

    def test_mpz_is_an_unbounded_integer(self):
        assert mpz(2 ** 70) == 2 ** 70
        assert isinstance(mpz(5.9), int)

    def test_rounding_matches_gmpy2_semantics(self):
        assert float(floor(mpfr("2.7"))) == 2.0
        assert float(ceil(mpfr("2.1"))) == 3.0
        # rint is round-half-to-even, not round-half-away-from-zero.
        assert float(rint(mpfr("2.5"))) == 2.0
        assert float(rint(mpfr("3.5"))) == 4.0

    def test_timing_arithmetic_is_exact_at_simulation_scale(self):
        """The reason SeQUeNCe reaches for gmpy2 at all.

        ``QuantumChannel`` converts between picosecond timestamps and time-bin
        indices as ``time * frequency / 1e12``.  Far into a long simulation
        that product runs past float64's 53-bit mantissa, so the shim has to
        carry real extra precision rather than quietly widening a double.  The
        reference is exact rational arithmetic.
        """
        from fractions import Fraction

        get_context().precision = 80
        picoseconds_per_second = 10 ** 12

        for time_ps, frequency_hz in ((123_456_789_012_345_678, 2_500_000_000),
                                      (999_999_999_999_999_999, 1_000_000_007),
                                      (10 ** 20 + 7, 3_333_333_333)):
            exact = Fraction(time_ps * frequency_hz, picoseconds_per_second)
            shimmed = mpfr(time_ps) * mpfr(frequency_hz) / mpfr(picoseconds_per_second)
            # 24 significant digits against a ~10^17 value: sub-picosecond.
            assert abs(Fraction(shimmed) - exact) < Fraction(1, 1000)

    def test_time_bin_index_matches_exact_arithmetic(self):
        get_context().precision = 80
        from fractions import Fraction

        time_ps = 987_654_321_098_765_432
        frequency_hz = 1_250_000_000  # 1.25e9 / 1e12 = 1/800 bins per picosecond

        time_bin = mpz(rint(mpfr(time_ps) * mpfr(frequency_hz) / mpfr(10 ** 12)))
        # round() on a Fraction is round-half-to-even, the same rule as rint.
        assert time_bin == round(Fraction(time_ps, 800))


class TestQutipShim:

    def test_qobj_round_trips_a_matrix(self):
        obj = Qobj(H, dims=[[2], [2]])
        assert np.allclose(obj.full(), H)
        assert obj.dims == [[2], [2]]

    def test_single_qubit_gate_expands_with_qubit_zero_most_significant(self):
        """qutip's convention: qubit 0 is the leftmost Kronecker factor."""
        assert np.allclose(expand_gate(X, 2, [0]), np.kron(X, np.eye(2)))
        assert np.allclose(expand_gate(X, 2, [1]), np.kron(np.eye(2), X))

    def test_two_qubit_gate_on_adjacent_qubits(self):
        assert np.allclose(expand_gate(CNOT, 2, [0, 1]), CNOT)

    def test_reversed_control_and_target(self):
        expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                            dtype=complex)
        assert np.allclose(expand_gate(CNOT, 2, [1, 0]), expected)

    def test_non_adjacent_targets(self):
        """A SWAP across a spectator qubit — the case a Kronecker-product
        implementation gets wrong."""
        result = expand_gate(SWAP, 3, [0, 2])
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    source = (a << 2) | (b << 1) | c
                    target = (c << 2) | (b << 1) | a
                    assert result[target, source] == pytest.approx(1.0)

    def test_expand_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            expand_gate(CNOT, 2, [0])
        with pytest.raises(ValueError):
            expand_gate(SWAP, 3, [1, 1])
        with pytest.raises(ValueError):
            expand_gate(X, 2, [5])

    def test_gate_sequence_product_applies_in_circuit_order(self):
        """``[U1, U2]`` means U1 first, so the product is ``U2 @ U1``."""
        first = Qobj(H)
        second = Qobj(X)
        assert np.allclose(gate_sequence_product([first, second]).full(), X @ H)

    def test_gate_sequence_product_needs_at_least_one(self):
        with pytest.raises(ValueError):
            gate_sequence_product([])

    def test_circuit_builds_a_bell_pair_unitary(self):
        circuit = QubitCircuit(2)
        circuit.add_gate("SNOT", 0)
        circuit.add_gate("CNOT", controls=0, targets=1)
        unitary = gate_sequence_product(circuit.propagators()).full()

        zero_zero = np.array([1, 0, 0, 0], dtype=complex)
        bell = unitary @ zero_zero
        assert np.allclose(bell, [SQRT_HALF, 0, 0, SQRT_HALF])

    def test_circuit_accepts_gate_objects(self):
        circuit = QubitCircuit(3)
        circuit.add_gate(Gate("SWAP", targets=[0, 2]))
        unitary = gate_sequence_product(circuit.propagators()).full()
        assert np.allclose(unitary, expand_gate(SWAP, 3, [0, 2]))

    def test_user_gates_are_looked_up_by_name(self):
        circuit = QubitCircuit(1)
        circuit.user_gates = {"MYGATE": lambda: Qobj(H)}
        circuit.add_gate("MYGATE", 0)
        assert np.allclose(gate_sequence_product(circuit.propagators()).full(), H)

    def test_phase_gate_takes_its_argument(self):
        circuit = QubitCircuit(1)
        circuit.add_gate("PHASEGATE", 0, arg_value=np.pi / 3)
        unitary = gate_sequence_product(circuit.propagators()).full()
        assert np.allclose(unitary, np.diag([1, np.exp(1j * np.pi / 3)]))

    def test_unknown_gate_fails_loudly(self):
        circuit = QubitCircuit(1)
        circuit.add_gate("NOTAGATE", 0)
        with pytest.raises(NotImplementedError):
            circuit.propagators()

    def test_every_produced_unitary_is_unitary(self, rng):
        circuit = QubitCircuit(3)
        circuit.add_gate("SNOT", 0)
        circuit.add_gate("CNOT", controls=0, targets=2)
        circuit.add_gate("SWAP", [1, 2])
        circuit.add_gate("PHASEGATE", 1, arg_value=0.7)
        circuit.add_gate("TOFFOLI", controls=[0, 1], targets=2)
        unitary = gate_sequence_product(circuit.propagators()).full()
        assert np.allclose(unitary.conj().T @ unitary, np.eye(8), atol=1e-12)


class TestSequenceCircuitThroughTheShim:
    """The shim's real job: SeQUeNCe's own ``Circuit`` must produce right answers."""

    def test_bell_pair_circuit(self):
        bootstrap()
        from sequence.components.circuit import Circuit

        circuit = Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        unitary = circuit.get_unitary_matrix()
        bell = unitary @ np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(bell, [SQRT_HALF, 0, 0, SQRT_HALF])

    def test_empty_circuit_is_the_identity(self):
        bootstrap()
        from sequence.components.circuit import Circuit

        assert np.allclose(Circuit(3).get_unitary_matrix(), np.eye(8))

    def test_swap_qubits_helper(self):
        bootstrap()
        from sequence.kernel.quantum_manager.utils import swap_qubits

        keys, matrix = swap_qubits([7, 3], [3, 7])
        assert keys == [3, 7]
        assert np.allclose(matrix, SWAP)
