"""A deliberately non-functional ``stim`` placeholder.

``stim`` is a real Clifford/stabilizer simulator; unlike ``gmpy2`` and the
qutip slice SeQUeNCe uses, it cannot be stood in for with fifty lines of numpy,
and pretending otherwise would silently produce wrong physics.

But SeQUeNCe imports it *eagerly*: ``sequence.kernel.quantum_manager.__init__``
pulls in ``QuantumManagerStabilizer``, so a missing ``stim`` takes the whole
kernel down — including the ket and density-matrix backends, which need
nothing from it.

This module therefore provides import-time names only.  Every one of them
raises :class:`StimUnavailable` the moment it is actually used, so:

* ``QuantumManagerKet`` and ``QuantumManagerDensity`` work normally;
* selecting the stabilizer formalism fails immediately, with a message saying
  to install ``stim``, instead of producing plausible nonsense.

Nothing on the repeater path (entanglement generation, purification, swapping,
memory decoherence) touches the stabilizer backend.
"""

from __future__ import annotations

__all__ = ["StimUnavailable", "TableauSimulator", "Tableau", "PauliString", "Circuit"]

_MESSAGE = (
    "the stabilizer (Clifford) backend needs the real `stim` package, which is "
    "not installed. Install it with `pip install stim`, or use the ket / density "
    "quantum managers instead — qnet_autopilot's physics layer only uses those."
)


class StimUnavailable(RuntimeError):
    """Raised when code reaches for stim functionality that is not installed."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name}: {_MESSAGE}")


class _Tombstone:
    """Import-time placeholder that refuses to be constructed or used."""

    def __init__(self, *args, **kwargs) -> None:
        raise StimUnavailable(type(self).__name__)

    def __class_getitem__(cls, item):  # pragma: no cover - typing only
        return cls

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:  # pragma: no cover
        super().__init_subclass__(**kwargs)


class TableauSimulator(_Tombstone):
    pass


class Tableau(_Tombstone):
    pass


class PauliString(_Tombstone):
    pass


class Circuit(_Tombstone):
    pass
