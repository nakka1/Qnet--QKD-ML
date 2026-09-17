"""Make SeQUeNCe importable in environments where its dependency tree is not.

``qnet_autopilot`` is written against the *real* SeQUeNCe public API, so on a
normal machine you should just::

    pip install sequence

and this module does nothing at all.  It exists for the environments that
cannot reach PyPI (locked-down CI runners, air-gapped clusters, the sandbox
this package was developed in), where SeQUeNCe's optional numerical
dependencies — ``gmpy2``, ``qutip``, ``qutip-qip``, ``stim`` — are unavailable
even though SeQUeNCe's own source can be fetched from GitHub.

:func:`bootstrap` fills the gaps, in this order:

1. For each of ``gmpy2``, ``qutip`` and ``qutip_qip``: import the real package
   if it is installed; otherwise register a shim in :data:`sys.modules`.  The
   shims cover only the API surface SeQUeNCe actually calls — see
   :mod:`._gmpy2_shim` and :mod:`._qutip_shim` for the exact list and the
   conventions they match.
2. Import ``sequence``.  If that fails, put a source checkout on ``sys.path``:
   the ``SEQUENCE_SRC`` environment variable if set, else ``vendor/SeQUeNCe``
   beside the repository root.

The stabilizer formalism (``sequence.kernel.quantum_state.stabilizer`` and its
quantum manager) needs ``stim``, which is **not** shimmed: it is a genuine
Clifford simulator, not a thin adapter, and faking it would produce plausible
nonsense.  SeQUeNCe imports it eagerly from ``kernel.quantum_manager``, though,
so a missing ``stim`` would take the ket and density backends down with it.
:mod:`._stim_tombstone` supplies import-time names that raise on first use —
the repeater path never touches them.

On Python 3.11, :func:`bootstrap` also back-ports ``typing.override`` from
``typing_extensions``.  SeQUeNCe declares ``requires-python >= 3.12`` and uses
that decorator throughout ``entanglement_management``; it is the only 3.12-ism
in the tree (the whole package byte-compiles under 3.11), and the decorator is
a no-op at runtime.  On 3.12+ nothing is patched.

Everything here is opt-in and idempotent::

    from qnet_autopilot.compat import bootstrap
    status = bootstrap()
    print(status.summary())
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["bootstrap", "BootstrapStatus", "SHIMMABLE"]

#: Dependencies this package knows how to stand in for.
SHIMMABLE = ("gmpy2", "qutip", "qutip_qip")

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BootstrapStatus:
    """What :func:`bootstrap` had to do, so callers can report it honestly."""

    real: list[str] = field(default_factory=list)
    shimmed: list[str] = field(default_factory=list)
    sequence_path: str | None = None
    sequence_version: str | None = None
    stim_available: bool = False
    typing_override_backported: bool = False

    @property
    def fully_native(self) -> bool:
        """True when nothing was shimmed — results are plain SeQUeNCe."""
        return not self.shimmed

    def summary(self) -> str:
        lines = [f"SeQUeNCe {self.sequence_version or '?'} from {self.sequence_path}"]
        if self.real:
            lines.append(f"  native deps : {', '.join(sorted(self.real))}")
        if self.shimmed:
            lines.append(f"  shimmed deps: {', '.join(sorted(self.shimmed))}")
        lines.append(f"  stim (stabilizer formalism): "
                     f"{'available' if self.stim_available else 'absent — Clifford backend disabled'}")
        if self.typing_override_backported:
            lines.append("  typing.override back-ported from typing_extensions (Python < 3.12)")
        return "\n".join(lines)


_STATUS: BootstrapStatus | None = None


def _install_gmpy2_shim() -> None:
    from . import _gmpy2_shim

    sys.modules["gmpy2"] = _gmpy2_shim


def _install_qutip_shims(which: set[str]) -> None:
    from . import _qutip_shim

    if "qutip" in which:
        qutip = types.ModuleType("qutip")
        qutip.Qobj = _qutip_shim.Qobj
        qutip.__doc__ = _qutip_shim.__doc__
        sys.modules["qutip"] = qutip

    if "qutip_qip" in which:
        qutip_qip = types.ModuleType("qutip_qip")
        qutip_qip.__path__ = []  # mark as a package so submodules can be registered
        qutip_qip.__doc__ = _qutip_shim.__doc__

        circuit = types.ModuleType("qutip_qip.circuit")
        circuit.QubitCircuit = _qutip_shim.QubitCircuit

        operations = types.ModuleType("qutip_qip.operations")
        operations.Gate = _qutip_shim.Gate
        operations.gate_sequence_product = _qutip_shim.gate_sequence_product
        operations.expand_gate = _qutip_shim.expand_gate

        qutip_qip.circuit = circuit
        qutip_qip.operations = operations

        sys.modules["qutip_qip"] = qutip_qip
        sys.modules["qutip_qip.circuit"] = circuit
        sys.modules["qutip_qip.operations"] = operations


def _backport_typing_override() -> bool:
    """Give Python 3.11 a ``typing.override``; return True if we had to.

    SeQUeNCe's ``entanglement_management`` package decorates overridden methods
    with ``typing.override``, which only exists from 3.12.  The decorator has no
    runtime effect beyond setting ``__override__``, so the ``typing_extensions``
    version is a faithful substitute.
    """
    import typing

    if hasattr(typing, "override"):
        return False
    try:
        from typing_extensions import override
    except ImportError:  # pragma: no cover - typing_extensions is near-universal
        def override(method):  # type: ignore[misc]
            try:
                method.__override__ = True
            except (AttributeError, TypeError):
                pass
            return method
    typing.override = override  # type: ignore[attr-defined]
    return True


def _candidate_sequence_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("SEQUENCE_SRC")
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.append(_REPO_ROOT / "vendor" / "SeQUeNCe")
    roots.append(Path.cwd() / "vendor" / "SeQUeNCe")
    return roots


def bootstrap(force: bool = False) -> BootstrapStatus:
    """Ensure ``import sequence`` works; shim only what is missing.

    Args:
        force: re-run even if a previous call already succeeded.

    Returns:
        BootstrapStatus: which dependencies are real, which are shims, and
        where SeQUeNCe was imported from.

    Raises:
        ImportError: if SeQUeNCe itself can be found neither on ``sys.path``
            nor in any candidate source checkout.
    """
    global _STATUS
    if _STATUS is not None and not force:
        return _STATUS

    status = BootstrapStatus()

    missing: set[str] = set()
    for name in SHIMMABLE:
        if name in sys.modules and not getattr(sys.modules[name], "__file__", None):
            # already shimmed by an earlier call
            status.shimmed.append(name)
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            missing.add(name)
        else:
            status.real.append(name)

    if "gmpy2" in missing:
        _install_gmpy2_shim()
        status.shimmed.append("gmpy2")
    qutip_missing = missing & {"qutip", "qutip_qip"}
    if qutip_missing:
        _install_qutip_shims(qutip_missing)
        status.shimmed.extend(sorted(qutip_missing))

    try:
        importlib.import_module("stim")
    except ImportError:
        from . import _stim_tombstone

        sys.modules["stim"] = _stim_tombstone
        status.stim_available = False
    else:
        status.stim_available = True

    status.typing_override_backported = _backport_typing_override()

    sequence = _import_sequence()
    status.sequence_path = str(Path(sequence.__file__).parent)
    status.sequence_version = getattr(sequence, "__version__", None)

    _STATUS = status
    return status


def _import_sequence():
    try:
        return importlib.import_module("sequence")
    except ImportError:
        pass

    tried: list[str] = []
    for root in _candidate_sequence_roots():
        tried.append(str(root))
        if (root / "sequence" / "__init__.py").is_file():
            sys.path.insert(0, str(root))
            return importlib.import_module("sequence")

    raise ImportError(
        "SeQUeNCe is not importable. Either `pip install sequence`, or point "
        "SEQUENCE_SRC at a checkout of https://github.com/sequence-toolbox/SeQUeNCe "
        f"(tried: {', '.join(tried)})."
    )
