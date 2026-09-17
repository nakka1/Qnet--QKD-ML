"""Physics and data layer for learned quantum-network control.

Built on SeQUeNCe.  Four pieces, in dependency order:

:mod:`~qnet_autopilot.compat`
    Makes ``import sequence`` work even where its optional numerical
    dependencies cannot be installed.  A no-op on a normal machine.

:mod:`~qnet_autopilot.physics`
    Stateless, vectorised models: Bell-diagonal purification and swapping
    (transcribed from SeQUeNCe and tested against it), fibre loss and
    polarisation drift, memory decoherence, link budgets and key rates.

:mod:`~qnet_autopilot.network`
    Scenario topologies and their translation into SeQUeNCe configurations.

:mod:`~qnet_autopilot.datagen`
    Dataset generators, each carrying its own provenance.

:mod:`~qnet_autopilot.validation`
    The cross-checks that say which of the above can be trusted, and where.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["compat", "datagen", "network", "physics", "validation", "__version__"]


def __getattr__(name: str):
    """Import submodules lazily.

    ``compat.bootstrap()`` touches ``sys.modules`` and puts a source checkout on
    the path; importing the whole package should not do that as a side effect of
    ``import qnet_autopilot``.
    """
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
