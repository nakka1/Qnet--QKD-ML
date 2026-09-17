"""Shared fixtures and the SeQUeNCe bootstrap for the whole test suite.

Importing this module puts the package under test on ``sys.path`` and runs
:func:`qnet_autopilot.compat.bootstrap`, so tests can ``import sequence``
whether SeQUeNCe was pip-installed or is sitting in ``vendor/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qnet_autopilot.compat import bootstrap  # noqa: E402

BOOTSTRAP_STATUS = bootstrap()


@pytest.fixture
def rng():
    """A seeded generator, so every stochastic test is reproducible."""
    import numpy as np

    return np.random.default_rng(20260911)


@pytest.fixture
def tmp_path():
    """A scratch directory per test.

    pytest ships one; the minimal runner in ``scripts/minipytest.py`` does not,
    and a test that only runs under one of the two runners is a test that stops
    being run.
    """
    import shutil
    import tempfile
    from pathlib import Path as _Path

    directory = tempfile.mkdtemp(prefix="qnet-test-")
    try:
        yield _Path(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def bds_timeline():
    """A SeQUeNCe timeline whose quantum manager uses the Bell-diagonal formalism."""
    from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
    from sequence.kernel.quantum_manager import QuantumManager
    from sequence.kernel.timeline import Timeline

    previous = QuantumManager.get_active_formalism()
    QuantumManager.set_global_manager_formalism(BELL_DIAGONAL_STATE_FORMALISM)
    timeline = Timeline(formalism=BELL_DIAGONAL_STATE_FORMALISM)
    timeline.show_progress = False
    try:
        yield timeline
    finally:
        QuantumManager.set_global_manager_formalism(previous)
