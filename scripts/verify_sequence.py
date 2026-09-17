#!/usr/bin/env python3
"""Run SeQUeNCe's own test suite under the compatibility shims.

The shims are only trustworthy if the code they stand under still passes its
own tests.  This runs that suite and reports what passed, what failed, and
which failures are expected (the stabilizer backend needs the real ``stim``;
the CLI tests need ``typer``).

    python scripts/verify_sequence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qnet_autopilot.compat import bootstrap  # noqa: E402


def main() -> int:
    status = bootstrap()
    print(status.summary(), end="\n\n")

    sequence_root = Path(status.sequence_path).parent
    tests = sequence_root / "tests"
    if not tests.is_dir():
        print(f"no test suite at {tests} — this is an installed wheel, not a checkout")
        return 0

    # SeQUeNCe's tests load JSON fixtures by paths relative to its own root.
    os.chdir(sequence_root)

    try:
        import pytest  # noqa: F401

        import subprocess

        return subprocess.call([sys.executable, "-m", "pytest", "tests", "-q"])
    except ImportError:
        import minipytest

        print("pytest not available — using the bundled runner\n")
        return minipytest.run(["tests"])


if __name__ == "__main__":
    raise SystemExit(main())
