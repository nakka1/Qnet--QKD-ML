#!/usr/bin/env python3
"""Render the HTML reports from the CSVs already in ``runs/``.

Separate from the evaluation scripts on purpose: rendering is cheap and the
measurements are not, so the prose and the figures can be reworked without
re-running anything.  Every number in the output comes from a file on disk — if
a results file is missing, the renderer raises rather than filling a gap.

    python scripts/render_reports.py --phase 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", type=int, nargs="+", default=[2, 3, 4],
                        choices=[2, 3, 4])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    for phase in args.phase:
        if phase == 2:
            from qnet_autopilot.agents import render_phase2_report
            path = render_phase2_report(args.runs / "agent", args.out / "phase2.html")
        elif phase == 3:
            from qnet_autopilot.compensation import render_phase3_report
            path = render_phase3_report(args.runs / "compensator",
                                        args.out / "phase3.html")
        else:
            from qnet_autopilot.allocation import render_phase4_report
            path = render_phase4_report(args.runs / "allocator",
                                        args.out / "phase4.html")
        print(f"phase {phase}: {path}  ({path.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
