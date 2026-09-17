#!/usr/bin/env python3
"""Score every compensator in the closed loop, on held-out fibre runs.

Writes CSVs the report is generated from.  Test seeds are disjoint from the
seeds the models were fitted on, and the alpha-beta baseline is re-tuned at
every operating point so the comparison is against a *tuned* classical
tracker rather than a straw man.

    python scripts/evaluate_phase3.py --out runs/compensator
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task3 import (FEATURE_SPEC, TEST_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS,  # noqa: E402
                   make_control, make_trajectories)
from qnet_autopilot.compensation import (AlphaBetaFilter, LinearExtrapolation,  # noqa: E402
                                         NoCompensation, Persistence, run_closed_loop)
from qnet_autopilot.compensation.train import (build_dataset, chronological_split,  # noqa: E402
                                               fit_ridge)

#: Grid the alpha-beta baseline is tuned over, at every operating point.
ALPHA_BETA_GRID = list(itertools.product([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0],
                                         [0.0, 0.005, 0.02, 0.05, 0.1]))

#: Tuning runs are short on purpose. The grid is searched at every operating
#: point in two sweeps, so a long tuning trajectory costs more wall-clock than
#: the whole rest of the evaluation — and eight seconds is already hundreds of
#: vibration periods and dozens of thermal correlation lengths, which is plenty
#: to rank filter gains.
TUNING_DURATION_S = 10.0


def tune_alpha_beta(trajectories, control, seed: int = 7) -> AlphaBetaFilter:
    """Grid-search alpha and beta on validation runs, by closed-loop QBER.

    Tuned per operating point because the right amount of smoothing depends on
    the latency: a filter tuned at 20 ms and run at 80 ms is a straw man, and
    beating one proves nothing.
    """
    best, best_score = None, np.inf
    for alpha, beta in ALPHA_BETA_GRID:
        candidate = AlphaBetaFilter(alpha=alpha, beta=beta)
        score = float(np.mean([
            run_closed_loop(trajectory, candidate, control, rng=seed + index
                            ).summary()["mean_qber"]
            for index, trajectory in enumerate(trajectories)]))
        if score < best_score:
            best, best_score = (alpha, beta), score
    return AlphaBetaFilter(alpha=best[0], beta=best[1],
                           name=f"filtro alfa-beta (a={best[0]:g}, b={best[1]:g})")


def score(trajectories, compensator, control, seed: int = 7000) -> dict:
    """Average a compensator's summary over several runs."""
    summaries = [run_closed_loop(trajectory, compensator, control, rng=seed + index).summary()
                 for index, trajectory in enumerate(trajectories)]
    aggregate = {key: float(np.mean([item[key] for item in summaries]))
                 for key in summaries[0] if key != "compensator"}
    aggregate["compensator"] = compensator.name
    return aggregate


def recovery(trajectories, compensator, control, seed: int = 7000) -> list[float]:
    """Recovery times after every jump, pooled across runs."""
    times: list[float] = []
    for index, trajectory in enumerate(trajectories):
        result = run_closed_loop(trajectory, compensator, control, rng=seed + index)
        times.extend(result.recovery_times_s())
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/compensator"))
    parser.add_argument("--test-runs", type=int, default=8)
    parser.add_argument("--sweep-runs", type=int, default=4)
    parser.add_argument("--only", nargs="+", default=["main", "recovery",
                                                      "latency", "photons", "trace"],
                        choices=["main", "recovery", "latency", "photons", "trace"],
                        help="run a subset — each section writes its own CSV, so "
                             "a long evaluation can be split across sessions")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "main.pkl", "rb") as handle:
        fitted = pickle.load(handle)

    control = make_control()
    validation = make_trajectories(VALIDATION_SEEDS[:3],
                                   duration_s=TUNING_DURATION_S)
    test = make_trajectories(TEST_SEEDS[:args.test_runs])

    print("tuning the alpha-beta baseline at the main operating point...")
    alpha_beta = tune_alpha_beta(validation, control)
    print(f"  -> {alpha_beta.name}")

    compensators = [NoCompensation(), Persistence(), LinearExtrapolation(),
                    alpha_beta, fitted["ridge"], fitted["mlp"]]

    if "main" in args.only:
        print("\n=== main comparison (held-out runs) ===")
        rows = [score(test, compensator, control) for compensator in compensators]
        main = pd.DataFrame(rows)[["compensator", "mean_qber", "p95_qber", "uptime",
                                   "mean_key_rate_hz", "mean_residual_mrad",
                                   "mean_prediction_error_mrad"]]
        print(main.round(4).to_string(index=False))
        main.to_csv(args.out / "main_comparison.csv", index=False)

    if "recovery" in args.only:
      print("\n=== recovery after a cable jump ===")
      recovery_rows = []
      for compensator in compensators:
        times = np.asarray(recovery(test, compensator, control))
        finite = times[np.isfinite(times)]
        recovery_rows.append({
            "compensator": compensator.name,
            "jumps": int(len(times)),
            "recovered": int(len(finite)),
            "median_recovery_ms": float(np.median(finite) * 1e3) if len(finite) else np.nan,
            "p90_recovery_ms": float(np.percentile(finite, 90) * 1e3) if len(finite) else np.nan,
        })
      recovery_frame = pd.DataFrame(recovery_rows)
      print(recovery_frame.round(2).to_string(index=False))
      recovery_frame.to_csv(args.out / "recovery.csv", index=False)

    if "trace" in args.only:
        # One run, kept sample by sample. The aggregates say which compensator
        # is better; only the trace says *why* — where the QBER sits between
        # jumps, and what each one does in the seconds after one.
        print("\n=== per-sample trace of one held-out run ===")
        chosen = next(index for index, trajectory in enumerate(test)
                      if len(trajectory.jump_times_s) >= 2)
        trajectory = test[chosen]
        columns = {"time_s": trajectory.time_s}
        for compensator in compensators:
            result = run_closed_loop(trajectory, compensator, control,
                                     rng=7000 + chosen)
            columns[compensator.name] = result.qber
        trace = pd.DataFrame(columns)
        trace.to_csv(args.out / "trace.csv", index=False)
        pd.DataFrame({"jump_time_s": trajectory.jump_times_s}).to_csv(
            args.out / "trace_jumps.csv", index=False)
        print(f"  run {chosen}, {len(trajectory.jump_times_s)} jumps, "
              f"{len(trace)} samples")

    if "latency" not in args.only:
        print("\nskipping the latency sweep")
    else:
     print("\n=== control-latency sweep (alpha-beta re-tuned, ridge refitted) ===")
     sweep_rows = []
     training = make_trajectories(TRAIN_SEEDS[:5])
     sweep_test = test[:args.sweep_runs]
     # Every latency is a whole number of 2 ms control periods; ControlConfig
     # refuses anything else, for the reason documented there.
     for latency_ms in (4, 8, 20, 40, 80, 160):
        swept = make_control(latency_s=latency_ms * 1e-3)
        dataset = build_dataset(training, swept, FEATURE_SPEC, rng=0)
        train_mask, _ = chronological_split(dataset, 0.85, 0.02)
        ridge = fit_ridge(dataset, train_mask)
        tuned = tune_alpha_beta(validation, swept)
        for compensator in (NoCompensation(), Persistence(), tuned, ridge):
            record = score(sweep_test, compensator, swept)
            label = ("filtro alfa-beta" if compensator.name.startswith("filtro")
                     else compensator.name)
            sweep_rows.append({"latency_ms": latency_ms, "compensator": label,
                               "mean_qber": record["mean_qber"],
                               "uptime": record["uptime"],
                               "mean_key_rate_hz": record["mean_key_rate_hz"]})
        print(f"  {latency_ms:>3} ms done")
     latency = pd.DataFrame(sweep_rows)
     latency.to_csv(args.out / "latency_sweep.csv", index=False)
     print(latency.pivot(index="latency_ms", columns="compensator",
                        values="mean_qber").round(4).to_string())

    if "photons" not in args.only:
        print("skipping the photon sweep")
        print(f"\nwrote CSVs to {args.out}")
        return 0

    print("\n=== monitoring-photon sweep ===")
    photon_rows = []
    training = make_trajectories(TRAIN_SEEDS[:5])
    sweep_test = test[:args.sweep_runs]
    for photons in (25, 50, 100, 200, 400, 800):
        swept = make_control(photons_per_basis=photons)
        dataset = build_dataset(training, swept, FEATURE_SPEC, rng=0)
        train_mask, _ = chronological_split(dataset, 0.85, 0.02)
        ridge = fit_ridge(dataset, train_mask)
        tuned = tune_alpha_beta(validation, swept)
        for compensator in (tuned, ridge):
            record = score(sweep_test, compensator, swept)
            label = ("filtro alfa-beta" if compensator.name.startswith("filtro")
                     else compensator.name)
            photon_rows.append({"photons_per_basis": photons, "compensator": label,
                                "mean_qber": record["mean_qber"],
                                "uptime": record["uptime"],
                                "mean_key_rate_hz": record["mean_key_rate_hz"]})
        print(f"  {photons:>4} photons done")
    photon = pd.DataFrame(photon_rows)
    photon.to_csv(args.out / "photon_sweep.csv", index=False)
    print(photon.pivot(index="photons_per_basis", columns="compensator",
                       values="mean_qber").round(4).to_string())

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
