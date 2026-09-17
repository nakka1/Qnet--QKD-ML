#!/usr/bin/env python3
"""Fit the drift forecaster the predictive allocator reads.

Only the forecaster is fitted. The allocation control law is fixed arithmetic
(see :class:`~qnet_autopilot.allocation.policies.PredictiveAllocator`), so
swapping the forecaster changes the forecast and nothing else — which is what
makes the phase-4 comparison a measurement of what prediction is worth.

    python scripts/train_allocator.py --out runs/allocator
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task4 import (FORECAST_HORIZON, FORECAST_WINDOW, TRAIN_SEEDS,  # noqa: E402
                   VALIDATION_SEEDS, make_env)
from qnet_autopilot.allocation.forecast import (PersistenceForecaster,  # noqa: E402
                                                build_windows,
                                                fit_ridge_forecaster)


def collect_traces(seeds, observed: bool = True) -> list[np.ndarray]:
    """Per-episode drift traces, as the allocator would have estimated them.

    With ``observed``, the trace carries the binomial measurement noise a real
    controller's estimate has — fitting on the noise-free truth would credit the
    model with denoising it cannot do in the field, the same mistake phase 3
    documents at length. The classical channel is silent while collecting, so
    the measurement is of the drift alone and nothing has to be subtracted.
    """
    traces = []
    for seed in seeds:
        env = make_env(seed=int(seed))
        env.reset(seed=int(seed))
        idle = np.zeros(env.num_links)
        rows = []
        done = False
        while not done:
            result = env.step(idle)
            rows.append(result.observation.measured_qber.copy() if observed
                        else env.state.qber.copy())
            done = result.done
        traces.append(np.asarray(rows))
    return traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/allocator"))
    parser.add_argument("--alpha", type=float, default=1e-6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"collecting {len(TRAIN_SEEDS)} training episodes...")
    train = collect_traces(TRAIN_SEEDS)
    print(f"collecting {len(VALIDATION_SEEDS)} validation episodes...")
    validation = collect_traces(VALIDATION_SEEDS)

    features, targets = build_windows(train, FORECAST_WINDOW, FORECAST_HORIZON)
    print(f"  {len(features)} windows of {FORECAST_WINDOW} -> {FORECAST_HORIZON}")

    forecaster = fit_ridge_forecaster(features, targets, FORECAST_WINDOW,
                                      FORECAST_HORIZON, alpha=args.alpha)

    # Score against persistence, per horizon step. Persistence is the null the
    # whole phase rests on: if the fitted model does not beat it here, nothing
    # downstream can be attributed to the forecast.
    v_features, v_targets = build_windows(validation, FORECAST_WINDOW,
                                          FORECAST_HORIZON)
    design = np.hstack([v_features, np.ones((len(v_features), 1))])
    predicted = np.maximum(design @ forecaster.coefficients, 0.0)
    persistence = np.repeat(v_features[:, -1:], FORECAST_HORIZON, axis=1)

    ridge_mse = np.mean((predicted - v_targets) ** 2, axis=0)
    persistence_mse = np.mean((persistence - v_targets) ** 2, axis=0)

    print("\nhorizon  ridge MSE   persistência MSE   redução")
    for step in range(FORECAST_HORIZON):
        gain = 1.0 - ridge_mse[step] / persistence_mse[step]
        print(f"{step + 1:>6}  {ridge_mse[step]:.3e}   {persistence_mse[step]:.3e}"
              f"   {gain:+.1%}")

    with open(args.out / "forecaster.pkl", "wb") as handle:
        pickle.dump(forecaster, handle)
    (args.out / "forecast_scores.json").write_text(json.dumps({
        "horizon": FORECAST_HORIZON,
        "window": FORECAST_WINDOW,
        "alpha": args.alpha,
        "train_windows": int(len(features)),
        "ridge_mse": ridge_mse.tolist(),
        "persistence_mse": persistence_mse.tolist(),
    }, indent=2))
    print(f"\nwrote {args.out}/forecaster.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
