#!/usr/bin/env python3
"""Fit the phase-3 compensators and save them.

    python scripts/train_compensator.py --out runs/compensator
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

from task3 import (FEATURE_SPEC, LATENCY_S, TRAIN_SEEDS, make_control,  # noqa: E402
                   make_trajectories)
from qnet_autopilot.compensation.train import (build_dataset, chronological_split,  # noqa: E402
                                               fit_mlp, fit_ridge)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/compensator"))
    parser.add_argument("--latency-ms", type=float, default=LATENCY_S * 1e3)
    parser.add_argument("--photons", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, nargs="+", default=[192, 192])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    control = make_control(latency_s=args.latency_ms * 1e-3,
                           photons_per_basis=args.photons)
    dataset = build_dataset(make_trajectories(TRAIN_SEEDS), control, FEATURE_SPEC,
                            target="measured", rng=args.seed)
    train, validation = chronological_split(dataset, 0.75, 0.05)
    print(f"[{args.tag}] {len(dataset)} rows, {train.sum()} train / {validation.sum()} val")

    ridge = fit_ridge(dataset, train)
    mlp, history = fit_mlp(dataset, train, validation, hidden=tuple(args.hidden),
                           epochs=args.epochs, seed=args.seed)

    def validation_mse(compensator):
        features = compensator.standardiser.transform(dataset.features[validation])
        predicted = np.asarray(compensator.model.predict(features)).reshape(-1, 3)
        return float(np.mean((predicted - dataset.targets[validation]) ** 2))

    scores = {"ridge": validation_mse(ridge), "mlp": validation_mse(mlp)}
    print(f"[{args.tag}] validation MSE  ridge {scores['ridge']:.1f}  mlp {scores['mlp']:.1f}"
          f"  ({(1 - scores['mlp'] / scores['ridge']):+.1%} vs ridge)")

    with open(args.out / f"{args.tag}.pkl", "wb") as handle:
        pickle.dump({"ridge": ridge, "mlp": mlp}, handle)
    (args.out / f"{args.tag}_history.json").write_text(json.dumps(
        {"history": history, "validation_mse": scores,
         "latency_ms": args.latency_ms, "photons": args.photons,
         "hidden": args.hidden, "epochs": args.epochs}, indent=2))
    print(f"[{args.tag}] saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
