#!/usr/bin/env python3
"""Score every allocation policy on held-out episodes, and trace their frontiers.

There is no single winner to crown here, because the problem has two objectives
that trade against each other. What the evaluation produces instead is, for each
policy family, the frontier its tuning knob traces through (key served,
classical served) — and then compares families at matched service levels, which
is the only comparison that means anything when policies sit at different
operating points.

    python scripts/evaluate_phase4.py --out runs/allocator
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

from task4 import (CLASSICAL_POWER_W, FORECAST_HORIZON, TEST_SEEDS,  # noqa: E402
                   VALIDATION_SEEDS, make_env)
from qnet_autopilot.allocation import (ClassicalFirst, PersistenceForecaster,  # noqa: E402
                                       PredictiveAllocator, QuantumFirst,
                                       ReactiveThreshold, StaticAdmission,
                                       evaluate, run_episode)
from qnet_autopilot.allocation.forecast import OracleForecaster  # noqa: E402

#: Coverage targets that trace the predictive policy's frontier. The range has
#: to span far enough on both sides to overlap the baselines' frontiers —
#: comparing two policies that never reach the same service level says nothing
#: about which is better, only that they were tuned differently.
COVERAGE_GRID = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

#: Dead bands the reactive baseline is tuned over.
REACTIVE_GRID = tuple((low, high) for low, high in
                      itertools.product((0.02, 0.035, 0.05, 0.07), (0.04, 0.06, 0.08, 0.10))
                      if high > low)


def mean_point(env, policy, seeds) -> dict:
    """Average one policy's (key, classical) service over seeds."""
    rows = [run_episode(env, policy, seed=int(seed)).as_row() for seed in seeds]
    frame = pd.DataFrame(rows)
    return {
        "policy": frame["policy"].iloc[0],
        "served_qkd": float(frame["served_qkd"].mean()),
        "served_classical": float(frame["served_classical"].mean()),
        "served_entanglement": float(frame["served_entanglement"].mean()),
        "qkd_stderr": float(frame["served_qkd"].std(ddof=1) / np.sqrt(len(frame))),
        "classical_stderr": float(
            frame["served_classical"].std(ddof=1) / np.sqrt(len(frame))),
        "mean_admission": float(frame["mean_admission"].mean()),
        "mean_qber": float(frame["mean_qber"].mean()),
        "dead_link_fraction": float(frame["dead_link_fraction"].mean()),
    }


def tune_reactive(env, seeds) -> ReactiveThreshold:
    """Pick the dead band that maximises key service on validation episodes.

    Tuned on key rather than on a weighted score, because a weighted score would
    need a weight and the whole report refuses to pick one. Ties on key are
    broken by classical throughput, which is the natural secondary objective.
    """
    best, best_key = None, (-np.inf, -np.inf)
    for low, high in REACTIVE_GRID:
        point = mean_point(env, ReactiveThreshold(low=low, high=high), seeds)
        score = (round(point["served_qkd"], 3), point["served_classical"])
        if score > best_key:
            best, best_key = (low, high), score
    return ReactiveThreshold(low=best[0], high=best[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/allocator"))
    parser.add_argument("--test-seeds", type=int, default=8)
    parser.add_argument("--sweep-seeds", type=int, default=4)
    parser.add_argument("--only", nargs="+",
                        default=["frontier", "ablation", "latency", "power",
                                 "decisions", "behaviour"],
                        choices=["frontier", "ablation", "latency", "power",
                                 "decisions", "behaviour"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "forecaster.pkl", "rb") as handle:
        ridge = pickle.load(handle)

    test = TEST_SEEDS[:args.test_seeds]
    sweep = TEST_SEEDS[:args.sweep_seeds]
    env = make_env(seed=0)

    if "frontier" in args.only:
        print("=== tuning the reactive baseline on validation episodes ===")
        reactive = tune_reactive(env, VALIDATION_SEEDS[:3])
        print(f"  -> {reactive.name}")

        print("\n=== frontiers on held-out episodes ===")
        rows = []
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            point = mean_point(env, StaticAdmission(level), test)
            rows.append({"family": "admissão estática", "knob": level, **point})
        for low, high in ((0.01, 0.02), (0.02, 0.04), (0.035, 0.06),
                          (0.05, 0.08), (0.07, 0.10), (0.09, 0.14)):
            point = mean_point(env, ReactiveThreshold(low=low, high=high), test)
            rows.append({"family": "limiar reativo", "knob": high, **point})
        for coverage in COVERAGE_GRID:
            point = mean_point(env, PredictiveAllocator(
                forecaster=PersistenceForecaster(horizon=FORECAST_HORIZON),
                coverage_target=coverage), test)
            rows.append({"family": "preditivo", "knob": coverage, **point})
        for policy in (ClassicalFirst(), QuantumFirst()):
            point = mean_point(env, policy, test)
            rows.append({"family": point["policy"], "knob": np.nan, **point})

        frontier = pd.DataFrame(rows)
        frontier.to_csv(args.out / "frontier.csv", index=False)
        print(frontier[["family", "knob", "served_qkd", "served_classical",
                        "mean_admission"]].round(4).to_string(index=False))

    if "ablation" in args.only:
        print("\n=== what the forecast is worth (same control law, paired) ===")
        forecasters = [PersistenceForecaster(horizon=FORECAST_HORIZON), ridge,
                       OracleForecaster(horizon=FORECAST_HORIZON)]
        policies, rows = [], []
        for coverage in (0.25, 0.5, 1.0, 2.0):
            for forecaster in forecasters:
                policy = PredictiveAllocator(forecaster=forecaster,
                                             coverage_target=coverage)
                policy.name = f"{forecaster.name} · cob {coverage:g}"
                policies.append(policy)
        result = evaluate(env, policies, test)
        result.frame.to_csv(args.out / "ablation_episodes.csv", index=False)

        for coverage in (0.25, 0.5, 1.0, 2.0):
            base = f"persistência · cob {coverage:g}"
            for other in (f"ridge · cob {coverage:g}", f"oráculo · cob {coverage:g}"):
                for metric in ("served_qkd", "served_classical"):
                    difference = result.paired_difference(other, base, metric)
                    rows.append({"coverage": coverage, **difference})
        ablation = pd.DataFrame(rows)
        ablation.to_csv(args.out / "ablation.csv", index=False)
        print(ablation[["coverage", "policy", "metric", "mean", "stderr",
                        "wins", "losses"]].round(5).to_string(index=False))

    if "latency" in args.only:
        print("\n=== reconfiguration-latency sweep ===")
        rows = []
        for latency in (0.0, 15.0, 30.0, 60.0, 120.0):
            swept = make_env(seed=0, reconfiguration_latency_s=latency)
            for forecaster in (PersistenceForecaster(horizon=FORECAST_HORIZON),
                               ridge, OracleForecaster(horizon=FORECAST_HORIZON)):
                point = mean_point(swept, PredictiveAllocator(
                    forecaster=forecaster, coverage_target=1.0), sweep)
                rows.append({"latency_s": latency, "forecaster": forecaster.name,
                             **point})
            print(f"  {latency:>5.0f} s done")
        latency_frame = pd.DataFrame(rows)
        latency_frame.to_csv(args.out / "latency_sweep.csv", index=False)
        print(latency_frame.pivot(index="latency_s", columns="forecaster",
                                  values="served_qkd").round(4).to_string())

    if "power" in args.only:
        print("\n=== classical launch-power sweep ===")
        rows = []
        for power_mw in (0.03, 0.1, 0.3, 1.0):
            swept = make_env(seed=0, classical_power_full_w=power_mw * 1e-3)
            for policy in (ClassicalFirst(), QuantumFirst(),
                           StaticAdmission(0.5),
                           PredictiveAllocator(
                               forecaster=PersistenceForecaster(
                                   horizon=FORECAST_HORIZON),
                               coverage_target=1.0)):
                point = mean_point(swept, policy, sweep)
                rows.append({"power_mw": power_mw, **point})
            print(f"  {power_mw:>5.2f} mW done")
        power_frame = pd.DataFrame(rows)
        power_frame.to_csv(args.out / "power_sweep.csv", index=False)
        print(power_frame.pivot(index="power_mw", columns="policy",
                                values="served_qkd").round(4).to_string())

    if "decisions" in args.only:
        print("\n=== how often the forecast changes the decision ===")
        rows = []
        for latency in (0.0, 30.0, 60.0, 120.0):
            swept = make_env(seed=0, reconfiguration_latency_s=latency)
            for seed in sweep:
                traces = {}
                for forecaster in (PersistenceForecaster(horizon=FORECAST_HORIZON),
                                   ridge, OracleForecaster(horizon=FORECAST_HORIZON)):
                    result = run_episode(swept, PredictiveAllocator(
                        forecaster=forecaster, coverage_target=1.0),
                        seed=int(seed), keep_traces=True)
                    traces[forecaster.name] = (result.admission, result.served["qkd"],
                                               result.served["classical"])
                base = traces["persistência"]
                for name in ("ridge", "oráculo"):
                    other = traces[name]
                    rows.append({
                        "latency_s": latency, "seed": int(seed), "forecaster": name,
                        "decisions_changed": float(np.mean(base[0] != other[0])),
                        "delta_qkd": other[1] - base[1],
                        "delta_classical": other[2] - base[2],
                    })
        decisions = pd.DataFrame(rows)
        decisions.to_csv(args.out / "decisions.csv", index=False)
        print(decisions.groupby(["latency_s", "forecaster"])[
            ["decisions_changed", "delta_qkd", "delta_classical"]
        ].mean().round(5).to_string())

    if "behaviour" in args.only:
        # What each policy actually does, per link. The aggregate scores say
        # which policy is better; only this says what it learned about the
        # plant, and it is where the span-length story is visible.
        print("\n=== admission by link, per policy ===")
        rows = []
        policies = [StaticAdmission(0.5), ReactiveThreshold(low=0.02, high=0.04),
                    PredictiveAllocator(
                        forecaster=PersistenceForecaster(horizon=FORECAST_HORIZON),
                        coverage_target=1.0)]
        for policy in policies:
            family = ("preditivo" if policy.name.startswith("preditivo")
                      else "limiar reativo" if policy.name.startswith("limiar")
                      else "admissão estática")
            applied, owing = [], []
            for seed in sweep:
                # Stepped by hand rather than through run_episode because the
                # interesting quantity is admission *conditioned on the link
                # owing key*. Unconditional admission mixes that decision with
                # "this link has nothing queued, so open it up", and the second
                # has nothing to do with coexistence.
                observation = env.reset(seed=int(seed))
                policy.reset(env)
                step, done = 0, False
                while not done:
                    action = policy.act(observation, step)
                    result = env.step(action)
                    applied.append(np.asarray(result.observation.admission))
                    owing.append(observation.backlog_bits > 0)
                    observation = result.observation
                    done, step = result.done, step + 1

            applied = np.asarray(applied)
            owing = np.asarray(owing)
            for index, link in enumerate(env.links):
                mask = owing[:, index]
                rows.append({
                    "policy": family, "link": link.name,
                    "length_km": link.length_km,
                    "mean_admission": float(applied[:, index].mean()),
                    "mean_admission_when_owing": float(applied[mask, index].mean())
                    if mask.any() else float("nan"),
                    "owing_fraction": float(mask.mean()),
                })
        behaviour = pd.DataFrame(rows)
        behaviour.to_csv(args.out / "behaviour.csv", index=False)
        print(behaviour.pivot_table(index="length_km", columns="policy",
                                    values="mean_admission_when_owing"
                                    ).round(3).to_string())

    print(f"\nwrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
