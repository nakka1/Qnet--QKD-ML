#!/usr/bin/env python3
"""Train the phase-5 key-rate agent and save it.

    python scripts/train_keyrate.py --out runs/keyrate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task5 import make_keyrate_env, make_commissioned  # noqa: E402
from qnet_autopilot.qkd.keyrate import (KeyRatePolicy, StaticPolicy,  # noqa: E402
                                        TrainingConfig, run_episode,
                                        train_key_rate_policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/keyrate"))
    parser.add_argument("--km", type=float, nargs="+", default=[50.0],
                        help="span(s) to train on; more than one trains a "
                             "generalist, drawing a span per episode")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spans = [float(value) for value in args.km]
    generalist = len(spans) > 1
    for span in spans:
        commissioned = make_commissioned(span)
        print(f"[{args.tag}] comissionamento {span:.0f} km: "
              f"mu {commissioned.signal_intensity:.3f}  "
              f"nu {commissioned.decoy_intensity:.3f}  "
              f"clock {commissioned.clock_rate_hz / 1e9:.2f} GHz  "
              f"p_sinal {commissioned.signal_probability:.2f}")
    commissioned = make_commissioned(spans[0])

    def env_factory():
        if generalist:
            # Every episode draws a span and starts from that span's own
            # commissioning point, so the agent sees the whole range of
            # operating points rather than one of them.
            return make_keyrate_env(spans[0], rng=args.seed,
                                    lengths_km=tuple(spans))
        return make_keyrate_env(spans[0], rng=args.seed, initial=commissioned)

    config = TrainingConfig(hidden=tuple(args.hidden), iterations=args.iterations,
                            batch_episodes=args.batch_episodes, seed=args.seed)
    policy, history = train_key_rate_policy(
        env_factory, config, progress=lambda line: print(f"[{args.tag}] {line}",
                                                         flush=True))

    static = StaticPolicy()
    policy.temperature = 1.0
    seeds = range(900_000, 900_024)
    agent_rates: list[float] = []
    static_rates: list[float] = []
    for span in spans:
        env = make_keyrate_env(span, rng=args.seed, initial=make_commissioned(span))
        agent = [run_episode(env, policy, seed=seed).mean_key_rate_hz for seed in seeds]
        fixed = [run_episode(env, static, seed=seed).mean_key_rate_hz for seed in seeds]
        agent_rates.extend(agent)
        static_rates.extend(fixed)
        print(f"[{args.tag}] teste {span:.0f} km: agente {np.mean(agent) / 1e6:.3f} "
              f"Mbit/s  estático {np.mean(fixed) / 1e6:.3f} Mbit/s  "
              f"({np.mean(agent) / np.mean(fixed) - 1:+.1%})")

    gain = float(np.mean(agent_rates) / np.mean(static_rates) - 1.0)
    print(f"[{args.tag}] teste (todos os vãos): agente {np.mean(agent_rates) / 1e6:.3f} "
          f"Mbit/s  estático {np.mean(static_rates) / 1e6:.3f} Mbit/s  ({gain:+.1%})")

    policy.save(args.out / f"{args.tag}_policy.npz")
    (args.out / f"{args.tag}_history.json").write_text(json.dumps({
        "iteration": history.iteration,
        "train_key_rate_hz": history.train_key_rate_hz,
        "entropy": history.entropy,
        "grad_norm": history.grad_norm,
        "eval_iteration": history.eval_iteration,
        "eval_key_rate_hz": history.eval_key_rate_hz,
        "best_validation": history.best_validation,
        "length_km": spans,
        "commissioned": {"signal_intensity": commissioned.signal_intensity,
                         "decoy_intensity": commissioned.decoy_intensity,
                         "clock_rate_hz": commissioned.clock_rate_hz,
                         "signal_probability": commissioned.signal_probability},
        "test_agent_hz": float(np.mean(agent_rates)),
        "test_static_hz": float(np.mean(static_rates)),
    }, indent=2))
    print(f"[{args.tag}] salvo em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
