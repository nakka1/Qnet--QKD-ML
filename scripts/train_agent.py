#!/usr/bin/env python3
"""Train the phase-2 agent, with resume so a long run survives a short shell.

    python scripts/train_agent.py --iterations 200 --out runs/agent
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

from task import MEMORY_SIZE, make_env, make_request  # noqa: E402
from qnet_autopilot.agents.actor_critic import (ActorCriticPolicy, TrainingConfig,  # noqa: E402
                                                train)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-end", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--shaping", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/agent"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deadline-scale", type=float, default=1.0,
                        help="multiplies the request deadline; the control for "
                             "a heralding cost that changed")
    parser.add_argument("--heralding-round-trips", type=float, default=1.0,
                        help="span traversals charged per heralding attempt; "
                             "1 is what phase 2 published, 4.1 is what "
                             "SeQUeNCe's protocol was measured to cost")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.out / "policy.npz"
    history_path = args.out / "history.json"

    policy = None
    if args.resume and weights.is_file():
        policy = ActorCriticPolicy(rng=args.seed)
        policy.load(weights)
        print(f"resumed from {weights}")

    config = TrainingConfig(
        iterations=args.iterations, batch_episodes=args.batch,
        learning_rate=args.lr, learning_rate_end=args.lr_end,
        entropy_start=args.entropy, entropy_end=args.entropy_end,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        seed=args.seed)

    policy, history = train(lambda: make_env(
                                shaping_weight=args.shaping,
                                heralding_round_trips=args.heralding_round_trips),
                            lambda rng: make_request(rng, args.deadline_scale),
                            config, memory_size=MEMORY_SIZE,
                            policy=policy, progress=print)

    policy.save(weights)
    previous = json.loads(history_path.read_text()) if (args.resume and history_path.is_file()) else None
    record = {
        "train": history.to_frame().to_dict(orient="list"),
        "eval": history.eval_frame().to_dict(orient="list"),
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
    }
    if previous is not None:
        offset = max(previous["train"]["iteration"]) + 1
        for key in record["train"]:
            record["train"][key] = previous["train"][key] + (
                [v + offset for v in record["train"][key]] if key == "iteration"
                else record["train"][key])
        for key in record["eval"]:
            record["eval"][key] = previous["eval"][key] + (
                [v + offset for v in record["eval"][key]] if key == "iteration"
                else record["eval"][key])
    history_path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved {weights} and {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
