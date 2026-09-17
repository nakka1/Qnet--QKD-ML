#!/usr/bin/env python3
"""Compare the trained agent against every baseline, on held-out test episodes.

The seed range here is disjoint from both the training seeds and the validation
seeds used for checkpoint selection, so the numbers are not a maximum over
checkpoints.

    python scripts/evaluate_phase2.py --policy runs/agent/policy.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task import MEMORY_SIZE, make_env, make_request  # noqa: E402
from qnet_autopilot.agents import (GreedyFidelityPolicy, RandomPolicy,  # noqa: E402
                                   ShortestPathPolicy, ThresholdPolicy, evaluate)
from qnet_autopilot.agents.actor_critic import ActorCriticPolicy  # noqa: E402

#: Disjoint from training (small seeds) and validation (900_000+).
TEST_SEED = 500_000


def build_policies(weights: Path | None):
    policies = [
        RandomPolicy(0),
        ShortestPathPolicy(),
        GreedyFidelityPolicy(),
        ThresholdPolicy(name="threshold_0.90", target_fidelity=0.90),
        ThresholdPolicy(name="threshold_0.94", target_fidelity=0.94),
    ]
    if weights is not None and weights.is_file():
        agent = ActorCriticPolicy(rng=0, name="actor_critic")
        agent.load(weights)
        # The deployment temperature is chosen on the validation seeds by
        # scripts/analyse_phase2.py; falling back to 1.0 (as trained) if that
        # has not been run. Either way it is never chosen on these seeds.
        chosen = weights.parent / "temperature.json"
        agent.temperature = (json.loads(chosen.read_text())["temperature"]
                             if chosen.is_file() else 1.0)
        agent.greedy = agent.temperature <= 0
        agent.name = f"actor_critic (T={agent.temperature:g})"
        policies.append(agent)
    return policies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, default=Path("runs/agent/policy.npz"))
    parser.add_argument("--deadline-scale", type=float, default=1.0,
                        help="multiplies the request deadline; the control for "
                             "a heralding cost that changed")
    parser.add_argument("--heralding-round-trips", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(lambda: make_env(
                          shaping_weight=0.0,
                          heralding_round_trips=args.heralding_round_trips),
                      build_policies(args.policy),
                      episodes=args.episodes,
                      request_factory=lambda rng: make_request(
                          rng, args.deadline_scale),
                      memory_size=MEMORY_SIZE, seed=TEST_SEED)

    summary = result.summary()
    print(summary.round(4).to_string(index=False))

    agent_name = next((name for name in summary["policy"]
                       if name.startswith("actor_critic")), None)
    if agent_name is not None:
        print("\npaired difference vs each baseline (same episodes, same link outcomes):")
        for reference in ("threshold_0.90", "threshold_0.94", "shortest_path",
                          "greedy_fidelity", "random"):
            difference = result.paired_difference(agent_name, reference)
            print(f"  vs {reference:<16s} {difference['mean']:+.3f} "
                  f"+/- {difference['stderr']:.3f}   "
                  f"W/L/T {difference['wins']}/{difference['losses']}/{difference['ties']}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.frame.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
