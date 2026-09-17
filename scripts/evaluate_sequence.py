#!/usr/bin/env python3
"""The phase-2 agent, run inside SeQUeNCe's event loop and against the twin.

The last validation gap. ``tests/test_env_matches_sequence.py`` showed the
analytic environment's state arithmetic is SeQUeNCe's along whole trajectories;
it could say nothing about the parts of the simulator that are not arithmetic —
photons that arrive or do not, detectors that click, protocols that negotiate
over channels with delay, memories that are held while they do.

This runs the trained policy in both simulators on the same requests and
compares what comes out: how often it delivers, at what fidelity, how long it
takes, and how many heralding attempts it burns. Where they differ, the
difference is the discrete-event simulator's, because the physics underneath is
the same code and is already validated against it.

    python scripts/evaluate_sequence.py --out runs/sequence

Note:
    Generation is single-heralded, which is the scheme phase 1 validated and the
    only one the Bell-diagonal formalism can represent — it has no way to write
    down a single photon, so Barrett-Kok's two-photon interference cannot run
    under it. That is a constraint of the formalism, not a choice made here, and
    it is the reason this comparison is possible at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task import MEMORY_SIZE, make_config, make_env, make_request  # noqa: E402
from qnet_autopilot.agents.actor_critic import ActorCriticPolicy  # noqa: E402
from qnet_autopilot.agents.baselines import (GreedyFidelityPolicy,  # noqa: E402
                                             ShortestPathPolicy,
                                             ThresholdPolicy)
from qnet_autopilot.agents.rollout import run_episode  # noqa: E402
from qnet_autopilot.env.sequence_runner import run_sequence_episode  # noqa: E402
from qnet_autopilot.physics.constants import (  # noqa: E402
    MEASURED_HERALDING_ROUND_TRIPS)


def build_policies(weights: Path | None) -> list:
    """The learned policy and the two baselines worth the simulator time."""
    policies = [ShortestPathPolicy(), GreedyFidelityPolicy(),
                ThresholdPolicy(name="threshold_0.94", target_fidelity=0.94)]
    if weights is not None and weights.is_file():
        agent = ActorCriticPolicy(rng=0, name="actor_critic")
        agent.load(weights)
        chosen = weights.parent / "temperature.json"
        agent.temperature = (json.loads(chosen.read_text())["temperature"]
                             if chosen.is_file() else 1.0)
        agent.greedy = agent.temperature <= 0
        agent.name = f"actor_critic (T={agent.temperature:g})"
        policies.insert(0, agent)
    return policies


def compare(policies, seeds, memory_size: int,
            heralding_round_trips: float = 1.0) -> pd.DataFrame:
    """Every policy through every seed, in both simulators.

    The request is drawn from the seed, so a given seed is the same *task* in
    both simulators even though the link outcomes are drawn by different
    generators — which is the honest pairing available here. Pairing the
    heralding luck as well would require the two simulators to share a random
    stream, and they cannot: one samples a geometric number of rounds, the
    other samples a detector click per round.
    """
    from dataclasses import replace

    from qnet_autopilot.env.network_env import QuantumNetworkEnv

    config = make_config()
    analytic_config = replace(config,
                              heralding_round_trips=heralding_round_trips)
    analytic = QuantumNetworkEnv(analytic_config, rng=0)
    rows = []

    for policy in policies:
        for seed in seeds:
            request = make_request(np.random.default_rng(10_000 + seed))

            episode = run_episode(analytic, policy, request=request,
                                  memory_size=memory_size, seed=int(seed),
                                  collect=False)
            rows.append({
                "simulator": f"twin (custo x{heralding_round_trips:g})",
                "policy": policy.name,
                "seed": int(seed), "succeeded": bool(episode.succeeded),
                "delivered": episode.delivered,
                "mean_fidelity": episode.mean_fidelity,
                "elapsed_s": episode.elapsed_s, "steps": episode.steps,
                "attempts": episode.attempts,
                "purifications": episode.purifications, "swaps": episode.swaps,
                "discarded": episode.discarded, "legality_disagreements": 0,
            })

            record = run_sequence_episode(config, policy, request,
                                          seed=int(seed),
                                          memory_size=memory_size)
            record["simulator"] = "SeQUeNCe"
            rows.append(record)
        print(f"  {policy.name}: done")

    return pd.DataFrame(rows)


#: Spans the heralding round is timed over. Wide enough that a constant factor
#: and an additive handshake look different: an additive one would shrink as a
#: fraction of the round as the span grows.
ROUND_COST_SPANS_KM = (10.0, 20.0, 40.0, 80.0)


def study_round_cost(successes: int = 80) -> pd.DataFrame:
    """Time one heralding attempt in each simulator, across span lengths.

    The analytic environment charges one traversal of the span per attempt —
    the photons fly to the midpoint and the herald comes back. SeQUeNCe's
    protocol negotiates first, so an attempt is several messages. This measures
    the ratio, and whether it is a constant or a fixed overhead.
    """
    import networkx as nx

    from qnet_autopilot.env.network_env import EnvironmentConfig
    from qnet_autopilot.env.sequence_runner import SequenceNetwork

    base = make_config()
    rows = []
    for span_km in ROUND_COST_SPANS_KM:
        graph = nx.Graph()
        graph.add_edge("0", "1", length=span_km)
        config = EnvironmentConfig(
            graph=graph, memory=base.memory, gate_fid=base.gate_fid,
            meas_fid=base.meas_fid,
            coupling_efficiency=base.coupling_efficiency,
            detector_efficiency=base.detector_efficiency,
            generate_budget_s=8.0)
        network = SequenceNetwork(config, memory_size=4, seed=0)

        # Every round is counted, successful budget or not. Estimating the
        # probability from the mean rounds-to-success over successful budgets
        # only would drop the long runs that ran out of budget, which is
        # exactly the tail that makes the estimate honest — at 80 km it biases
        # the estimate up by tens of percent.
        total_rounds = 0
        total_successes = 0
        durations, per_success_rounds = [], []
        while total_successes < successes and total_rounds < 400_000:
            outcome = network.generate("0", "1")
            total_rounds += outcome.info.get("rounds", 0)
            if outcome.outcome != "generate_succeeded":
                continue
            total_successes += 1
            per_success_rounds.append(outcome.info["rounds"])
            durations.append(outcome.duration_s)
            network.discard(next(iter(network.pairs)))

        if not per_success_rounds:
            continue
        per_round = float(np.mean(np.asarray(durations)
                                  / np.asarray(per_success_rounds)))
        analytic = config.link_round_trip_s[config.key("0", "1")]
        measured = total_successes / total_rounds
        rows.append({
            "span_km": span_km,
            "sequence_ms_per_round": per_round * 1e3,
            "analytic_ms_per_round": analytic * 1e3,
            "ratio": per_round / analytic,
            "measured_probability": measured,
            "probability_stderr": measured / np.sqrt(total_successes),
            "analytic_probability": config.link_probability[config.key("0", "1")],
            "successes": total_successes,
            "rounds": total_rounds,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/sequence"))
    parser.add_argument("--policy", type=Path,
                        default=Path("runs/agent/policy.npz"))
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--memory-size", type=int, default=MEMORY_SIZE)
    parser.add_argument("--heralding-round-trips", type=float, nargs="+",
                        default=[1.0, MEASURED_HERALDING_ROUND_TRIPS],
                        help="what the twin charges per heralding attempt; the "
                             "published value and the measured one")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=== what one heralding attempt costs in each simulator ===")
    cost = study_round_cost()
    cost.to_csv(args.out / "round_cost.csv", index=False)
    print(cost.round(4).to_string(index=False))

    policies = build_policies(args.policy)
    print(f"=== {len(policies)} policies x {args.seeds} seeds, two simulators ===")
    frames = []
    for index, overhead in enumerate(args.heralding_round_trips):
        print(f"-- twin charging {overhead:g} span traversals per attempt")
        frame = compare(policies, range(args.seeds), args.memory_size,
                        heralding_round_trips=overhead)
        # SeQUeNCe does not change between the two, so it is run once and the
        # duplicate rows dropped rather than paying for it twice.
        if index > 0:
            frame = frame[frame["simulator"] != "SeQUeNCe"]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.out / "comparison.csv", index=False)

    summary = frame.groupby(["policy", "simulator"]).agg(
        success=("succeeded", "mean"),
        fidelity=("mean_fidelity", "mean"),
        elapsed_ms=("elapsed_s", lambda values: 1e3 * values.mean()),
        attempts=("attempts", "mean"),
        swaps=("swaps", "mean"),
        purifications=("purifications", "mean"),
        disagreements=("legality_disagreements", "sum")).reset_index()
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    total = int(frame["legality_disagreements"].sum())
    print(f"\naction sets differed on {total} steps "
          f"({'the two simulators agree on what is legal' if total == 0 else 'INVESTIGATE'})")
    print(f"wrote CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
