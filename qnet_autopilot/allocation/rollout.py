"""Running a policy over episodes, and scoring it without picking a winner.

There is no single number here, on purpose
------------------------------------------
This problem has two objectives that trade against each other: secret key
delivered on time, and classical throughput. Collapsing them into one weighted
score would make every result a statement about the weight, and the weight is
not a physical quantity — it is a commercial decision that differs between a
bank and a research network.

So :func:`evaluate` reports both, and the report draws the frontier. A policy is
better than another only if it beats it on one axis without losing on the
other; where two policies sit at different points of the same frontier, the
honest description is that they are different operating points, and that is
said rather than hidden behind an aggregate.

Pairing
-------
Every policy sees the same seeds, which means the same fibre realisation, the
same arrivals and the same measurement noise. Differences are therefore paired,
and :meth:`EvaluationResult.paired_difference` reports them that way — the
episode-to-episode variance of this problem is large enough that unpaired
comparisons at this sample size would say nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .env import HybridAllocationEnv
from .state import SERVICES, Demand

__all__ = ["EpisodeResult", "run_episode", "evaluate", "EvaluationResult"]


@dataclass
class EpisodeResult:
    """What one episode produced.

    Attributes:
        policy: display name.
        seed: episode seed.
        served: per-service fraction of *volume* delivered before the deadline.
            Volume rather than count, because a demand-count metric lets a
            policy look good by serving many small requests and dropping the
            large ones.
        completed: per-service fraction of demands fully served in time.
        key_bits, classical_bits: totals produced across all links.
        qkd_delivered_bits, classical_delivered_bits, pairs_delivered: totals
            handed to demands.
        mean_admission: average classical admission fraction applied.
        mean_qber: average true QBER across links and steps.
        dead_link_fraction: share of link-steps whose QBER sat above the BB84
            threshold, producing no key at all.
        admission: the full ``(steps, links)`` admission trace, kept for the
            report's behavioural figures.
        qber: the matching true-QBER trace.
    """

    policy: str
    seed: int
    served: dict[str, float]
    completed: dict[str, float]
    key_bits: float
    classical_bits: float
    qkd_delivered_bits: float
    classical_delivered_bits: float
    pairs_delivered: float
    mean_admission: float
    mean_qber: float
    dead_link_fraction: float
    admission: np.ndarray
    qber: np.ndarray

    def as_row(self) -> dict:
        row = {"policy": self.policy, "seed": self.seed,
               "key_bits": self.key_bits, "classical_bits": self.classical_bits,
               "qkd_delivered_bits": self.qkd_delivered_bits,
               "classical_delivered_bits": self.classical_delivered_bits,
               "pairs_delivered": self.pairs_delivered,
               "mean_admission": self.mean_admission,
               "mean_qber": self.mean_qber,
               "dead_link_fraction": self.dead_link_fraction}
        for service in SERVICES:
            row[f"served_{service}"] = self.served.get(service, float("nan"))
            row[f"completed_{service}"] = self.completed.get(service, float("nan"))
        return row


def run_episode(env: HybridAllocationEnv, policy, seed: int = 0,
                keep_traces: bool = True) -> EpisodeResult:
    """Run one policy for one episode and score it.

    Args:
        env: the allocation environment.
        policy: anything satisfying :class:`~.policies.AllocationPolicy`.
        seed: episode seed — the fibre, the arrivals and the measurement noise.
        keep_traces: retain the per-step admission and QBER arrays. Off for
            large sweeps, where they dominate memory.

    Returns:
        EpisodeResult.
    """
    observation = env.reset(seed=seed)
    policy.reset(env)

    admissions: list[np.ndarray] = []
    qbers: list[np.ndarray] = []
    key_bits = classical_bits = 0.0
    qkd_delivered = classical_delivered = pairs_delivered = 0.0
    dead_steps = 0
    total_link_steps = 0

    step = 0
    done = False
    while not done:
        action = policy.act(observation, step)
        result = env.step(action)

        admissions.append(np.asarray(action, dtype=float).copy())
        qbers.append(env.state.qber.copy())
        key_bits += float(result.key_bits.sum())
        classical_bits += float(result.classical_bits.sum())
        qkd_delivered += result.qkd_delivered_bits
        classical_delivered += result.classical_delivered_bits
        pairs_delivered += result.pairs_delivered
        dead_steps += int(np.sum(env.state.qber >= env.config.qber_threshold))
        total_link_steps += env.num_links

        observation = result.observation
        done = result.done
        step += 1

    served, completed = _service_metrics(env)
    admission_trace = np.asarray(admissions)
    qber_trace = np.asarray(qbers)

    return EpisodeResult(
        policy=getattr(policy, "name", type(policy).__name__),
        seed=seed,
        served=served,
        completed=completed,
        key_bits=key_bits,
        classical_bits=classical_bits,
        qkd_delivered_bits=qkd_delivered,
        classical_delivered_bits=classical_delivered,
        pairs_delivered=pairs_delivered,
        mean_admission=float(admission_trace.mean()),
        mean_qber=float(qber_trace.mean()),
        dead_link_fraction=dead_steps / max(total_link_steps, 1),
        admission=admission_trace if keep_traces else np.empty((0, 0)),
        qber=qber_trace if keep_traces else np.empty((0, 0)),
    )


def _service_metrics(env: HybridAllocationEnv) -> tuple[dict, dict]:
    """Fraction of volume served, and fraction of demands completed, per service.

    Only demands that arrived during the episode are counted. A demand whose
    deadline falls after the episode ends is excluded from both: it was never
    given its full chance, and counting it as a failure would penalise whichever
    policy happened to face more of them.
    """
    served: dict[str, float] = {}
    completed: dict[str, float] = {}

    for service in SERVICES:
        scored = [demand for demand in env.demands
                  if demand.service == service
                  and demand.arrival_s <= env.state.time_s
                  and demand.deadline_s <= env.state.time_s]
        if not scored:
            served[service] = float("nan")
            completed[service] = float("nan")
            continue

        offered = sum(demand.volume for demand in scored)
        delivered = sum(min(env.state.progress[demand.demand_id].delivered,
                            demand.volume) for demand in scored)
        finished = sum(1 for demand in scored
                       if env.state.progress[demand.demand_id].done)
        served[service] = delivered / offered if offered > 0 else float("nan")
        completed[service] = finished / len(scored)

    return served, completed


@dataclass
class EvaluationResult:
    """Episode rows for several policies on shared seeds."""

    frame: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Mean and standard error per policy, over episodes."""
        grouped = self.frame.groupby("policy", sort=False)
        summary = grouped.mean(numeric_only=True).drop(columns=["seed"])
        counts = grouped.size()
        for column in ("served_qkd", "served_classical", "served_entanglement"):
            if column in self.frame:
                summary[f"{column}_stderr"] = (
                    grouped[column].std(ddof=1) / np.sqrt(counts))
        summary["episodes"] = counts
        return summary.reset_index()

    def paired_difference(self, policy: str, reference: str,
                          column: str = "served_qkd") -> dict:
        """Per-seed difference between two policies on one metric.

        Returns:
            Mean, standard error, and the win/loss/tie record over seeds.

        Raises:
            KeyError: if either policy is missing from the frame.
        """
        left = self.frame[self.frame["policy"] == policy].set_index("seed")[column]
        right = self.frame[self.frame["policy"] == reference].set_index("seed")[column]
        if left.empty or right.empty:
            raise KeyError(f"no rows for {policy!r} or {reference!r}")

        shared = left.index.intersection(right.index)
        difference = (left.loc[shared] - right.loc[shared]).to_numpy()
        return {
            "policy": policy,
            "reference": reference,
            "metric": column,
            "mean": float(np.mean(difference)),
            "stderr": float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1 else 0.0,
            "wins": int(np.sum(difference > 0)),
            "losses": int(np.sum(difference < 0)),
            "ties": int(np.sum(difference == 0)),
        }


def evaluate(env: HybridAllocationEnv, policies: Sequence, seeds: Sequence[int],
             keep_traces: bool = False, progress=None) -> EvaluationResult:
    """Run every policy on every seed.

    Args:
        env: the environment, reused across runs — ``reset`` restores it fully.
        policies: the policies to compare.
        seeds: episode seeds, shared by all policies so the comparison is paired.
        keep_traces: retain per-step arrays on every episode.
        progress: called with ``(policy_name, seed)`` after each episode.

    Returns:
        EvaluationResult.
    """
    rows = []
    for policy in policies:
        for seed in seeds:
            result = run_episode(env, policy, seed=int(seed), keep_traces=keep_traces)
            rows.append(result.as_row())
            if progress is not None:
                progress(result.policy, int(seed))
    return EvaluationResult(frame=pd.DataFrame(rows))
