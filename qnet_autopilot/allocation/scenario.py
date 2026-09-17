"""Turning the phase-1 demand stream into demands this environment can serve.

The traffic is not regenerated here. :func:`build_demands` consumes exactly
what :func:`qnet_autopilot.datagen.generate_demand_stream` produces — the same
non-homogeneous Poisson arrivals, the same volume and deadline distributions,
the same priorities — and adds only the two things the environment needs and
the generator deliberately does not decide: the route each demand takes, and
the link indices along it.

Keeping the generator untouched matters. It was validated in phase 1, its
metadata records the parameters that produced any given stream, and a phase-4
result that quietly used different traffic would not be comparable with the
capacity traces phase 1 shipped.

Two rescalings are applied and both are visible in the signature, because a
benchmark whose offered load is far from capacity measures nothing: at 5% load
every policy serves everything, and at 500% every policy fails, and neither
says which policy is better.

* **Volume** sets the *load* — the total work asked of the plant over an
  episode, against what it can produce. The defaults put each service near, but
  under, its capacity.
* **Deadline slack** sets the *urgency* — how long a demand has, and therefore
  how many control periods a policy can wait before committing. It is scaled
  separately from the volume, because the phase-1 generator ties the two
  together (slack is a multiple of volume over a nominal rate) and this problem
  needs them independent: a plant loaded to 70% with hour-long deadlines is a
  different problem from one loaded to 70% with minute-long deadlines, and only
  the second is a scheduling problem.

Both are arguments with measured defaults, both are reported alongside every
result, and the load is swept.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ..datagen.traffic import TrafficProfile, generate_demand_stream
from .state import Demand

__all__ = ["build_demands", "shortest_paths", "offered_load",
           "BENCHMARK_PROFILE", "VOLUME_SCALES", "DEADLINE_SCALES"]


def shortest_paths(graph: nx.Graph) -> tuple[dict, dict]:
    """Shortest path by fibre length between every pair, as link indices.

    Returns:
        ``(paths, link_index)`` where ``paths[(u, v)]`` is a tuple of link
        indices and ``link_index[(u, v)]`` maps an unordered edge to its
        position in the environment's arrays.
    """
    link_index: dict[frozenset, int] = {}
    for index, (u, v) in enumerate(graph.edges()):
        link_index[frozenset((u, v))] = index

    paths: dict[tuple, tuple[int, ...]] = {}
    for source, targets in nx.all_pairs_dijkstra_path(graph, weight="length"):
        for target, node_path in targets.items():
            if source == target:
                continue
            paths[(source, target)] = tuple(
                link_index[frozenset((left, right))]
                for left, right in zip(node_path[:-1], node_path[1:]))
    return paths, link_index


#: Arrival rates that put each service near its capacity on the metro plant.
#: QKD arrives ten times more often than the phase-1 default, in sessions a
#: tenth the size: same offered load, but deadlines that fit inside an episode
#: instead of running past its end, where they could never be scored.
BENCHMARK_PROFILE = TrafficProfile(qkd_arrivals_per_s=0.5,
                                   entanglement_arrivals_per_s=0.1)

#: Volume multipliers, measured against the plant's capacity (see the module
#: docstring). Classical traffic is scaled up because the phase-1 default
#: offers a fraction of a percent of a 100 Gbit/s plant, which no scheduler
#: would ever have to think about.
#:
#: The absolute level is set so that neither end of the trade-off saturates:
#: at this load, silencing the classical channel entirely serves 73% of the
#: offered key and admitting everything serves 19%, so both a floor and a
#: ceiling are visible and every policy has somewhere to move. Half this load
#: pins the quantum-first end at 94% and hides the differences between good
#: policies; double it pushes every policy into failure. ``load_scale`` sweeps
#: it, and the report shows that sweep.
VOLUME_SCALES = {"qkd": 0.05, "classical": 190.0, "entanglement": 0.125}

#: Deadline multipliers, chosen so slack lands in the tens of control periods.
DEADLINE_SCALES = {"qkd": 0.10, "classical": 1.0, "entanglement": 1.0}


def build_demands(graph: nx.Graph,
                  duration_s: float = 1800.0,
                  seed: int = 0,
                  profile: TrafficProfile | None = None,
                  volume_scales: dict[str, float] | None = None,
                  deadline_scales: dict[str, float] | None = None,
                  load_scale: float = 1.0,
                  frame: pd.DataFrame | None = None) -> list[Demand]:
    """Generate a demand stream and resolve every demand onto a route.

    Args:
        graph: topology with ``length`` on every edge.
        duration_s: window to generate over.
        seed: passed through to the phase-1 generator.
        profile: arrival statistics; :data:`BENCHMARK_PROFILE` if omitted.
        volume_scales: per-service volume multipliers; :data:`VOLUME_SCALES`
            if omitted.
        deadline_scales: per-service slack multipliers;
            :data:`DEADLINE_SCALES` if omitted.
        load_scale: multiplies every volume, for sweeping the offered load
            without touching the relative mix between services.
        frame: use this demand table instead of generating one. For tests, and
            for replaying a stream saved by the phase-1 CLI.

    Returns:
        Demands sorted by arrival, each carrying its resolved path.

    Note:
        A demand whose source and destination are not connected is dropped
        rather than served over a route that does not exist. On a connected
        graph that never happens; on a partitioned one it would otherwise fail
        silently deep inside the environment.
    """
    if frame is None:
        dataset = generate_demand_stream(graph, duration_s=duration_s,
                                         profile=profile or BENCHMARK_PROFILE,
                                         seed=seed)
        frame = dataset.frame

    paths, _ = shortest_paths(graph)
    volume_scales = volume_scales or VOLUME_SCALES
    deadline_scales = deadline_scales or DEADLINE_SCALES

    demands: list[Demand] = []
    for row in frame.itertuples(index=False):
        route = paths.get((row.source, row.destination))
        if not route:
            continue
        volume = (float(row.volume) * volume_scales.get(row.service, 1.0)
                  * load_scale)
        slack = ((float(row.deadline_s) - float(row.arrival_s))
                 * deadline_scales.get(row.service, 1.0))
        demands.append(Demand(
            demand_id=int(row.demand_id),
            service=str(row.service),
            source=str(row.source),
            destination=str(row.destination),
            path=route,
            volume=volume,
            arrival_s=float(row.arrival_s),
            deadline_s=float(row.arrival_s) + slack,
            priority=int(row.priority),
            min_fidelity=float(row.min_fidelity),
        ))

    demands.sort(key=lambda demand: demand.arrival_s)
    return demands


def offered_load(demands: Sequence[Demand], service: str) -> float:
    """Total volume requested by one service, for reporting the load.

    A success rate means nothing without it: 100% served at a tenth of capacity
    and 100% served at capacity are different results.
    """
    return float(sum(demand.volume for demand in demands
                     if demand.service == service))
