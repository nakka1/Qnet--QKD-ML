"""Network topologies to run experiments on, as NetworkX graphs.

A scenario is just a graph whose edges carry ``length`` (km) and
``attenuation`` (dB/m — SeQUeNCe's unit, not dB/km) so that
:mod:`qnet_autopilot.network.topology` can turn it into a SeQUeNCe
configuration.  Keeping them as plain NetworkX graphs means routing baselines,
plotting and graph metrics all work without an adapter.

The four shapes are chosen because they stress different parts of a routing
policy:

``chain``
    The repeater textbook case.  One path, no routing decision at all — which
    makes it the right place to validate *repeater control* (when to purify,
    when to swap) with nothing else moving.

``ring``
    Two paths between any pair, of different lengths.  The smallest topology
    where a routing policy can be wrong.

``grid``
    Many near-equivalent paths.  Congestion and memory contention start to
    matter more than distance.

``metro``
    An irregular metropolitan graph with realistic span lengths and a few
    high-degree hubs.  Distances are drawn from inter-city distances in the
    São Paulo–Rio corridor, rounded, and are meant as a plausible operating
    scale — they are **not** a model of any real deployed fibre route, whose
    lengths follow duct paths rather than straight lines and typically run
    20–40% longer.

Every builder takes its distances explicitly, so a scenario can be rescaled to
ask "at what span length does this policy stop working?" — usually the most
informative sweep there is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from ..physics import constants as C

__all__ = [
    "DEFAULT_ATTENUATION_DB_PER_M",
    "chain", "ring", "grid", "metro",
    "scenario", "SCENARIOS",
    "annotate_link_budgets",
]

#: SeQUeNCe expresses channel attenuation in dB per **metre**.
DEFAULT_ATTENUATION_DB_PER_M = C.DEFAULT_ATTENUATION_DB_PER_KM / 1000.0


def _edge_attributes(length_km: float, attenuation_db_per_m: float) -> dict:
    return {"length": float(length_km), "attenuation": float(attenuation_db_per_m)}


def chain(num_nodes: int = 5, span_km: float = 25.0,
          attenuation_db_per_m: float = DEFAULT_ATTENUATION_DB_PER_M) -> nx.Graph:
    """A linear chain of repeaters: ``0 — 1 — ... — n-1``.

    Args:
        num_nodes: number of routers, at least two.
        span_km: length of every span.
        attenuation_db_per_m: fibre loss.

    Returns:
        Graph with ``length`` and ``attenuation`` on every edge.

    Raises:
        ValueError: if ``num_nodes < 2``.
    """
    if num_nodes < 2:
        raise ValueError("a chain needs at least two nodes")
    graph = nx.Graph(name=f"chain-{num_nodes}")
    for index in range(num_nodes - 1):
        graph.add_edge(index, index + 1,
                       **_edge_attributes(span_km, attenuation_db_per_m))
    return graph


def ring(num_nodes: int = 6, span_km: float = 25.0,
         attenuation_db_per_m: float = DEFAULT_ATTENUATION_DB_PER_M,
         length_jitter: float = 0.0,
         rng: np.random.Generator | int | None = None) -> nx.Graph:
    """A ring, optionally with unequal spans.

    With ``length_jitter = 0`` the two routes between opposite nodes are the
    same length and a routing policy has nothing to learn.  Give it some
    jitter — a fractional spread on each span — and the two routes differ, so
    the shortest path becomes a real (and sometimes wrong) choice.

    Args:
        num_nodes: ring size, at least three.
        span_km: mean span length.
        attenuation_db_per_m: fibre loss.
        length_jitter: fractional spread of span lengths, e.g. 0.3 for ±30%.
        rng: seed or generator for the jitter.

    Raises:
        ValueError: if ``num_nodes < 3`` or the jitter is not in [0, 1).
    """
    if num_nodes < 3:
        raise ValueError("a ring needs at least three nodes")
    if not 0.0 <= length_jitter < 1.0:
        raise ValueError("length_jitter must be in [0, 1)")

    generator = np.random.default_rng(rng)
    graph = nx.Graph(name=f"ring-{num_nodes}")
    for index in range(num_nodes):
        scale = 1.0 + length_jitter * generator.uniform(-1.0, 1.0)
        graph.add_edge(index, (index + 1) % num_nodes,
                       **_edge_attributes(span_km * scale, attenuation_db_per_m))
    return graph


def grid(rows: int = 3, columns: int = 3, span_km: float = 20.0,
         attenuation_db_per_m: float = DEFAULT_ATTENUATION_DB_PER_M) -> nx.Graph:
    """A rectangular lattice — many equivalent paths, so contention dominates.

    Nodes are relabelled to consecutive integers so downstream code never has
    to handle tuple node ids.
    """
    if rows < 1 or columns < 1:
        raise ValueError("grid needs at least one row and one column")
    lattice = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, columns))
    graph = nx.Graph(name=f"grid-{rows}x{columns}")
    for left, right in lattice.edges():
        graph.add_edge(left, right, **_edge_attributes(span_km, attenuation_db_per_m))
    return graph


#: An irregular metropolitan graph. Node labels are indicative place names in
#: the São Paulo–Rio corridor; distances are rounded straight-line figures, not
#: fibre route lengths (see the module docstring).
_METRO_EDGES: tuple[tuple[str, str, float], ...] = (
    ("sao_paulo", "campinas", 95.0),
    ("sao_paulo", "sao_jose_dos_campos", 90.0),
    ("sao_paulo", "santos", 70.0),
    ("campinas", "sao_jose_dos_campos", 130.0),
    ("sao_jose_dos_campos", "taubate", 45.0),
    ("taubate", "resende", 105.0),
    ("resende", "volta_redonda", 30.0),
    ("volta_redonda", "rio_de_janeiro", 100.0),
    ("resende", "juiz_de_fora", 130.0),
    ("juiz_de_fora", "rio_de_janeiro", 145.0),
    ("santos", "sao_jose_dos_campos", 120.0),
)


def metro(attenuation_db_per_m: float = DEFAULT_ATTENUATION_DB_PER_M,
          scale: float = 1.0) -> nx.Graph:
    """An irregular metro/regional graph with hubs and unequal spans.

    Args:
        attenuation_db_per_m: fibre loss.
        scale: multiply every distance, to sweep the whole network's size.
            ``scale=0.25`` turns this regional graph into a metro-sized one.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    graph = nx.Graph(name="metro")
    for left, right, length_km in _METRO_EDGES:
        graph.add_edge(left, right,
                       **_edge_attributes(length_km * scale, attenuation_db_per_m))
    return graph


#: Builders reachable by name, for the CLI and for sweeps.
SCENARIOS = {"chain": chain, "ring": ring, "grid": grid, "metro": metro}


def scenario(name: str, **kwargs) -> nx.Graph:
    """Build a scenario by name.

    Raises:
        KeyError: if the name is not one of :data:`SCENARIOS`.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name](**kwargs)


def annotate_link_budgets(graph: nx.Graph, scheme: str = "single_heralded",
                          **budget_kwargs) -> nx.Graph:
    """Attach a :class:`~qnet_autopilot.physics.link.LinkBudget` to every edge.

    This is what turns a topology into something a router can reason about:
    after this call each edge carries ``rate_hz``, ``expected_time_s`` and
    ``delay_s``, so shortest-path by *expected entanglement time* — a far more
    useful metric than hop count or kilometres — is one ``nx`` call away.

    Modifies ``graph`` in place and returns it.
    """
    from ..physics.link import LinkBudget

    for left, right, data in graph.edges(data=True):
        budget = LinkBudget.compute(length_km=data["length"], scheme=scheme,
                                    **budget_kwargs)
        data["rate_hz"] = budget.rate_hz
        data["expected_time_s"] = budget.expected_time_s
        data["delay_s"] = budget.delay_s
        data["arm_efficiency"] = budget.arm_efficiency
    return graph
