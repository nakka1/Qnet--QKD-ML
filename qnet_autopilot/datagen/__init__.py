"""Data generators: polarisation drift, link-level entanglement, hybrid traffic."""

from __future__ import annotations

from . import entanglement, polarization, qkd, schema, traffic
from .entanglement import LinkExperiment, generate_link_dataset
from .polarization import (chronological_split, generate_polarization_dataset,
                           make_supervised_windows)
from .qkd import draw_event, generate_qkd_session_dataset
from .schema import Dataset, DatasetMetadata, load
from .traffic import TrafficProfile, generate_capacity_trace, generate_demand_stream

__all__ = [
    "schema", "polarization", "entanglement", "traffic", "qkd",
    "Dataset", "DatasetMetadata", "load",
    "generate_qkd_session_dataset", "draw_event",
    "generate_polarization_dataset", "make_supervised_windows", "chronological_split",
    "LinkExperiment", "generate_link_dataset",
    "TrafficProfile", "generate_capacity_trace", "generate_demand_stream",
]
