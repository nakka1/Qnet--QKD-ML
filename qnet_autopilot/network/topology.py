"""Turn a scenario graph into a running SeQUeNCe network.

SeQUeNCe already knows how to build a meet-in-the-middle topology from a
NetworkX graph: ``sequence.utils.nx_converter.generate_config``.  This module
does not reimplement that.  What it adds is the part that converter leaves to
the caller and that is easy to get subtly wrong:

* **hardware templates** — memory count, coherence time, detector efficiency
  and dark counts, gate and measurement fidelity — assembled from
  :class:`HardwareProfile` so that one object describes the whole network's
  hardware and can be swept;
* **formalism selection** — Bell-diagonal, so that fidelity is tracked
  analytically and the physics layer's closed forms apply;
* **unit discipline** — SeQUeNCe wants picoseconds, metres and dB/m; the
  physics layer works in seconds, kilometres and dB/km.  Every conversion in
  the package happens here, in one place, rather than being sprinkled through
  call sites.

The output is either a config ``dict`` (JSON-serialisable, useful for
reproducibility and for handing to SeQUeNCe's own tooling) or a live
``RouterNetTopo``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

from ..compat import bootstrap
from ..physics import constants as C

__all__ = ["HardwareProfile", "build_config", "build_topology", "SECOND_PS", "MILLISECOND_PS"]

#: Picoseconds per second — SeQUeNCe's clock unit.
SECOND_PS = 1_000_000_000_000
#: Picoseconds per millisecond.
MILLISECOND_PS = 1_000_000_000


@dataclass(frozen=True)
class HardwareProfile:
    """The hardware every node in the network is built from.

    One object, so an experiment sweeps hardware by replacing it rather than by
    editing a nested JSON blob.

    Attributes:
        memory_size: memories per router.  This is the buffer depth an agent
            allocates from, and the hardest constraint in practice: purification
            consumes two pairs to make one, so a shallow bank forecloses it.
        memory_coherence_time_s: memory decay constant.
        memory_efficiency: readout efficiency.
        memory_fidelity: raw fidelity of a freshly heralded pair.
        memory_frequency_hz: maximum excitation rate.
        detector_efficiency: single-photon detection efficiency at the BSM.
        dark_count_rate_hz: detector dark counts — the floor on QBER at long
            distance, where the signal is weaker than the noise.
        time_resolution_ps: detector timing jitter.
        count_rate_hz: detector maximum count rate.
        gate_fidelity: SeQUeNCe's ``gate_fid``, per node.
        measurement_fidelity: SeQUeNCe's ``meas_fid``, per node.
        classical_delay_ms: delay on the all-to-all classical control network.
            Control latency is what makes a centralised policy expensive; keep
            it honest.
    """

    memory_size: int = 25
    memory_coherence_time_s: float = C.DEFAULT_MEMORY_COHERENCE_TIME_S
    memory_efficiency: float = 0.75
    memory_fidelity: float = C.DEFAULT_RAW_FIDELITY
    memory_frequency_hz: float = 2.0e8
    detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY
    dark_count_rate_hz: float = C.DEFAULT_DARK_COUNT_RATE_HZ
    time_resolution_ps: int = int(C.DEFAULT_TIME_RESOLUTION_PS)
    count_rate_hz: float = 1.0e11
    gate_fidelity: float = C.DEFAULT_GATE_FIDELITY
    measurement_fidelity: float = C.DEFAULT_MEAS_FIDELITY
    classical_delay_ms: float = 1.0

    def to_templates(self) -> dict[str, Any]:
        """Render as the ``router_template`` / ``bsm_template`` pair SeQUeNCe wants."""
        detector = {
            "efficiency": self.detector_efficiency,
            "dark_count": self.dark_count_rate_hz,
            "time_resolution": self.time_resolution_ps,
            "count_rate": self.count_rate_hz,
        }
        return {
            "router_template": {
                "MemoryArray": {
                    "frequency": self.memory_frequency_hz,
                    "coherence_time": self.memory_coherence_time_s,
                    "efficiency": self.memory_efficiency,
                    "fidelity": self.memory_fidelity,
                }
            },
            "bsm_template": {
                "encoding_type": "single_heralded",
                # Two detectors: the two output ports of the interference
                # beamsplitter. A coincidence between them is the herald.
                "SingleHeraldedBSM": {"detectors": [dict(detector), dict(detector)]},
            },
        }

    def as_dict(self) -> dict[str, Any]:
        """Plain dict, for recording alongside a dataset."""
        return asdict(self)


def build_config(graph: nx.Graph,
                 hardware: HardwareProfile | None = None,
                 stop_time_s: float | None = None,
                 formalism: str | None = None,
                 output_file: str | None = None,
                 output_directory: str | None = None,
                 ) -> tuple[dict[str, Any], dict[Any, str]]:
    """Build a SeQUeNCe ``RouterNetTopo`` configuration from a scenario graph.

    Args:
        graph: scenario graph with ``length`` (km) and ``attenuation`` (dB/m)
            on every edge — see :mod:`qnet_autopilot.network.scenarios`.
        hardware: hardware for every node; defaults to :class:`HardwareProfile`.
        stop_time_s: simulation stop time.  ``None`` leaves SeQUeNCe's default.
        formalism: quantum state representation.  Defaults to the Bell-diagonal
            formalism, which is what the physics layer's closed forms assume;
            pass an explicit value to override.
        output_file, output_directory: write the config to disk as well.  Both
            must be given, or neither.

    Returns:
        ``(config, node_name_map)`` — the configuration dict, and the mapping
        from graph node label to SeQUeNCe router name.

    Raises:
        ValueError: if an edge is missing ``length``, or only one of the two
            output arguments is given.
    """
    bootstrap()
    from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
    from sequence.utils.nx_converter import generate_config

    if (output_file is None) != (output_directory is None):
        raise ValueError("pass both output_file and output_directory, or neither")

    missing = [(u, v) for u, v, data in graph.edges(data=True) if "length" not in data]
    if missing:
        raise ValueError(f"edges without a 'length' attribute: {missing[:3]}")

    hardware = hardware or HardwareProfile()

    config, name_map = generate_config(
        graph,
        cc_delay=hardware.classical_delay_ms,
        memory_size=hardware.memory_size,
        output_file=output_file or "",
        output_directory=output_directory or "",
        stop_time=stop_time_s,
        formalism=formalism or BELL_DIAGONAL_STATE_FORMALISM,
        node_template=hardware.to_templates(),
        gate_fid=hardware.gate_fidelity,
        meas_fid=hardware.measurement_fidelity,
    )
    return config, name_map


def build_topology(graph: nx.Graph,
                   hardware: HardwareProfile | None = None,
                   stop_time_s: float | None = None,
                   formalism: str | None = None,
                   apply_node_fidelity_workaround: bool = True):
    """Build a live ``RouterNetTopo`` from a scenario graph.

    Args:
        graph: scenario graph, as for :func:`build_config`.
        hardware: hardware for every node.
        stop_time_s: simulation stop time in seconds.
        formalism: quantum state representation; defaults to Bell-diagonal.
        apply_node_fidelity_workaround: see the note below.  Leave this on
            unless you are deliberately reproducing stock SeQUeNCe behaviour.

    Returns:
        tuple: ``(topology, node_name_map)``.  The topology's timeline is
        constructed but **not** initialised or run — the caller decides when,
        because installing protocols and applications has to happen first.

    Note:
        **Upstream bug, worked around here.**  In SeQUeNCe 1.0.0,
        ``nx_converter.generate_config`` writes ``gate_fidelity`` and
        ``measurement_fidelity`` into each node's configuration, but
        ``RouterNetTopo._add_nodes`` never reads them: it constructs
        ``QuantumRouter(name, tl, memo_size, component_templates=template)``
        and nothing else, so every router ends up with the constructor
        defaults ``gate_fid = meas_fid = 1``.  The keys are accepted and
        silently discarded.

        That is not a cosmetic problem.  Gate and measurement fidelity are
        exactly the parameters that decide whether purification helps or hurts
        and how fast fidelity degrades across a chain of swaps.  A study run on
        the stock path gets perfect local operations no matter what it
        configures, and will overstate every end-to-end fidelity it reports.

        So after constructing the topology we set ``gate_fid`` and
        ``meas_fid`` on each router directly.  Both attributes are plain
        floats read at protocol time — ``BBPSSW_BDS``, ``DEJMPS_BDS`` and
        ``EntanglementSwappingA_BDS`` all fetch them from ``self.owner`` when
        they run — so assigning them after construction is equivalent to
        having passed them in.  ``tests/test_topology.py`` asserts the values
        actually arrive, and would catch the day upstream fixes this.
    """
    bootstrap()
    from sequence.topology.router_net_topo import RouterNetTopo

    hardware = hardware or HardwareProfile()
    config, name_map = build_config(graph, hardware, stop_time_s, formalism)
    topology = RouterNetTopo(config)

    if apply_node_fidelity_workaround:
        for router in topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
            router.gate_fid = hardware.gate_fidelity
            router.meas_fid = hardware.measurement_fidelity

    return topology, name_map
