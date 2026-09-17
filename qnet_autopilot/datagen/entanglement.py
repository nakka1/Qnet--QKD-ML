"""Link-level entanglement data, measured by actually running SeQUeNCe.

This is the dataset the analytic link model in
:mod:`qnet_autopilot.physics.link` has to answer to.  Nothing here uses a
closed form: a two-node link with a Bell-state measurement in the middle is
built out of real SeQUeNCe components, entanglement generation protocols are
started, the discrete-event simulation runs, and the outcome of every attempt
is recorded — succeeded or not, at what simulated time, with which memories.

Two things come out of that:

* **Ground truth for the rate models.**  Sweeping distance and comparing the
  measured success rate against :func:`~qnet_autopilot.physics.link.attempt_success_probability`
  tells you where the closed form is usable and where it is not.  That
  comparison lives in :mod:`qnet_autopilot.validation`.
* **A link-state dataset** for anything that needs to predict how a span will
  behave — the input to a routing policy that wants "how long until this edge
  gives me a pair?" rather than "how many kilometres is this edge?".

The harness follows the pattern in SeQUeNCe's own
``tests/entanglement_management/test_generation.py``: ``FakeRouter`` /
``FakeBSMNode`` with a minimal resource manager that logs state transitions.
Using their pattern rather than inventing one means the protocol sees exactly
the environment it was written for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..compat import bootstrap
from ..physics import constants as C
from .schema import Dataset, DatasetMetadata, describe_columns

__all__ = ["LinkExperiment", "generate_link_dataset", "COLUMNS"]

COLUMNS = {
    "length_km": "total span length between the two end nodes",
    "attempt": "index of the entanglement generation attempt",
    "succeeded": "whether the attempt heralded an entangled pair",
    "herald_time_s": "simulated time at which the attempt resolved",
    "fidelity": "fidelity of the heralded pair, NaN if the attempt failed",
    "detector_efficiency": "detector efficiency used for this run",
    "attenuation_db_per_km": "fibre loss coefficient used for this run",
    "memory_efficiency": "memory readout efficiency used for this run",
    "coherence_time_s": "memory coherence time used for this run",
    "raw_fidelity": "configured raw fidelity of a fresh pair",
    "seed": "RNG seed for this run",
}


@dataclass
class LinkExperiment:
    """One two-node link with midpoint heralding, as SeQUeNCe components.

    Attributes:
        length_km: total span; each arm is half of it.
        attenuation_db_per_km: fibre loss.
        detector_efficiency: BSM detector efficiency.
        dark_count_rate_hz: BSM detector dark counts.
        memory_efficiency: memory readout efficiency.
        coherence_time_s: memory coherence time.
        raw_fidelity: fidelity of a fresh heralded pair.
        classical_delay_s: control-channel delay between the nodes and the BSM.
            Defaults to the physical propagation delay over half the span,
            which is the honest value — SeQUeNCe lets you set anything.
        num_attempts: how many independent generation attempts to run.
        formalism: quantum state representation for this run. The heralded
            generation protocol works in the ket formalism; the Bell-diagonal
            formalism has no state to report until entanglement exists, so
            running generation under it raises. This is set and restored around
            each run because SeQUeNCe's formalism is **global process state** —
            building a Bell-diagonal topology elsewhere in the same process
            silently changes it, and without this a link experiment would fail
            or, worse, behave differently depending on what ran before it.
    """

    length_km: float = 20.0
    attenuation_db_per_km: float = C.DEFAULT_ATTENUATION_DB_PER_KM
    detector_efficiency: float = C.DEFAULT_DETECTOR_EFFICIENCY
    dark_count_rate_hz: float = 0.0
    memory_efficiency: float = 1.0
    coherence_time_s: float = C.DEFAULT_MEMORY_COHERENCE_TIME_S
    raw_fidelity: float = C.DEFAULT_RAW_FIDELITY
    classical_delay_s: float | None = None
    num_attempts: int = 200
    formalism: str | None = None

    def parameters(self) -> dict[str, Any]:
        """JSON-serialisable record of this experiment's configuration."""
        return {
            "length_km": self.length_km,
            "attenuation_db_per_km": self.attenuation_db_per_km,
            "detector_efficiency": self.detector_efficiency,
            "dark_count_rate_hz": self.dark_count_rate_hz,
            "memory_efficiency": self.memory_efficiency,
            "coherence_time_s": self.coherence_time_s,
            "raw_fidelity": self.raw_fidelity,
            "classical_delay_s": self.classical_delay_s,
            "num_attempts": self.num_attempts,
            "formalism": self.formalism or "ket_vector",
        }

    def run(self, seed: int = 0) -> pd.DataFrame:
        """Run the link and return one row per attempt.

        Attempts are launched one per millisecond of simulated time so that
        they do not interfere: each gets its own pair of memories and its own
        heralding window.  That measures the *per-attempt* success probability,
        which is what the analytic model predicts; a pipelined link that reuses
        memories has a different, lower effective rate and is a separate
        experiment.
        """
        bootstrap()
        from sequence.constants import KET_VECTOR_FORMALISM
        from sequence.kernel.quantum_manager import QuantumManager

        formalism = self.formalism or KET_VECTOR_FORMALISM
        previous_formalism = QuantumManager.get_active_formalism()
        QuantumManager.set_global_manager_formalism(formalism)
        try:
            return self._run(seed, formalism)
        finally:
            QuantumManager.set_global_manager_formalism(previous_formalism)

    def _run(self, seed: int, formalism: str) -> pd.DataFrame:
        """Build and run the link. Called by :meth:`run` inside the formalism guard."""
        from sequence.components.bsm import make_bsm
        from sequence.components.memory import MemoryArray
        from sequence.components.optical_channel import ClassicalChannel, QuantumChannel
        from sequence.entanglement_management.generation import (EntanglementGenerationA,
                                                                 EntanglementGenerationB)
        from sequence.kernel.event import Event
        from sequence.kernel.process import Process
        from sequence.kernel.timeline import Timeline
        from sequence.message import Message
        from sequence.topology.node import Node

        arm_m = self.length_km * 1e3 / 2.0
        attenuation_db_per_m = self.attenuation_db_per_km / 1e3
        classical_delay_s = (self.classical_delay_s if self.classical_delay_s is not None
                             else arm_m * C.FIBER_GROUP_INDEX / C.SPEED_OF_LIGHT_VACUUM)
        classical_delay_ps = max(int(classical_delay_s * 1e12), 1)

        # (memory, state, timestamp_ps, fidelity_at_that_moment). Fidelity is
        # captured here rather than read back later: an expired memory is
        # reset, which zeroes `memory.fidelity`, so reading it after the run
        # would report 0 for every pair that later timed out.
        log: list[tuple[Any, str, int, float]] = []

        class _ResourceManager:
            """Minimal stand-in: record every state transition with its timestamp."""

            def __init__(self, timeline):
                self.timeline = timeline

            def update(self, protocol, memory, state):
                log.append((memory, state, self.timeline.now(), memory.fidelity))

        class _Router(Node):
            def __init__(self, name, timeline):
                super().__init__(name, timeline)
                self.resource_manager = _ResourceManager(timeline)
                self.memory_array = None

            def init(self):
                self.memory_array.add_receiver(self)

            def get(self, photon, **kwargs):
                self.send_qubit(kwargs["dst"], photon)

            def memory_expire(self, memory):
                """A stored pair aged past its cutoff.

                ``QuantumRouter`` hands this to its resource manager, which
                releases the memory back to the pool.  Here we only need to
                record it: ``_collect`` keeps the *first* transition per
                memory, so a later expiry never overwrites the heralding
                outcome it is reporting.
                """
                log.append((memory, "EXPIRED", self.timeline.now(), memory.fidelity))

        class _BSMNode(Node):
            def __init__(self, name, timeline):
                super().__init__(name, timeline)
                self.bsm = None

            def receive_message(self, src: str, msg: "Message"):
                super().receive_message(src, msg)

            def receive_qubit(self, src: str, qubit):
                self.bsm.get(qubit)

        timeline = Timeline(formalism=formalism)
        timeline.show_progress = False

        left = _Router("left", timeline)
        right = _Router("right", timeline)
        middle = _BSMNode("middle", timeline)
        for index, node in enumerate((left, middle, right)):
            node.set_seed(3 * seed + index)

        for node, name in ((left, "qc_left"), (right, "qc_right")):
            channel = QuantumChannel(name, timeline, attenuation_db_per_m, arm_m)
            channel.set_ends(node, middle.name)

        for source in (left, right, middle):
            for destination in (left, right, middle):
                if source.name != destination.name:
                    channel = ClassicalChannel(
                        f"cc_{source.name}_{destination.name}", timeline,
                        arm_m, delay=classical_delay_ps)
                    channel.set_ends(source, destination.name)

        for node in (left, right):
            node.memory_array = MemoryArray(
                f"{node.name}.memory_array", timeline,
                num_memories=self.num_attempts,
                fidelity=self.raw_fidelity,
                efficiency=self.memory_efficiency,
                coherence_time=self.coherence_time_s)
            node.memory_array.owner = node

        detectors = [{"efficiency": self.detector_efficiency,
                      "dark_count": self.dark_count_rate_hz,
                      "count_rate": 1e11}] * 2
        middle.bsm = make_bsm("middle.bsm", timeline, encoding_type="single_atom",
                              detectors=detectors)
        middle.bsm.owner = middle

        generation_b = EntanglementGenerationB.create(
            middle, "eg_middle", others=[left.name, right.name])
        middle.bsm.attach(generation_b)

        timeline.init()

        # One attempt per millisecond of simulated time: far longer than the
        # heralding round trip at any distance we sweep, so attempts never
        # overlap and each success is attributable.
        attempt_spacing_ps = 1e9
        for index in range(self.num_attempts):
            protocol_left = EntanglementGenerationA.create(
                left, f"eg_left[{index}]", middle=middle.name, other=right.name,
                memory=left.memory_array[index])
            protocol_right = EntanglementGenerationA.create(
                right, f"eg_right[{index}]", middle=middle.name, other=left.name,
                memory=right.memory_array[index])
            left.protocols.append(protocol_left)
            right.protocols.append(protocol_right)
            protocol_left.set_others(protocol_right.name, right.name,
                                     [right.memory_array[index].name])
            protocol_right.set_others(protocol_left.name, left.name,
                                      [left.memory_array[index].name])

            for protocol in (protocol_left, protocol_right):
                timeline.schedule(Event(index * attempt_spacing_ps,
                                        Process(protocol, "start", [])))

        timeline.run()

        return self._collect(log, left, seed)

    def _collect(self, log, left, seed: int) -> pd.DataFrame:
        """Turn the resource-manager log into one row per attempt."""
        memory_names = {memory.name: index for index, memory
                        in enumerate(left.memory_array)}

        succeeded = np.zeros(self.num_attempts, dtype=bool)
        herald_time_s = np.full(self.num_attempts, np.nan)
        fidelity = np.full(self.num_attempts, np.nan)

        resolved = np.zeros(self.num_attempts, dtype=bool)
        for memory, state, timestamp, memory_fidelity in log:
            index = memory_names.get(memory.name)
            if index is None:  # the right-hand node's copy of the same attempt
                continue
            if resolved[index]:
                # A later EXPIRED event for a pair we already scored. The
                # attempt's outcome is its first transition, not its last.
                continue
            if state == "EXPIRED":
                continue
            resolved[index] = True
            herald_time_s[index] = timestamp * 1e-12
            if state == "ENTANGLED":
                succeeded[index] = True
                fidelity[index] = memory_fidelity

        return pd.DataFrame({
            "length_km": self.length_km,
            "attempt": np.arange(self.num_attempts),
            "succeeded": succeeded,
            "herald_time_s": herald_time_s,
            "fidelity": fidelity,
            "detector_efficiency": self.detector_efficiency,
            "attenuation_db_per_km": self.attenuation_db_per_km,
            "memory_efficiency": self.memory_efficiency,
            "coherence_time_s": self.coherence_time_s,
            "raw_fidelity": self.raw_fidelity,
            "seed": seed,
        })


def generate_link_dataset(lengths_km: Sequence[float] = (5.0, 10.0, 20.0, 40.0, 60.0),
                          num_attempts: int = 200,
                          repeats: int = 3,
                          seed: int = 0,
                          **experiment_kwargs) -> Dataset:
    """Sweep span length and record every attempt from real SeQUeNCe runs.

    Args:
        lengths_km: span lengths to sweep.  Loss is exponential, so the useful
            range is narrow: beyond ~80 km at default settings almost every
            attempt fails and the estimate becomes noise.
        num_attempts: attempts per run.  The success rate is a binomial
            estimate, so its standard error is
            :math:`\\sqrt{p(1-p)/n}` — at ``p = 0.05`` and ``n = 200`` that is
            already ±1.5 percentage points, i.e. ±30% relative.  Increase this
            before drawing conclusions at long distance.
        repeats: independent runs per length, with different seeds.
        seed: base seed.
        **experiment_kwargs: forwarded to :class:`LinkExperiment` — detector
            efficiency, attenuation, memory parameters.

    Returns:
        Dataset: one row per attempt, across all lengths and repeats.
    """
    frames = []
    run_index = 0
    for length_km in lengths_km:
        for repeat in range(repeats):
            experiment = LinkExperiment(length_km=length_km, num_attempts=num_attempts,
                                        **experiment_kwargs)
            frames.append(experiment.run(seed=seed + run_index))
            run_index += 1

    frame = pd.concat(frames, ignore_index=True)

    metadata = DatasetMetadata(
        name="link_entanglement",
        description=("One row per entanglement generation attempt on a two-node "
                     "midpoint-heralded link, simulated event-by-event in SeQUeNCe."),
        parameters={
            "lengths_km": list(lengths_km),
            "num_attempts": num_attempts,
            "repeats": repeats,
            **experiment_kwargs,
        },
        seed=seed,
        columns=describe_columns(COLUMNS, frame),
    )
    return Dataset(frame=frame, metadata=metadata)
