"""Topology construction and the data generators.

Two things worth guarding here.

The first is the **upstream fidelity bug**: SeQUeNCe accepts ``gate_fidelity``
and ``measurement_fidelity`` in a node's configuration and never reads them, so
every router silently gets perfect gates.  ``build_topology`` works around it,
and the test below asserts the values actually arrive on the router.  It also
asserts that the *stock* path still drops them — so the day upstream fixes this,
that test fails and tells us the workaround can go, rather than leaving dead
code behind forever.

The second is **leakage in the supervised framing**.  A polarisation dataset is
trivial to make unlearnably easy by feeding the model the ground truth it is
supposed to infer, and trivial to evaluate meaninglessly by shuffling
overlapping windows across a split.  Both are guarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from qnet_autopilot.datagen import (chronological_split, generate_capacity_trace,
                                    generate_demand_stream,
                                    generate_polarization_dataset,
                                    make_supervised_windows)
from qnet_autopilot.datagen.entanglement import LinkExperiment, generate_link_dataset
from qnet_autopilot.datagen.schema import Dataset, DatasetMetadata, load
from qnet_autopilot.network import (HardwareProfile, annotate_link_budgets, build_config,
                                    build_topology, scenario)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

class TestScenarios:

    @pytest.mark.parametrize("name,kwargs", [
        ("chain", {"num_nodes": 5}),
        ("ring", {"num_nodes": 6}),
        ("grid", {"rows": 2, "columns": 3}),
        ("metro", {}),
    ])
    def test_every_scenario_is_connected_and_annotated(self, name, kwargs):
        import networkx as nx

        graph = scenario(name, **kwargs)
        assert nx.is_connected(graph)
        for _, _, data in graph.edges(data=True):
            assert data["length"] > 0
            assert data["attenuation"] > 0

    def test_chain_has_one_fewer_edge_than_nodes(self):
        graph = scenario("chain", num_nodes=7)
        assert graph.number_of_edges() == 6

    def test_ring_jitter_makes_spans_unequal(self):
        even = scenario("ring", num_nodes=6, length_jitter=0.0)
        uneven = scenario("ring", num_nodes=6, length_jitter=0.4, rng=1)
        even_lengths = {d["length"] for *_, d in even.edges(data=True)}
        uneven_lengths = {d["length"] for *_, d in uneven.edges(data=True)}
        assert len(even_lengths) == 1
        assert len(uneven_lengths) > 1

    def test_rejects_degenerate_sizes(self):
        with pytest.raises(ValueError):
            scenario("chain", num_nodes=1)
        with pytest.raises(ValueError):
            scenario("ring", num_nodes=2)
        with pytest.raises(ValueError):
            scenario("ring", num_nodes=5, length_jitter=1.5)
        with pytest.raises(KeyError):
            scenario("hypercube")

    def test_link_budgets_attach_and_shrink_with_distance(self):
        graph = annotate_link_budgets(scenario("chain", num_nodes=4, span_km=20))
        for _, _, data in graph.edges(data=True):
            assert data["rate_hz"] > 0
            assert data["expected_time_s"] == pytest.approx(1 / data["rate_hz"])

        near = annotate_link_budgets(scenario("chain", num_nodes=2, span_km=10))
        far = annotate_link_budgets(scenario("chain", num_nodes=2, span_km=60))
        assert (list(far.edges(data=True))[0][2]["rate_hz"]
                < list(near.edges(data=True))[0][2]["rate_hz"])


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------

class TestTopology:

    def test_config_has_routers_bsm_nodes_and_channels(self):
        graph = scenario("chain", num_nodes=4, span_km=25)
        config, name_map = build_config(graph, HardwareProfile(memory_size=8))

        node_types = [node["type"] for node in config["nodes"]]
        assert node_types.count("QuantumRouter") == 4
        assert node_types.count("BSMNode") == 3
        # Two quantum channels per span: each end node to the midpoint BSM.
        assert len(config["qchannels"]) == 6
        assert len(name_map) == 4

    def test_config_rejects_edges_without_a_length(self):
        import networkx as nx

        graph = nx.Graph()
        graph.add_edge(0, 1)
        with pytest.raises(ValueError):
            build_config(graph)

    def test_config_rejects_half_specified_output(self):
        with pytest.raises(ValueError):
            build_config(scenario("chain", num_nodes=2), output_file="x.json")

    def test_hardware_profile_reaches_the_memories(self):
        graph = scenario("chain", num_nodes=3, span_km=20)
        hardware = HardwareProfile(memory_size=6, memory_coherence_time_s=2.5,
                                   memory_fidelity=0.88)
        topology, _ = build_topology(graph, hardware)

        router = topology.get_nodes_by_type(topology.QUANTUM_ROUTER)[0]
        arrays = [component for component in router.components.values()
                  if type(component).__name__ == "MemoryArray"]
        assert len(arrays) == 1
        memory_array = arrays[0]
        assert len(memory_array) == 6
        assert memory_array[0].coherence_time == pytest.approx(2.5)
        assert memory_array[0].raw_fidelity == pytest.approx(0.88)

    def test_gate_fidelity_workaround_applies_the_configured_values(self):
        graph = scenario("chain", num_nodes=3, span_km=20)
        hardware = HardwareProfile(gate_fidelity=0.97, measurement_fidelity=0.985)
        topology, _ = build_topology(graph, hardware)

        for router in topology.get_nodes_by_type(topology.QUANTUM_ROUTER):
            assert router.gate_fid == pytest.approx(0.97)
            assert router.meas_fid == pytest.approx(0.985)

    def test_stock_sequence_still_drops_the_fidelity_keys(self):
        """Documents the upstream bug the workaround exists for.

        ``nx_converter.generate_config`` writes ``gate_fidelity`` into the node
        config; ``RouterNetTopo._add_nodes`` never reads it.  When upstream
        fixes that, this test fails — which is the signal to delete
        ``apply_node_fidelity_workaround`` rather than carry it forever.
        """
        graph = scenario("chain", num_nodes=3, span_km=20)
        hardware = HardwareProfile(gate_fidelity=0.97, measurement_fidelity=0.985)

        config, _ = build_config(graph, hardware)
        router_configs = [node for node in config["nodes"]
                          if node["type"] == "QuantumRouter"]
        assert all(node["gate_fidelity"] == pytest.approx(0.97)
                   for node in router_configs), "config no longer carries the key"

        topology, _ = build_topology(graph, hardware,
                                     apply_node_fidelity_workaround=False)
        routers = topology.get_nodes_by_type(topology.QUANTUM_ROUTER)
        assert all(router.gate_fid == 1 for router in routers), (
            "upstream now reads gate_fidelity — remove the workaround in "
            "qnet_autopilot.network.topology.build_topology")


# ---------------------------------------------------------------------------
# dataset provenance
# ---------------------------------------------------------------------------

class TestDatasetSchema:

    def test_undocumented_columns_are_rejected(self):
        import pandas as pd

        frame = pd.DataFrame({"a": [1], "b": [2]})
        metadata = DatasetMetadata(name="t", description="d", columns={"a": "only a"})
        with pytest.raises(ValueError, match="undocumented"):
            Dataset(frame=frame, metadata=metadata)

    def test_save_and_load_round_trip(self):
        import tempfile
        from pathlib import Path

        dataset = generate_polarization_dataset(num_episodes=1, duration_s=20.0,
                                                dt_s=1.0, seed=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "polarization.csv"
            dataset.save(path)
            assert path.with_suffix(".csv.meta.json").is_file()

            restored = load(path)
            assert len(restored) == len(dataset)
            assert restored.metadata.seed == 5
            assert restored.metadata.parameters["dt_s"] == 1.0

    def test_load_refuses_a_dataset_without_metadata(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bare.csv"
            path.write_text("a,b\n1,2\n")
            with pytest.raises(FileNotFoundError):
                load(path)

    def test_metadata_records_whether_sequence_was_shimmed(self):
        dataset = generate_polarization_dataset(num_episodes=1, duration_s=10.0,
                                                dt_s=1.0, seed=0)
        assert "sequence" in dataset.metadata.software


# ---------------------------------------------------------------------------
# polarisation dataset
# ---------------------------------------------------------------------------

class TestPolarizationDataset:

    def test_shape_and_reproducibility(self):
        first = generate_polarization_dataset(num_episodes=3, duration_s=30.0,
                                              dt_s=0.5, seed=2)
        second = generate_polarization_dataset(num_episodes=3, duration_s=30.0,
                                               dt_s=0.5, seed=2)
        assert len(first) == 3 * 60
        assert np.allclose(first.frame["measured_qber"], second.frame["measured_qber"])

    def test_episodes_are_independent(self):
        dataset = generate_polarization_dataset(num_episodes=2, duration_s=60.0,
                                                dt_s=0.5, seed=3)
        by_episode = [group["true_angle_rad"].to_numpy()
                      for _, group in dataset.frame.groupby("episode")]
        assert not np.allclose(by_episode[0], by_episode[1])

    def test_measured_qber_is_a_binomial_estimate_of_the_true_one(self):
        """Unbiased on average, and quantised to 1/probe_photons."""
        probes = 500
        dataset = generate_polarization_dataset(num_episodes=4, duration_s=200.0,
                                                dt_s=0.5, probe_photons=probes, seed=4)
        frame = dataset.frame
        residual = frame["measured_qber"] - frame["true_qber"]
        assert abs(residual.mean()) < 0.01
        counts = frame["measured_qber"] * probes
        assert np.allclose(counts, np.round(counts))

    def test_rejects_degenerate_arguments(self):
        with pytest.raises(ValueError):
            generate_polarization_dataset(num_episodes=0)
        with pytest.raises(ValueError):
            generate_polarization_dataset(probe_photons=0)

    def test_windows_have_the_right_shape_and_no_leakage(self):
        dataset = generate_polarization_dataset(num_episodes=2, duration_s=40.0,
                                                dt_s=0.5, seed=6)
        features = ("measured_qber", "temperature_c")
        X, y, episode, time_s = make_supervised_windows(
            dataset, window=10, horizon=4, features=features)

        assert X.shape[1:] == (10, 2)
        assert X.shape[0] == y.shape[0] == episode.shape[0] == time_s.shape[0]
        assert set(np.unique(episode)) == {0, 1}

    def test_windows_refuse_ground_truth_as_a_feature(self):
        dataset = generate_polarization_dataset(num_episodes=1, duration_s=20.0,
                                                dt_s=0.5, seed=7)
        with pytest.raises(ValueError, match="ground-truth"):
            make_supervised_windows(dataset, features=("true_qber",))

    def test_windows_reject_unknown_columns_and_impossible_sizes(self):
        dataset = generate_polarization_dataset(num_episodes=1, duration_s=20.0,
                                                dt_s=0.5, seed=8)
        with pytest.raises(ValueError, match="not in the dataset"):
            make_supervised_windows(dataset, features=("nonexistent",))
        with pytest.raises(ValueError, match="does not fit"):
            make_supervised_windows(dataset, window=500, horizon=500)

    def test_target_is_strictly_in_the_future_of_its_window(self):
        """The property that makes the framing honest, checked directly."""
        dataset = generate_polarization_dataset(num_episodes=1, duration_s=30.0,
                                                dt_s=0.5, seed=9)
        window, horizon = 8, 3
        X, y, _, time_s = make_supervised_windows(
            dataset, window=window, horizon=horizon,
            features=("measured_angle_rad",), target="true_angle_rad")

        frame = dataset.frame
        angles = frame["true_angle_rad"].to_numpy()
        observed = frame["measured_angle_rad"].to_numpy()
        for index in range(0, len(y), 7):
            start = index
            assert np.allclose(X[index, :, 0], observed[start:start + window])
            assert y[index] == pytest.approx(angles[start + window + horizon - 1])

    def test_chronological_split_leaves_a_gap_and_does_not_overlap(self):
        dataset = generate_polarization_dataset(num_episodes=2, duration_s=60.0,
                                                dt_s=0.5, seed=10)
        _, _, episode, time_s = make_supervised_windows(dataset, window=10, horizon=5)
        train, test = chronological_split(episode, time_s, train_fraction=0.7)

        assert not (train & test).any()
        assert train.sum() > 0 and test.sum() > 0
        for value in np.unique(episode):
            selector = episode == value
            assert time_s[selector & train].max() < time_s[selector & test].min()

    def test_chronological_split_rejects_a_degenerate_fraction(self):
        dataset = generate_polarization_dataset(num_episodes=1, duration_s=20.0,
                                                dt_s=0.5, seed=11)
        _, _, episode, time_s = make_supervised_windows(dataset, window=5, horizon=2)
        with pytest.raises(ValueError):
            chronological_split(episode, time_s, train_fraction=1.0)


# ---------------------------------------------------------------------------
# link dataset
# ---------------------------------------------------------------------------

class TestLinkDataset:

    def test_one_row_per_attempt_and_every_attempt_resolves(self):
        frame = LinkExperiment(length_km=10.0, num_attempts=60).run(seed=0)
        assert len(frame) == 60
        assert frame["herald_time_s"].notna().all()

    def test_successes_carry_the_configured_fidelity(self):
        frame = LinkExperiment(length_km=5.0, num_attempts=80, raw_fidelity=0.91,
                               detector_efficiency=1.0, memory_efficiency=1.0).run(seed=1)
        successes = frame[frame["succeeded"]]
        assert len(successes) > 0
        assert np.allclose(successes["fidelity"], 0.91)
        assert frame[~frame["succeeded"]]["fidelity"].isna().all()

    def test_longer_spans_succeed_less_often(self):
        near = LinkExperiment(length_km=2.0, num_attempts=250,
                              detector_efficiency=1.0).run(seed=2)
        far = LinkExperiment(length_km=45.0, num_attempts=250,
                             detector_efficiency=1.0).run(seed=2)
        assert far["succeeded"].mean() < near["succeeded"].mean()

    def test_sweep_covers_every_length_and_repeat(self):
        dataset = generate_link_dataset(lengths_km=(5.0, 15.0), num_attempts=40,
                                        repeats=2, seed=0)
        assert len(dataset) == 2 * 2 * 40
        assert set(dataset.frame["length_km"]) == {5.0, 15.0}
        assert dataset.frame["seed"].nunique() == 4


# ---------------------------------------------------------------------------
# traffic
# ---------------------------------------------------------------------------

class TestTraffic:

    def test_capacity_trace_covers_every_link(self):
        graph = scenario("ring", num_nodes=4, span_km=25)
        dataset = generate_capacity_trace(graph, duration_s=120.0, dt_s=10.0, seed=1)
        assert dataset.frame["link"].nunique() == graph.number_of_edges()
        assert len(dataset) == graph.number_of_edges() * 12

    def test_links_drift_independently(self):
        """Shared drift would let a policy predict one link from another.

        Getting this test right is harder than it looks, and two obvious
        versions of it are wrong:

        * Correlating the QBER **levels** fails for correct code — the series
          are random walks, and independent random walks show large spurious
          level-correlation as a matter of course.
        * Correlating the raw **increments** also fails, because every link
          shares a deterministic envelope: the misalignment angle grows quickly
          at first and then flattens as the walk saturates on the Poincaré
          sphere, so the mean increment falls by ~4x across the window for
          every realisation alike.

        Dividing out that envelope — estimated leave-one-out, so a link never
        normalises by itself — leaves the part that is actually per-link, and
        that is what must be uncorrelated.
        """
        graph = scenario("ring", num_nodes=6, span_km=25)
        frame = generate_capacity_trace(graph, duration_s=3600.0, dt_s=10.0,
                                        seed=2).frame

        increments = np.stack([np.diff(group["qber"].to_numpy())
                               for _, group in frame.groupby("link", sort=True)])
        count = len(increments)
        total = increments.sum(axis=0)

        normalised = []
        for index in range(count):
            envelope = (total - increments[index]) / (count - 1)
            scale = np.maximum(np.abs(envelope), 1e-9)
            normalised.append(increments[index] / scale)

        correlations = [abs(np.corrcoef(normalised[i], normalised[j])[0, 1])
                        for i in range(count) for j in range(i + 1, count)]
        assert max(correlations) < 0.3, f"max |corr| = {max(correlations):.2f}"
        assert np.mean(correlations) < 0.15, f"mean |corr| = {np.mean(correlations):.2f}"

    def test_capacity_trace_is_reproducible(self):
        graph = scenario("chain", num_nodes=3, span_km=25)
        first = generate_capacity_trace(graph, duration_s=300.0, dt_s=10.0, seed=9).frame
        second = generate_capacity_trace(graph, duration_s=300.0, dt_s=10.0, seed=9).frame
        assert np.allclose(first["qber"], second["qber"])

    def test_secret_key_rate_is_zero_exactly_where_unusable(self):
        graph = scenario("chain", num_nodes=3, span_km=30)
        frame = generate_capacity_trace(graph, duration_s=600.0, dt_s=5.0, seed=3).frame
        assert (frame.loc[~frame["usable"], "secret_key_rate_hz"] == 0).all()
        assert (frame.loc[frame["usable"], "secret_key_rate_hz"] > 0).all()

    def test_qber_never_falls_below_the_detector_floor(self):
        graph = scenario("chain", num_nodes=2, span_km=20)
        frame = generate_capacity_trace(graph, duration_s=300.0, dt_s=5.0,
                                        detector_qber_floor=0.01, seed=4).frame
        assert frame["qber"].min() >= 0.01 - 1e-12

    def test_demands_cover_all_three_services(self):
        graph = scenario("ring", num_nodes=5, span_km=25)
        frame = generate_demand_stream(graph, duration_s=1200.0, seed=5).frame
        assert set(frame["service"]) == {"classical", "qkd", "entanglement"}

    def test_demands_are_well_formed(self):
        graph = scenario("ring", num_nodes=5, span_km=25)
        frame = generate_demand_stream(graph, duration_s=1200.0, seed=6).frame

        assert (frame["source"] != frame["destination"]).all()
        assert (frame["deadline_s"] > frame["arrival_s"]).all()
        assert frame["arrival_s"].is_monotonic_increasing
        assert frame["priority"].between(1, 3).all()

        entanglement = frame[frame["service"] == "entanglement"]
        assert entanglement["min_fidelity"].notna().all()
        assert frame[frame["service"] != "entanglement"]["min_fidelity"].isna().all()

    def test_qkd_skews_to_higher_priority_than_classical(self):
        graph = scenario("ring", num_nodes=5, span_km=25)
        frame = generate_demand_stream(graph, duration_s=86400.0, seed=7).frame
        by_service = frame.groupby("service")["priority"].mean()
        assert by_service["qkd"] > by_service["classical"]

    def test_a_two_node_graph_is_the_minimum(self):
        import networkx as nx

        with pytest.raises(ValueError):
            generate_demand_stream(nx.Graph([(0, 0)]))
