"""Tests for replaying a run from its provenance record."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from gwmock_pop.provenance import (
    EngineDescription,
    build_provenance_record,
    external_engine_origin,
    graph_simulation_origin,
    reconstruct_run,
    replay_catalogue,
    run_metadata,
)

_GRAPH_CONFIG = {
    "mass_1": {
        "sampler": {
            "function": "log_uniform",
            "arguments": {"minimum": 5.0, "maximum": 100.0},
        }
    },
    "mass_2": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@mass_1", "value": 1.4},
        }
    },
}


def _record(**overrides: Any) -> dict[str, Any]:
    """Build a replayable graph-simulation record."""
    record = build_provenance_record(
        origin=graph_simulation_origin(graph_config=_GRAPH_CONFIG, configuration={"parameters": {}, "run": {}}),
        source_type="bbh",
        parameter_names=["mass_1", "mass_2"],
        n_samples=5,
        file_format="hdf5",
        writer="tests",
        run=run_metadata(name="a-run", seed=17, seed_source="cli"),
    )
    record.update(overrides)
    return record


class TestReconstructRun:
    """Tests for rebuilding the simulator a record describes."""

    def test_rebuilds_the_simulator_and_the_sample_count(self) -> None:
        """The record alone yields a configured simulator and the row count."""
        simulator, n_samples = reconstruct_run(_record())

        assert n_samples == 5
        assert simulator.parameter_names == ["mass_1", "mass_2"]
        assert simulator.source_type == "bbh"
        assert simulator.rng_manager.resolved_seed == 17

    def test_replays_the_same_catalogue_as_the_original_run(self) -> None:
        """Replaying a record reproduces the run it describes."""
        record = _record()
        simulator, n_samples = reconstruct_run(record)
        expected = simulator.simulate(n_samples)

        replayed = replay_catalogue(record)

        assert list(replayed) == list(expected)
        for name in expected:
            np.testing.assert_array_equal(np.asarray(replayed[name]), np.asarray(expected[name]))

    def test_rejects_an_origin_it_cannot_replay(self) -> None:
        """An externally produced catalogue cannot be replayed by this package."""
        record = _record()
        record["origin"] = external_engine_origin(engine=EngineDescription(name="some-engine"))

        with pytest.raises(ValueError, match="external_engine"):
            reconstruct_run(record)

    def test_rejects_a_record_with_no_parameter_graph(self) -> None:
        """A record without the graph cannot describe a run."""
        record = _record()
        record["origin"]["configuration"]["parameters"] = {}

        with pytest.raises(ValueError, match="parameter graph"):
            reconstruct_run(record)

    def test_rejects_a_record_whose_columns_disagree_with_the_graph(self) -> None:
        """A record whose column list cannot come from its graph is not trustworthy."""
        record = _record()
        record["catalogue"]["parameter_names"] = ["mass_2", "mass_1"]

        with pytest.raises(ValueError, match="parameter_names"):
            reconstruct_run(record)

    def test_rejects_a_record_with_no_seed(self) -> None:
        """Without the seed actually used there is nothing to replay."""
        record = _record()
        record["run"] = None

        with pytest.raises(ValueError, match="seed"):
            reconstruct_run(record)

    def test_warns_when_the_recording_package_version_differs(self, caplog: pytest.LogCaptureFixture) -> None:
        """A version mismatch is surfaced rather than assumed harmless."""
        record = _record()
        record["tool"]["version"] = "0.0.0-not-this-one"

        with caplog.at_level(logging.WARNING, logger="gwmock_pop"):
            reconstruct_run(record)

        assert "0.0.0-not-this-one" in caplog.text
