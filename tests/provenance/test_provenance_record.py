"""Tests for building provenance records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import gwmock_pop
from gwmock_pop.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    EngineDescription,
    build_provenance_record,
    configuration_hash,
    converted_catalogue_origin,
    external_engine_origin,
    graph_simulation_origin,
    resolve_graph_config_defaults,
    run_metadata,
    source_code_provenance,
)

_FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _graph_config() -> dict[str, Any]:
    """Return a two-node graph config whose sampler has unstated defaults."""
    return {
        "distance": {
            "sampler": {
                "function": "uniform_comoving_volume_distance",
                "arguments": {"d_max": 1000.0},
            }
        },
        "coa_phase": {
            "transform": {
                "function": "constant_like",
                "arguments": {"reference": "@distance", "value": 0.0},
            }
        },
    }


class TestSourceCodeProvenance:
    """Tests for the package/source-state block."""

    def test_reports_the_running_package_version(self) -> None:
        """The record names the package and the version in use."""
        source = source_code_provenance()

        assert source["name"] == "gwmock-pop"
        assert source["version"] == gwmock_pop.__version__

    def test_git_state_is_absent_or_a_commit_and_dirty_flag(self) -> None:
        """From a checkout the commit and dirty flag are reported; otherwise ``None``."""
        git = source_code_provenance()["git"]

        if git is None:
            pytest.skip("Not running from a git checkout.")
        assert set(git) == {"commit", "dirty"}
        assert len(git["commit"]) == 40
        assert isinstance(git["dirty"], bool)


class TestConfigurationHash:
    """Tests for the configuration digest."""

    def test_is_stable_for_equal_payloads(self) -> None:
        """The same configuration hashes to the same digest."""
        assert configuration_hash({"a": 1}) == configuration_hash({"a": 1})

    def test_changes_with_the_payload(self) -> None:
        """A different configuration hashes differently."""
        assert configuration_hash({"a": 1}) != configuration_hash({"a": 2})

    def test_is_sensitive_to_key_order(self) -> None:
        """Graph node order fixes both column order and RNG consumption order.

        Two configs that differ only in node order are different runs, so the
        digest must not be order-blind.
        """
        assert configuration_hash({"a": 1, "b": 2}) != configuration_hash({"b": 2, "a": 1})

    def test_is_prefixed_with_the_algorithm(self) -> None:
        """The digest names its own algorithm."""
        assert configuration_hash({}).startswith("sha256:")


class TestResolveGraphConfigDefaults:
    """Tests for filling unstated sampler and transform defaults."""

    def test_fills_defaults_absent_from_the_config(self) -> None:
        """Defaults that live in the callable signature become explicit."""
        resolution = resolve_graph_config_defaults(_graph_config())

        arguments = resolution.config["distance"]["sampler"]["arguments"]
        assert arguments["d_max"] == 1000.0
        assert arguments["d_min"] == 0.0
        assert "hubble_constant" in arguments
        assert "n_grid" in arguments
        assert resolution.unresolved_nodes == ()

    def test_never_records_the_injected_arguments(self) -> None:
        """``key`` and ``n_samples`` are supplied per call, not configured."""
        resolution = resolve_graph_config_defaults(_graph_config())

        arguments = resolution.config["distance"]["sampler"]["arguments"]
        assert "key" not in arguments
        assert "n_samples" not in arguments

    def test_keeps_explicit_values(self) -> None:
        """A configured value is never replaced by the callable's default."""
        config = _graph_config()
        config["distance"]["sampler"]["arguments"]["d_min"] = 25.0

        resolution = resolve_graph_config_defaults(config)

        assert resolution.config["distance"]["sampler"]["arguments"]["d_min"] == 25.0

    def test_leaves_the_input_untouched(self) -> None:
        """Resolution is not allowed to mutate the caller's configuration."""
        config = _graph_config()

        resolve_graph_config_defaults(config)

        assert config["distance"]["sampler"]["arguments"] == {"d_max": 1000.0}

    def test_reports_nodes_whose_defaults_could_not_be_resolved(self) -> None:
        """An unimportable callable is named rather than silently skipped."""
        config = {"x": {"sampler": {"function": "not_a_real_sampler", "arguments": {}}}}

        resolution = resolve_graph_config_defaults(config)

        assert resolution.unresolved_nodes == ("x",)
        assert resolution.config["x"]["sampler"]["arguments"] == {}

    def test_passes_through_nodes_without_a_mapping_block(self) -> None:
        """String-style transforms are recorded verbatim."""
        config = {"x": {"transform": "some_expression"}}

        resolution = resolve_graph_config_defaults(config)

        assert resolution.config == config
        assert resolution.unresolved_nodes == ("x",)


class TestBuildProvenanceRecord:
    """Tests for assembling a complete record."""

    def test_carries_the_catalogue_shape_and_the_effective_run(self) -> None:
        """The record states what was written and what the run actually did."""
        origin = graph_simulation_origin(
            graph_config=_graph_config(),
            configuration={"run": {"seed": 1}, "parameters": {}},
            preset="bbh_flat",
        )
        record = build_provenance_record(
            origin=origin,
            source_type="bbh",
            parameter_names=["distance", "coa_phase"],
            n_samples=7,
            file_format="hdf5",
            writer="tests",
            run=run_metadata(name="a-run", seed=11, seed_source="cli"),
            created=_FIXED_TIME,
        )

        assert record["schema_version"] == PROVENANCE_SCHEMA_VERSION
        assert record["created_utc"] == "2026-01-02T03:04:05+00:00"
        assert record["tool"]["writer"] == "tests"
        assert record["catalogue"] == {
            "source_type": "bbh",
            "n_samples": 7,
            "parameter_names": ["distance", "coa_phase"],
            "file_format": "hdf5",
        }
        assert record["run"] == {"name": "a-run", "seed": 11, "seed_source": "cli"}
        assert record["origin"]["kind"] == "graph_simulation"
        assert record["origin"]["preset"] == "bbh_flat"

    def test_records_the_source_type_exactly_once(self) -> None:
        """The catalogue block is the single home of the source type."""
        record = build_provenance_record(
            origin=graph_simulation_origin(graph_config={}, configuration={}),
            source_type="bns",
            parameter_names=[],
            n_samples=0,
            file_format="csv",
            writer="tests",
        )

        assert record["catalogue"]["source_type"] == "bns"
        assert "source_type" not in record["origin"]

    def test_graph_origin_hashes_the_configuration_it_records(self) -> None:
        """The digest is derived from the stored configuration, not a second copy."""
        origin = graph_simulation_origin(
            graph_config=_graph_config(),
            configuration={"parameters": {}, "run": {}},
        )

        assert origin["configuration_hash"] == configuration_hash(origin["configuration"])

    def test_graph_origin_stores_the_resolved_graph_in_the_configuration(self) -> None:
        """There is one copy of the parameter graph, and it has defaults filled in."""
        origin = graph_simulation_origin(
            graph_config=_graph_config(),
            configuration={"parameters": {"stale": "value"}, "run": {}},
        )

        parameters = origin["configuration"]["parameters"]
        assert "stale" not in parameters
        assert parameters["distance"]["sampler"]["arguments"]["d_min"] == 0.0

    def test_external_engine_origin_carries_the_engine_and_its_run_record(self) -> None:
        """An externally produced catalogue names its engine and where its run is described."""
        origin = external_engine_origin(
            engine=EngineDescription(
                name="some-engine",
                version="2.1.0",
                config_hash="sha256:abc",
                run_record="https://example.invalid/runs/17",
            ),
            input_path="/data/catalogue.hdf5",
            fetch={"url": "https://example.invalid/c.hdf5", "etag": 'W/"1"', "cache_key": "abc123"},
        )

        assert origin["kind"] == "external_engine"
        assert origin["engine"] == {
            "name": "some-engine",
            "version": "2.1.0",
            "config_hash": "sha256:abc",
            "run_record": "https://example.invalid/runs/17",
        }
        assert origin["input"]["fetch"]["etag"] == 'W/"1"'
        assert origin["input"]["path"] == "/data/catalogue.hdf5"

    def test_converted_origin_chains_the_upstream_record(self) -> None:
        """A converted catalogue keeps the record of the file it came from."""
        upstream = {"schema_version": PROVENANCE_SCHEMA_VERSION, "marker": "upstream"}

        origin = converted_catalogue_origin(
            input_path="/data/in.csv",
            column_map={"m1": "detector_frame_mass_1"},
            upstream=upstream,
        )

        assert origin["kind"] == "converted_catalogue"
        assert origin["input"]["provenance"] == upstream
        assert origin["column_map"] == {"m1": "detector_frame_mass_1"}
