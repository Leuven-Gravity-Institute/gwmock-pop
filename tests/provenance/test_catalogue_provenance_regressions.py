"""Regression tests for population-catalogue provenance.

Each test here pins a defect that let a catalogue leave this package with no
machine-readable record of how it was produced:

* ``write_population_catalogue`` -- the writer the CLI actually calls -- wrote a
  bare ``data`` dataset and nothing else: no attributes, no metadata group, no
  package version, no seed and no configuration.
* the simulator base class carried the only code that wrote an HDF5 ``metadata``
  group, nothing outside the test suite called it, and the layout it wrote was
  one this package's own reader rejects: an unnamed 2-D ``data`` dataset for
  HDF5, and a header-less file for CSV.
* ``OutputConfiguration.save_metadata`` defaulted to ``True`` and controlled
  nothing at all.
* ``MainConfiguration`` was disconnected from the simulate path, so ``run.seed``,
  ``run.name`` and the whole ``output`` block were silently ignored and the CLI
  read a bare ``--seed`` option instead.

Together those left two divergent persistence paths, with the metadata-free one
wired to the CLI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

import gwmock_pop
from gwmock_pop import read_population_catalogue, write_population_catalogue
from gwmock_pop.cli.main import app
from gwmock_pop.provenance import (
    HDF5_PROVENANCE_PATH,
    PROVENANCE_SCHEMA_VERSION,
    EngineDescription,
    provenance_sidecar_path,
    read_provenance,
    replay_catalogue,
)

_RUNNER = CliRunner()


def _simulate(*args: str) -> None:
    """Run the simulate command and fail loudly on a non-zero exit."""
    result = _RUNNER.invoke(app, ["simulate", *args])
    assert result.exit_code == 0, result.output


def _assert_same_catalogue(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    """Assert two catalogues hold the same columns, in order, with equal values."""
    assert list(left) == list(right)
    for name, values in left.items():
        np.testing.assert_array_equal(
            np.asarray(values, dtype=float),
            np.asarray(right[name], dtype=float),
        )


# ---------------------------------------------------------------------------
# The writer the CLI calls wrote no metadata at all.
# ---------------------------------------------------------------------------


def test_cli_hdf5_catalogue_carries_a_provenance_record(tmp_path: Path) -> None:
    """The HDF5 file the CLI writes must hold a record beside its ``data`` dataset."""
    output_path = tmp_path / "population.hdf5"
    _simulate("--config", "bbh_flat", "--n", "8", "--output", str(output_path), "--seed", "42")

    with h5py.File(output_path, "r") as handle:
        assert HDF5_PROVENANCE_PATH in handle
        stored_columns = list(handle["data"].dtype.names or ())

    record = read_provenance(output_path)
    assert record is not None
    assert record["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert record["created_utc"]
    assert record["tool"]["name"] == "gwmock-pop"
    assert record["tool"]["version"] == gwmock_pop.__version__
    assert record["catalogue"] == {
        "source_type": "bbh",
        "n_samples": 8,
        "parameter_names": stored_columns,
        "file_format": "hdf5",
    }
    assert record["run"]["seed"] == 42
    assert record["origin"]["preset"] == "bbh_flat"
    assert record["origin"]["configuration"]["parameters"]


def test_cli_records_the_complete_resolved_configuration(
    tmp_path: Path, write_graph_config: Callable[..., Path]
) -> None:
    """Sampler defaults absent from the input file must appear in the record."""
    config_path = write_graph_config(
        parameters=(
            "parameters:\n"
            "  distance:\n"
            "    sampler:\n"
            "      function: uniform_comoving_volume_distance\n"
            "      arguments:\n"
            "        d_max: 1000.0\n"
        ),
    )
    output_path = tmp_path / "population.hdf5"
    _simulate("--config", str(config_path), "--n", "4", "--output", str(output_path), "--seed", "1")

    record = read_provenance(output_path)
    assert record is not None
    arguments = record["origin"]["configuration"]["parameters"]["distance"]["sampler"]["arguments"]
    assert arguments["d_max"] == 1000.0
    # These are defaults of the sampler function, never written in the config file.
    # A record of only the overrides cannot reconstruct the run.
    assert set(arguments) >= {"d_max", "d_min", "hubble_constant", "omega_m", "max_redshift", "n_grid"}
    assert record["origin"]["unresolved_default_nodes"] == []
    # Every block of the configuration tree is present, defaults filled in.
    configuration = record["origin"]["configuration"]
    assert set(configuration) >= {"config_version", "run", "cosmology", "advanced", "parameters"}
    assert configuration["run"]["output"]["format"] == "hdf5"


def test_write_population_catalogue_embeds_the_record_it_is_given(tmp_path: Path) -> None:
    """The library writer must persist a caller-supplied record, not drop it."""
    population = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    record = {"schema_version": PROVENANCE_SCHEMA_VERSION, "marker": "embedded"}
    output_path = tmp_path / "catalogue.hdf5"

    write_population_catalogue(output_path=output_path, population=population, provenance=record)

    assert read_provenance(output_path) == record
    _assert_same_catalogue(read_population_catalogue(output_path), population)


# ---------------------------------------------------------------------------
# Two divergent persistence paths, only one of them reachable.
# ---------------------------------------------------------------------------


def test_simulator_persistence_goes_through_the_shared_named_column_writer(tmp_path: Path) -> None:
    """A simulator must persist a catalogue this package's own reader accepts.

    The retired persistence path wrote an unnamed 2-D ``data`` dataset for HDF5
    and a header-less file for CSV, so ``read_population_catalogue`` could not
    read back anything the simulator had written.
    """
    simulator = gwmock_pop.GraphSimulator.from_preset("bbh_flat", seed=3)
    simulator.simulate(6)
    output_path = tmp_path / "population.hdf5"

    simulator.save_catalogue(output_path)

    catalogue = read_population_catalogue(output_path)
    assert list(catalogue) == simulator.parameter_names
    record = read_provenance(output_path)
    assert record is not None
    assert record["run"]["seed"] == 3
    assert record["catalogue"]["n_samples"] == 6
    _assert_same_catalogue(catalogue, replay_catalogue(record))


def test_a_library_written_catalogue_replays_from_its_own_record(tmp_path: Path) -> None:
    """The library path is publication-grade too, not only the CLI one."""
    simulator = gwmock_pop.BBHSimulator(seed=5)
    simulator.simulate(4)
    output_path = tmp_path / "population.hdf5"

    simulator.save_catalogue(output_path)

    record = read_provenance(output_path)
    assert record is not None
    assert record["run"] == {"name": None, "seed": 5, "seed_source": "library"}
    _assert_same_catalogue(read_population_catalogue(output_path), replay_catalogue(record))


def test_the_package_keeps_exactly_one_catalogue_persistence_path() -> None:
    """The divergent simulator-side writer and reader must be gone for good."""
    from gwmock_pop.simulators.simulator import Simulator

    assert hasattr(Simulator, "save_catalogue")
    assert not hasattr(Simulator, "save")
    assert not hasattr(Simulator, "load")
    assert not hasattr(Simulator, "_save_hdf5")
    assert not hasattr(Simulator, "_load_hdf5")


# ---------------------------------------------------------------------------
# save_metadata controlled nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("declared", "expect_record"), [("true", True), ("false", False)])
def test_output_save_metadata_controls_whether_the_record_is_written(
    tmp_path: Path, declared: str, expect_record: bool, write_graph_config: Callable[..., Path]
) -> None:
    """``run.output.save_metadata`` must switch the record on and off."""
    config_path = write_graph_config(
        preamble=f"run:\n  output:\n    save_metadata: {declared}\n",
    )
    output_path = tmp_path / "population.hdf5"
    _simulate("--config", str(config_path), "--n", "4", "--output", str(output_path), "--seed", "5")

    assert (read_provenance(output_path) is not None) is expect_record
    # The catalogue itself is written either way.
    assert read_population_catalogue(output_path)["mass_1"].shape == (4,)


def test_csv_catalogue_carries_its_record_in_a_sidecar(tmp_path: Path, write_graph_config: Callable[..., Path]) -> None:
    """CSV has nowhere to embed a record, so it gets a JSON sidecar."""
    config_path = write_graph_config()
    output_path = tmp_path / "population.csv"
    _simulate("--config", str(config_path), "--n", "4", "--output", str(output_path), "--seed", "5")

    sidecar = provenance_sidecar_path(output_path)
    assert sidecar.is_file()
    record = read_provenance(output_path)
    assert record == json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["catalogue"]["file_format"] == "csv"


# ---------------------------------------------------------------------------
# MainConfiguration was orphaned from the simulate path.
# ---------------------------------------------------------------------------


def test_simulate_honours_the_configured_run_seed(tmp_path: Path, write_graph_config: Callable[..., Path]) -> None:
    """A configured ``run.seed`` must drive the run when ``--seed`` is omitted."""
    config_path = write_graph_config(preamble="run:\n  name: seeded-run\n  seed: 1234\n")
    from_config = tmp_path / "from_config.hdf5"
    from_cli = tmp_path / "from_cli.hdf5"

    _simulate("--config", str(config_path), "--n", "16", "--output", str(from_config))
    _simulate("--config", str(config_path), "--n", "16", "--output", str(from_cli), "--seed", "1234")

    _assert_same_catalogue(read_population_catalogue(from_config), read_population_catalogue(from_cli))
    assert read_provenance(from_config)["run"] == {"name": "seeded-run", "seed": 1234, "seed_source": "config"}
    assert read_provenance(from_cli)["run"]["seed_source"] == "cli"


def test_cli_seed_overrides_the_configured_run_seed(tmp_path: Path, write_graph_config: Callable[..., Path]) -> None:
    """``--seed`` must win over a configured ``run.seed``."""
    config_path = write_graph_config(preamble="run:\n  seed: 1234\n")
    overridden = tmp_path / "overridden.hdf5"
    reference = tmp_path / "reference.hdf5"

    _simulate("--config", str(config_path), "--n", "16", "--output", str(overridden), "--seed", "99")
    _simulate("--config", str(config_path), "--n", "16", "--output", str(reference), "--seed", "99")

    _assert_same_catalogue(read_population_catalogue(overridden), read_population_catalogue(reference))
    assert read_provenance(overridden)["run"]["seed"] == 99


def test_simulate_honours_the_configured_sample_count(tmp_path: Path, write_graph_config: Callable[..., Path]) -> None:
    """``run.n_samples`` must supply the sample count when ``--n`` is omitted."""
    config_path = write_graph_config(preamble="run:\n  n_samples: 12\n  seed: 7\n")
    output_path = tmp_path / "population.hdf5"

    _simulate("--config", str(config_path), "--output", str(output_path))

    assert read_population_catalogue(output_path)["mass_1"].shape == (12,)
    assert read_provenance(output_path)["catalogue"]["n_samples"] == 12


def test_simulate_without_a_sample_count_anywhere_fails_clearly(
    tmp_path: Path, write_graph_config: Callable[..., Path]
) -> None:
    """Neither ``--n`` nor a configured count is an error, not a silent default."""
    config_path = write_graph_config()
    output_path = tmp_path / "population.hdf5"

    result = _RUNNER.invoke(app, ["simulate", "--config", str(config_path), "--output", str(output_path)])

    assert result.exit_code == 1
    assert "run.n_samples" in result.output
    assert not output_path.exists()


def test_simulate_honours_the_configured_overwrite_flag(
    tmp_path: Path, write_graph_config: Callable[..., Path]
) -> None:
    """``run.output.overwrite`` must permit replacing an existing catalogue."""
    config_path = write_graph_config(preamble="run:\n  output:\n    overwrite: true\n")
    output_path = tmp_path / "population.hdf5"
    output_path.write_text("stale\n", encoding="utf-8")

    _simulate("--config", str(config_path), "--n", "4", "--output", str(output_path), "--seed", "5")

    assert read_population_catalogue(output_path)["mass_1"].shape == (4,)


def test_simulate_rejects_a_configured_format_that_contradicts_the_output_suffix(
    tmp_path: Path, write_graph_config: Callable[..., Path]
) -> None:
    """A declared ``output.format`` must not silently disagree with ``--output``."""
    config_path = write_graph_config(preamble="run:\n  output:\n    format: csv\n")
    output_path = tmp_path / "population.hdf5"

    result = _RUNNER.invoke(
        app, ["simulate", "--config", str(config_path), "--n", "4", "--output", str(output_path), "--seed", "5"]
    )

    assert result.exit_code == 1
    assert "csv" in result.output
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# Catalogues that came from somewhere else keep saying where.
# ---------------------------------------------------------------------------


def test_loader_reports_the_record_the_input_catalogue_carried(tmp_path: Path) -> None:
    """A loaded catalogue's own record is readable through the loader."""
    source = tmp_path / "source.hdf5"
    _simulate("--config", "bbh_flat", "--n", "4", "--output", str(source), "--seed", "2")

    loader = gwmock_pop.FilePopulationLoader("bbh", source)

    assert loader.provenance == read_provenance(source)


def test_loader_origin_folds_in_the_fetch_payload_and_chains_the_record(tmp_path: Path) -> None:
    """An externally derived catalogue names its engine without restating the fetch details."""
    source = tmp_path / "source.hdf5"
    _simulate("--config", "bbh_flat", "--n", "4", "--output", str(source), "--seed", "2")
    loader = gwmock_pop.FilePopulationLoader("bbh", source)

    origin = loader.provenance_origin(
        EngineDescription(name="some-engine", version="1.2.3", run_record="https://example.invalid/runs/9")
    )

    assert origin["kind"] == "external_engine"
    assert origin["engine"]["name"] == "some-engine"
    assert origin["engine"]["run_record"] == "https://example.invalid/runs/9"
    # The loader already records the fetch details; they are reused, not restated.
    assert origin["input"]["fetch"] == loader.metadata["fetch"]
    assert origin["input"]["provenance"] == read_provenance(source)


def test_convert_chains_the_record_of_the_file_it_converted(tmp_path: Path) -> None:
    """A conversion must not be where a catalogue loses its provenance."""
    source = tmp_path / "source.hdf5"
    destination = tmp_path / "converted.csv"
    _simulate("--config", "bbh_flat", "--n", "4", "--output", str(source), "--seed", "2")

    result = _RUNNER.invoke(app, ["convert", "--input", str(source), "--output", str(destination)])
    assert result.exit_code == 0, result.output

    record = read_provenance(destination)
    assert record is not None
    assert record["origin"]["kind"] == "converted_catalogue"
    assert record["origin"]["input"]["path"] == str(source)
    assert record["origin"]["input"]["provenance"] == read_provenance(source)
    assert record["catalogue"]["file_format"] == "csv"


# ---------------------------------------------------------------------------
# Reproducibility: the part that gives the record teeth.
# ---------------------------------------------------------------------------


def test_simulate_rejects_a_configured_format_the_package_cannot_write(
    tmp_path: Path, write_graph_config: Callable[..., Path]
) -> None:
    """``output.format`` accepts a value no catalogue writer produces; say so."""
    config_path = write_graph_config(preamble="run:\n  output:\n    format: npz\n")
    output_path = tmp_path / "population.hdf5"

    result = _RUNNER.invoke(
        app, ["simulate", "--config", str(config_path), "--n", "4", "--output", str(output_path), "--seed", "5"]
    )

    assert result.exit_code == 1
    assert "not a catalogue" in result.output
    assert "npz" in result.output
    assert not output_path.exists()


def test_simulate_honours_the_configured_output_directory(
    tmp_path: Path, write_graph_config: Callable[..., Path]
) -> None:
    """A declared ``output.directory`` places a relative ``--output`` under it."""
    destination = tmp_path / "catalogues"
    config_path = write_graph_config(preamble=f"run:\n  output:\n    directory: {destination}\n")

    _simulate("--config", str(config_path), "--n", "4", "--output", "population.hdf5", "--seed", "5")

    assert (destination / "population.hdf5").is_file()


def test_same_seed_reproduces_the_catalogue_and_every_reconstruction_input(tmp_path: Path) -> None:
    """Two runs at one seed must agree on the data and on the whole record but its timestamp."""
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    for output_path in (first, second):
        _simulate("--config", "bbh_flat", "--n", "8", "--output", str(output_path), "--seed", "31")

    _assert_same_catalogue(read_population_catalogue(first), read_population_catalogue(second))

    first_record = read_provenance(first)
    second_record = read_provenance(second)
    assert first_record is not None
    assert second_record is not None
    assert first_record.pop("created_utc") != ""
    assert second_record.pop("created_utc") != ""
    assert first_record == second_record


def test_catalogue_is_reconstructible_from_its_provenance_record_alone(tmp_path: Path) -> None:
    """With no seed given, the drawn seed must be recorded and replay the catalogue."""
    output_path = tmp_path / "population.hdf5"
    _simulate("--config", "bbh_flat", "--n", "8", "--output", str(output_path))

    record = read_provenance(output_path)
    assert record is not None
    assert record["run"]["seed_source"] == "drawn"
    assert isinstance(record["run"]["seed"], int)

    _assert_same_catalogue(read_population_catalogue(output_path), replay_catalogue(record))


def test_replay_needs_nothing_but_the_record(tmp_path: Path, write_graph_config: Callable[..., Path]) -> None:
    """A record moved away from its catalogue and its config file still replays."""
    config_path = write_graph_config(preamble="run:\n  seed: 8\n")
    output_path = tmp_path / "population.hdf5"
    _simulate("--config", str(config_path), "--n", "6", "--output", str(output_path))

    record = read_provenance(output_path)
    stored = read_population_catalogue(output_path)
    detached = json.loads(json.dumps(record))
    config_path.unlink()
    output_path.unlink()

    _assert_same_catalogue(stored, replay_catalogue(detached))


def test_omitting_the_seed_draws_a_fresh_seed_each_run(tmp_path: Path, write_graph_config: Callable[..., Path]) -> None:
    """A recorded seed is only worth having if it was not a fixed fallback."""
    seeds = set()
    for index in range(3):
        output_path = tmp_path / f"population_{index}.hdf5"
        _simulate("--config", str(write_graph_config()), "--n", "2", "--output", str(output_path))
        seeds.add(read_provenance(output_path)["run"]["seed"])

    assert len(seeds) == 3
