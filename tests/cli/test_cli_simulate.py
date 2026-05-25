"""Tests for the CLI simulate command."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from gwmock_pop import FilePopulationLoader
from gwmock_pop.cli.main import app, setup_logging
from gwmock_pop.utils.log import LoggingLevel

_RUNNER = CliRunner()
_EXPECTED_BBH_COLUMNS = {
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "eccentricity",
    "distance",
    "coa_phase",
    "inclination",
    "theta_jn",
    "long_asc_node",
    "mean_per_ano",
    "coa_time",
    "right_ascension",
    "declination",
    "polarization_angle",
    "redshift",
    "f_ref",
}
_EXPECTED_BNS_COLUMNS = _EXPECTED_BBH_COLUMNS | {"lambda_1", "lambda_2"}


def _minimal_graph_config(path: Path) -> None:
    """Write a minimal graph config for the file-path CLI route."""
    path.write_text(
        """
parameters:
  mass_1:
    sampler:
      function: planck_tapered_broken_power_law_plus_two_peaks
      arguments:
        alpha_1: 1.72
        alpha_2: 4.51
        transition: 35.6
        minimum: 5.06
        maximum: 300.0
        mean_1: 9.76
        sigma_1: 0.649
        mean_2: 32.8
        sigma_2: 3.92
        taper_range: 4.32
        lambda_0: 0.361
        lambda_1: 0.586
""".strip(),
        encoding="utf-8",
    )


def test_simulate_command_writes_hdf5_from_packaged_preset(tmp_path: Path) -> None:
    """The CLI samples the packaged GWTC4 preset into a named-column HDF5 file."""
    output_path = tmp_path / "population.hdf5"

    result = _RUNNER.invoke(
        app,
        ["simulate", "--config", "gwtc4", "--n", "100", "--output", str(output_path), "--seed", "42"],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    with h5py.File(output_path, "r") as handle:
        dataset = handle["data"]
        assert dataset.shape == (100,)
        assert set(dataset.dtype.names or ()) == _EXPECTED_BBH_COLUMNS


def test_simulate_command_writes_hdf5_from_bns_packaged_preset(tmp_path: Path) -> None:
    """The CLI samples the packaged BNS flat preset into a named-column HDF5 file."""
    output_path = tmp_path / "bns_population.hdf5"

    result = _RUNNER.invoke(
        app,
        ["simulate", "--config", "bns_flat", "--n", "64", "--output", str(output_path), "--seed", "42"],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    with h5py.File(output_path, "r") as handle:
        dataset = handle["data"]
        assert dataset.shape == (64,)
        assert set(dataset.dtype.names or ()) == _EXPECTED_BNS_COLUMNS
        assert np.all(dataset["detector_frame_mass_1"] >= dataset["detector_frame_mass_2"])
        assert np.all(dataset["detector_frame_mass_1"] >= 1.0)
        assert np.all(dataset["detector_frame_mass_2"] >= 1.0)
        assert np.all(dataset["detector_frame_mass_1"] <= 3.0)
        assert np.all(dataset["detector_frame_mass_2"] <= 3.0)


def test_simulate_command_writes_csv_from_config_file(tmp_path: Path) -> None:
    """The CLI samples a graph config file into a header-based CSV file."""
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "population.csv"
    _minimal_graph_config(config_path)

    result = _RUNNER.invoke(
        app,
        ["simulate", "--config", str(config_path), "--n", "50", "--output", str(output_path), "--seed", "7"],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    data = np.genfromtxt(output_path, delimiter=",", names=True, dtype=float)
    assert np.atleast_1d(data).shape == (50,)
    assert data.dtype.names == ("mass_1",)


@pytest.mark.integration
def test_hdf5_output_round_trips_through_file_population_loader(tmp_path: Path) -> None:
    """The CLI HDF5 output is readable through FilePopulationLoader."""
    output_path = tmp_path / "population.hdf5"

    result = _RUNNER.invoke(
        app,
        ["simulate", "--config", "gwtc4", "--n", "100", "--output", str(output_path), "--seed", "11"],
    )

    assert result.exit_code == 0, result.output

    loader = FilePopulationLoader("bbh", output_path)
    reloaded = loader.simulate(100, seed=5)

    assert len(loader.parameter_names) == 21
    assert set(loader.parameter_names) == _EXPECTED_BBH_COLUMNS
    assert set(reloaded) == _EXPECTED_BBH_COLUMNS
    assert all(values.shape == (100,) for values in reloaded.values())


def test_simulate_command_rejects_existing_output_path(tmp_path: Path) -> None:
    """The CLI refuses to overwrite an existing output file."""
    output_path = tmp_path / "population.csv"
    output_path.write_text("existing\n", encoding="utf-8")

    result = _RUNNER.invoke(app, ["simulate", "--config", "gwtc4", "--n", "10", "--output", str(output_path)])

    assert result.exit_code == 1
    assert "Refusing to overwrite existing output file" in result.output


def test_simulate_command_rejects_unknown_config_target(tmp_path: Path) -> None:
    """Unknown preset names and missing config files fail clearly."""
    output_path = tmp_path / "population.hdf5"

    result = _RUNNER.invoke(
        app,
        ["simulate", "--config", "does-not-exist", "--n", "10", "--output", str(output_path)],
    )

    assert result.exit_code == 1
    assert "Unknown preset or configuration path" in result.output


class TestSetupLogging:
    """Smoke tests for logging setup."""

    def test_setup_logging_with_info_level(self) -> None:
        """INFO setup does not raise."""
        setup_logging(LoggingLevel.INFO)
        assert True

    def test_setup_logging_with_debug_level(self) -> None:
        """DEBUG setup does not raise."""
        setup_logging(LoggingLevel.DEBUG)
        assert True

    def test_setup_logging_with_error_level(self) -> None:
        """ERROR setup does not raise."""
        setup_logging(LoggingLevel.ERROR)
        assert True
