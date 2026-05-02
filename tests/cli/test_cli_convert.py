"""Tests for the CLI convert command."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from typer.testing import CliRunner

from gwmock_pop.cli.main import app

_RUNNER = CliRunner()


def _write_hdf5_catalogue(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a structured HDF5 dataset with named columns."""
    dtype = [(name, np.float64) for name in columns]
    structured = np.zeros(len(next(iter(columns.values()))), dtype=dtype)
    for name, values in columns.items():
        structured[name] = values

    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=structured)


def _write_csv_catalogue(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a header-based CSV file with named columns."""
    matrix = np.column_stack([columns[name] for name in columns])
    np.savetxt(path, matrix, delimiter=",", header=",".join(columns), comments="")


def test_convert_command_converts_hdf5_to_csv_losslessly(tmp_path: Path) -> None:
    """The convert CLI preserves canonical floating-point columns across HDF5 -> CSV."""
    input_path = tmp_path / "population.hdf5"
    output_path = tmp_path / "population.csv"
    columns = {
        "detector_frame_mass_1": np.linspace(10.0, 20.0, 8),
        "detector_frame_mass_2": np.linspace(5.0, 15.0, 8),
        "redshift": np.linspace(0.01, 0.2, 8),
    }
    _write_hdf5_catalogue(input_path, columns)

    result = _RUNNER.invoke(app, ["convert", "--input", str(input_path), "--output", str(output_path)])

    assert result.exit_code == 0, result.output
    assert "Renamed columns: none" in result.output
    assert "Skipped columns: none" in result.output

    reloaded = np.genfromtxt(output_path, delimiter=",", names=True, dtype=float)
    assert reloaded.dtype.names == tuple(columns)
    for name, values in columns.items():
        np.testing.assert_allclose(np.asarray(reloaded[name]), values, rtol=0.0, atol=0.0)


def test_convert_command_applies_column_map_and_reports_skipped_columns(tmp_path: Path) -> None:
    """The convert CLI remaps external names and drops columns without canonical targets."""
    input_path = tmp_path / "external.csv"
    output_path = tmp_path / "canonical.hdf5"
    map_path = tmp_path / "map.json"
    columns = {
        "m1": np.linspace(30.0, 60.0, 6),
        "m2": np.linspace(20.0, 40.0, 6),
        "z": np.linspace(0.05, 0.5, 6),
        "external_weight": np.linspace(1.0, 2.0, 6),
    }
    _write_csv_catalogue(input_path, columns)
    map_path.write_text(
        json.dumps(
            {
                "m1": "detector_frame_mass_1",
                "m2": "detector_frame_mass_2",
                "z": "redshift",
            }
        ),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        ["convert", "--input", str(input_path), "--output", str(output_path), "--column-map", str(map_path)],
    )

    assert result.exit_code == 0, result.output
    assert "m1 -> detector_frame_mass_1" in result.output
    assert "m2 -> detector_frame_mass_2" in result.output
    assert "z -> redshift" in result.output
    assert "Skipped columns: external_weight" in result.output

    with h5py.File(output_path, "r") as handle:
        dataset = handle["data"]
        assert dataset.dtype.names == ("detector_frame_mass_1", "detector_frame_mass_2", "redshift")
        np.testing.assert_allclose(np.asarray(dataset["detector_frame_mass_1"]), columns["m1"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(dataset["detector_frame_mass_2"]), columns["m2"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(dataset["redshift"]), columns["z"], rtol=0.0, atol=0.0)
