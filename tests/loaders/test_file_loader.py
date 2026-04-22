"""Integration tests for ``FilePopulationLoader``."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gwmock_pop import FilePopulationLoader
from gwmock_pop.protocols import GWPopSimulator

pytestmark = pytest.mark.integration


def _catalogue_columns(n_rows: int = 200) -> dict[str, np.ndarray]:
    """Create a deterministic synthetic catalogue."""
    return {
        "mass_1": np.linspace(30.0, 60.0, n_rows),
        "mass_2": np.linspace(20.0, 40.0, n_rows),
        "redshift": np.linspace(0.05, 1.0, n_rows),
        "distance": np.linspace(100.0, 1000.0, n_rows),
        "spin_1z": np.linspace(-0.9, 0.9, n_rows),
    }


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


def test_hdf5_round_trip(tmp_path: Path) -> None:
    """Load an HDF5 catalogue and sample rows through the protocol surface."""
    path = tmp_path / "catalogue.hdf5"
    columns = _catalogue_columns()
    _write_hdf5_catalogue(path, columns)

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(50, seed=123)

    assert isinstance(loader, GWPopSimulator)
    assert list(result.keys()) == loader.parameter_names
    assert set(result) == set(columns)
    assert all(array.shape == (50,) for array in result.values())


def test_csv_round_trip(tmp_path: Path) -> None:
    """Load a CSV catalogue and sample rows through the protocol surface."""
    path = tmp_path / "catalogue.csv"
    columns = _catalogue_columns()
    _write_csv_catalogue(path, columns)

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(50, seed=123)

    assert isinstance(loader, GWPopSimulator)
    assert list(result.keys()) == loader.parameter_names
    assert set(result) == set(columns)
    assert all(array.shape == (50,) for array in result.values())


def test_column_map_renames_keys(tmp_path: Path) -> None:
    """Apply a column mapping and expose only canonical names."""
    path = tmp_path / "catalogue.csv"
    columns = {
        "m1": np.linspace(30.0, 60.0, 200),
        "m2": np.linspace(20.0, 40.0, 200),
        "z": np.linspace(0.05, 1.0, 200),
        "distance": np.linspace(100.0, 1000.0, 200),
        "spin1z": np.linspace(-0.9, 0.9, 200),
    }
    _write_csv_catalogue(path, columns)

    loader = FilePopulationLoader(
        "bbh",
        path,
        column_map={"m1": "mass_1", "m2": "mass_2", "z": "redshift", "spin1z": "spin_1z"},
    )
    result = loader.simulate(25, seed=9)

    assert "m1" not in loader.parameter_names
    assert "m2" not in loader.parameter_names
    assert "z" not in loader.parameter_names
    assert "spin1z" not in loader.parameter_names
    assert "mass_1" in loader.parameter_names
    assert "mass_2" in loader.parameter_names
    assert "redshift" in loader.parameter_names
    assert "spin_1z" in loader.parameter_names
    assert list(result.keys()) == loader.parameter_names


def test_simulate_exceeds_catalogue_raises(tmp_path: Path) -> None:
    """Reject sample requests larger than the catalogue."""
    path = tmp_path / "catalogue.csv"
    columns = _catalogue_columns(n_rows=20)
    _write_csv_catalogue(path, columns)

    loader = FilePopulationLoader("bbh", path)

    with pytest.raises(ValueError, match="Requested 21 samples"):
        loader.simulate(21)


def test_unsupported_format_raises(tmp_path: Path) -> None:
    """Reject unsupported file extensions at construction time."""
    path = tmp_path / "catalogue.fits"
    path.write_text("not-a-supported-catalogue\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Supported formats are: \.hdf5, \.h5, \.csv"):
        FilePopulationLoader("bbh", path)
