"""Integration tests for ``FilePopulationLoader``."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gwmock_pop import FilePopulationLoader, PopulationValidationError
from gwmock_pop.protocols import GWPopSimulator

pytestmark = pytest.mark.integration
_EXPECTED_CBC_COLUMNS = {
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "distance",
    "redshift",
    "spin_1z",
}


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


def _write_hdf5_group_catalogue(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write an HDF5 group where each column is its own dataset."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        for name, values in columns.items():
            group.create_dataset(name, data=values)


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
    assert set(result) == _EXPECTED_CBC_COLUMNS
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
    assert set(result) == _EXPECTED_CBC_COLUMNS
    assert all(array.shape == (50,) for array in result.values())


def test_hdf5_group_round_trip(tmp_path: Path) -> None:
    """Load an HDF5 group-of-datasets catalogue and sample rows."""
    path = tmp_path / "catalogue_group.hdf5"
    columns = _catalogue_columns()
    _write_hdf5_group_catalogue(path, columns)

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(40, seed=13)

    assert isinstance(loader, GWPopSimulator)
    assert list(result.keys()) == loader.parameter_names
    assert set(result) == _EXPECTED_CBC_COLUMNS
    assert all(array.shape == (40,) for array in result.values())


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
        column_map={
            "m1": "detector_frame_mass_1",
            "m2": "detector_frame_mass_2",
            "z": "redshift",
            "spin1z": "spin_1z",
        },
    )
    result = loader.simulate(25, seed=9)

    assert "m1" not in loader.parameter_names
    assert "m2" not in loader.parameter_names
    assert "z" not in loader.parameter_names
    assert "spin1z" not in loader.parameter_names
    assert "detector_frame_mass_1" in loader.parameter_names
    assert "detector_frame_mass_2" in loader.parameter_names
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


def test_simulate_negative_n_samples_raises(tmp_path: Path) -> None:
    """Reject negative sample counts."""
    path = tmp_path / "catalogue.csv"
    columns = _catalogue_columns(n_rows=20)
    _write_csv_catalogue(path, columns)

    loader = FilePopulationLoader("bbh", path)

    with pytest.raises(ValueError, match=r"n_samples must be >= 0"):
        loader.simulate(-1)


def test_simulate_accepts_backend_agnostic_kwargs(tmp_path: Path) -> None:
    """Accept alternative random-state kwargs without rejecting the call."""
    path = tmp_path / "catalogue.csv"
    columns = _catalogue_columns()
    _write_csv_catalogue(path, columns)

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(10, rng=42, unknown_backend_kwarg=True)

    assert list(result.keys()) == loader.parameter_names
    assert all(array.shape == (10,) for array in result.values())


def test_hdf5_group_mismatched_column_lengths_raise(tmp_path: Path) -> None:
    """Reject group-of-datasets catalogues with inconsistent column lengths."""
    path = tmp_path / "catalogue_bad_group.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("mass_1", data=np.linspace(30.0, 60.0, 20))
        group.create_dataset("mass_2", data=np.linspace(20.0, 40.0, 19))

    with pytest.raises(ValueError, match="mismatched column lengths"):
        FilePopulationLoader("bbh", path)


def test_unsupported_format_raises(tmp_path: Path) -> None:
    """Reject unsupported file extensions at construction time."""
    path = tmp_path / "catalogue.fits"
    path.write_text("not-a-supported-catalogue\n", encoding="utf-8")

    with pytest.raises(PopulationValidationError, match=r"Supported suffixes: \.csv, \.h5, \.hdf5"):
        FilePopulationLoader("bbh", path)


def test_simulate_none_returns_all_rows(tmp_path: Path) -> None:
    """simulate(n_samples=None) returns arrays whose length equals catalogue size."""
    n_rows = 15
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, _catalogue_columns(n_rows=n_rows))

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(n_samples=None)

    assert all(len(array) == n_rows for array in result.values())


def test_simulate_none_returns_all_columns(tmp_path: Path) -> None:
    """simulate(n_samples=None) returns all parameter_names keys."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, _catalogue_columns(n_rows=10))

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(n_samples=None)

    assert set(result.keys()) == set(loader.parameter_names)


def test_simulate_zero_returns_empty(tmp_path: Path) -> None:
    """simulate(n_samples=0) returns empty arrays without raising."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, _catalogue_columns(n_rows=10))

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(n_samples=0)

    assert all(len(array) == 0 for array in result.values())


def test_simulate_negative_still_raises(tmp_path: Path) -> None:
    """simulate(n_samples=-1) still raises ValueError after the signature change."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, _catalogue_columns(n_rows=10))

    loader = FilePopulationLoader("bbh", path)

    with pytest.raises(ValueError, match=r"n_samples must be >= 0"):
        loader.simulate(n_samples=-1)
