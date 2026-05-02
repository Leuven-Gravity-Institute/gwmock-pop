"""Concrete external population loader for file-backed catalogues."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_HDF5_DATASET_NAME = "data"
_SUPPORTED_CATALOGUE_SUFFIXES = {".csv": "csv", ".hdf5": "hdf5", ".h5": "hdf5"}


def infer_population_file_format(path: str | os.PathLike) -> str:
    """Infer the file format for a population catalogue path."""
    output_path = Path(path)
    file_format = _SUPPORTED_CATALOGUE_SUFFIXES.get(output_path.suffix.lower())
    if file_format is None:
        supported = ", ".join(sorted(_SUPPORTED_CATALOGUE_SUFFIXES))
        raise ValueError(f"Unsupported population-file format for {output_path}. Supported suffixes: {supported}.")
    return file_format


def apply_population_column_map(
    catalogue: Mapping[str, np.ndarray], column_map: Mapping[str, str] | None = None
) -> dict[str, np.ndarray]:
    """Apply a deterministic column-renaming map to a catalogue mapping."""
    return FilePopulationLoader._apply_column_map(dict(catalogue), dict(column_map or {}))


def read_population_catalogue(
    path: str | os.PathLike,
    *,
    column_map: Mapping[str, str] | None = None,
    hdf5_dataset: str = _HDF5_DATASET_NAME,
) -> dict[str, np.ndarray]:
    """Read a named-column population catalogue from disk."""
    catalogue = FilePopulationLoader._read_catalogue(Path(path), hdf5_dataset=hdf5_dataset)
    return apply_population_column_map(catalogue, column_map)


def _population_columns(population: Mapping[str, Array | np.ndarray]) -> list[str]:
    """Return the stable output column ordering for a population mapping."""
    return list(population)


def _population_size(population: Mapping[str, Array | np.ndarray], columns: Sequence[str]) -> int:
    """Return the number of rows in a population mapping."""
    if not columns:
        return 0
    return int(np.asarray(population[columns[0]]).shape[0])


def _population_matrix(population: Mapping[str, Array | np.ndarray], columns: Sequence[str], n_rows: int) -> np.ndarray:
    """Materialize a population mapping as a dense 2-D matrix."""
    matrix = np.empty((n_rows, len(columns)), dtype=float)
    for column_index, column_name in enumerate(columns):
        matrix[:, column_index] = np.asarray(population[column_name], dtype=float)
    return matrix


def _population_structured_array(
    population: Mapping[str, Array | np.ndarray], columns: Sequence[str], n_rows: int
) -> np.ndarray:
    """Materialize a population mapping as a structured array with named columns."""
    dtype = [(column_name, np.asarray(population[column_name]).dtype) for column_name in columns]
    structured = np.empty(n_rows, dtype=dtype)
    for column_name in columns:
        structured[column_name] = np.asarray(population[column_name])
    return structured


def _write_population_csv(
    output_path: Path, population: Mapping[str, Array | np.ndarray], columns: Sequence[str], n_rows: int
) -> None:
    """Persist a population as a header-based CSV file."""
    matrix = _population_matrix(population=population, columns=columns, n_rows=n_rows)
    np.savetxt(output_path, matrix, delimiter=",", header=",".join(columns), comments="")


def _write_population_hdf5(
    output_path: Path, population: Mapping[str, Array | np.ndarray], columns: Sequence[str], n_rows: int
) -> None:
    """Persist a population as a structured HDF5 dataset named ``data``."""
    structured = _population_structured_array(population=population, columns=columns, n_rows=n_rows)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(_HDF5_DATASET_NAME, data=structured)


def write_population_catalogue(output_path: str | os.PathLike, population: Mapping[str, Array | np.ndarray]) -> None:
    """Persist a population mapping using named-column CSV/HDF5 conventions."""
    destination = Path(output_path)
    columns = _population_columns(population)
    n_rows = _population_size(population=population, columns=columns)
    file_format = infer_population_file_format(destination)

    if file_format == "csv":
        _write_population_csv(output_path=destination, population=population, columns=columns, n_rows=n_rows)
        return
    if file_format == "hdf5":
        _write_population_hdf5(output_path=destination, population=population, columns=columns, n_rows=n_rows)
        return

    raise ValueError(f"Unsupported format {file_format!r}.")


class FilePopulationLoader:
    """Load an external population catalogue from HDF5 or CSV.

    The loader reads the input file eagerly during construction, normalizes the
    catalogue into a mapping of 1-D ``jax.Array`` columns, and then provides a
    :meth:`simulate` method that samples catalogue rows without replacement.
    This keeps the runtime surface compatible with
    :class:`gwmock_pop.protocols.ExternalPopulationLoader` and
    :class:`gwmock_pop.protocols.GWPopSimulator` without introducing a shared
    inheritance hierarchy.
    """

    def __init__(
        self,
        source_type: str,
        path: str | os.PathLike,
        *,
        column_map: dict[str, str] | None = None,
        hdf5_dataset: str = _HDF5_DATASET_NAME,
    ) -> None:
        """Load and validate a population catalogue from disk.

        Args:
            source_type: Non-empty routing key used by downstream simulators,
                such as ``"bbh"`` or ``"bns"``.
            path: Path to the input catalogue file. Supported formats are
                ``.hdf5``, ``.h5``, and ``.csv``.
            column_map: Optional mapping from file column names to canonical
                gwmock-pop parameter names. Keys must exist in the loaded
                catalogue.
            hdf5_dataset: Dataset name to read from HDF5 files.

        Raises:
            ValueError: If ``source_type`` is empty, the file format is
                unsupported, the file contents do not expose named columns, the
                column mapping is invalid, or the resulting catalogue is empty.
        """
        if not source_type:
            raise ValueError("source_type must be a non-empty string.")

        self._source_type = source_type
        self._path = Path(path)

        raw_catalogue = read_population_catalogue(self._path, column_map=column_map or {}, hdf5_dataset=hdf5_dataset)
        self._catalogue = {name: jnp.asarray(values) for name, values in raw_catalogue.items()}

        if not self._catalogue:
            raise ValueError("Loaded catalogue is empty.")

        self._parameter_names = sorted(self._catalogue)

    @property
    def parameter_names(self) -> list[str]:
        """Return the stable ordered parameter names exposed by this loader.

        Returns:
            Sorted list of parameter names available in the loaded catalogue.
        """
        return self._parameter_names

    @property
    def source_type(self) -> str:
        """Return the routing key associated with this population catalogue.

        Returns:
            Source type passed at construction time.
        """
        return self._source_type

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
        """Sample catalogue rows without replacement.

        Args:
            n_samples: Number of catalogue rows to draw.
            **kwargs: Optional backend-agnostic random-state hints. Supported
                keys include ``seed``, ``key``, and ``rng`` (in that priority).

        Returns:
            Mapping from parameter name to a 1-D ``jax.Array`` of sampled
            values.

        Raises:
            ValueError: If ``n_samples`` is negative or exceeds the number of
                rows in the loaded catalogue.
        """
        catalogue_size = len(next(iter(self._catalogue.values())))
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
        if n_samples > catalogue_size:
            raise ValueError(f"Requested {n_samples} samples from a catalogue with only {catalogue_size} rows.")

        seed = kwargs.get("seed", kwargs.get("key", kwargs.get("rng")))
        key = jax.random.PRNGKey(0 if seed is None else int(seed))
        indices = jax.random.choice(key, catalogue_size, shape=(n_samples,), replace=False)
        return {name: self._catalogue[name][indices] for name in self._parameter_names}

    @staticmethod
    def _read_catalogue(path: Path, *, hdf5_dataset: str) -> dict[str, np.ndarray]:
        """Read a named-column catalogue from an HDF5 or CSV file."""
        file_format = infer_population_file_format(path)
        if file_format == "hdf5":
            return FilePopulationLoader._read_hdf5_catalogue(path, dataset_name=hdf5_dataset)
        if file_format == "csv":
            return FilePopulationLoader._read_csv_catalogue(path)

        raise ValueError(f"Unsupported format {file_format!r}.")

    @staticmethod
    def _read_hdf5_catalogue(path: Path, *, dataset_name: str) -> dict[str, np.ndarray]:
        """Read an HDF5 catalogue from a compound dataset or group-of-datasets."""
        with h5py.File(path, "r") as handle:
            hdf5_object = handle[dataset_name]
            if isinstance(hdf5_object, h5py.Dataset):
                data = np.atleast_1d(hdf5_object[()])
                return FilePopulationLoader._structured_array_to_catalogue(data, file_format="HDF5")
            if isinstance(hdf5_object, h5py.Group):
                group_catalogue: dict[str, np.ndarray] = {}
                expected_length: int | None = None
                for child_name, child_object in hdf5_object.items():
                    if not isinstance(child_object, h5py.Dataset):
                        raise ValueError(f"HDF5 group '{dataset_name}' contains non-dataset member '{child_name}'.")
                    column = np.atleast_1d(child_object[()])
                    if column.ndim != 1:
                        raise ValueError(f"HDF5 group dataset '{dataset_name}/{child_name}' must be a 1-D array.")
                    current_length = len(column)
                    if expected_length is None:
                        expected_length = current_length
                    elif current_length != expected_length:
                        raise ValueError(
                            f"HDF5 group '{dataset_name}' has mismatched column lengths: "
                            f"'{child_name}' has {current_length}, expected {expected_length}."
                        )
                    group_catalogue[child_name] = column
                return FilePopulationLoader._structured_array_to_catalogue(group_catalogue, file_format="HDF5")
            raise ValueError(f"HDF5 object '{dataset_name}' must be a dataset or a group of datasets.")

    @staticmethod
    def _read_csv_catalogue(path: Path) -> dict[str, np.ndarray]:
        """Read a header-based CSV file into a catalogue mapping."""
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        data = np.atleast_1d(data)
        return FilePopulationLoader._structured_array_to_catalogue(data, file_format="CSV")

    @staticmethod
    def _structured_array_to_catalogue(
        data: np.ndarray | Mapping[str, np.ndarray], *, file_format: str
    ) -> dict[str, np.ndarray]:
        """Convert named-column data into a catalogue mapping of 1-D numpy arrays."""
        if isinstance(data, Mapping):
            catalogue = {name: np.atleast_1d(np.asarray(values)) for name, values in data.items()}
        else:
            column_names = data.dtype.names
            if column_names is None:
                raise ValueError(f"{file_format} catalogue must provide named columns.")
            catalogue = {name: np.atleast_1d(np.asarray(data[name])) for name in column_names}

        if not catalogue:
            raise ValueError(f"{file_format} catalogue is empty.")
        if any(values.ndim != 1 for values in catalogue.values()):
            raise ValueError(f"{file_format} catalogue columns must be 1-D arrays.")
        return catalogue

    @staticmethod
    def _apply_column_map(catalogue: dict[str, np.ndarray], column_map: dict[str, str]) -> dict[str, np.ndarray]:
        """Apply a deterministic column renaming map to a catalogue mapping."""
        unknown_columns = sorted(set(column_map) - set(catalogue))
        if unknown_columns:
            raise ValueError(f"Column map references unknown columns: {unknown_columns}")

        remapped_catalogue: dict[str, Array] = {}
        for source_name, values in catalogue.items():
            target_name = column_map.get(source_name, source_name)
            if target_name in remapped_catalogue:
                raise ValueError(f"Column mapping produces duplicate output column '{target_name}'.")
            remapped_catalogue[target_name] = values
        return remapped_catalogue
