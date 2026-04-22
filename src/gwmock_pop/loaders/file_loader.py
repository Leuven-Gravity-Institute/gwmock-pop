"""Concrete external population loader for file-backed catalogues."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


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
        hdf5_dataset: str = "data",
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

        catalogue = self._read_catalogue(self._path, hdf5_dataset=hdf5_dataset)
        self._catalogue = self._apply_column_map(catalogue, column_map or {})

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
    def _read_catalogue(path: Path, *, hdf5_dataset: str) -> dict[str, Array]:
        """Read a named-column catalogue from an HDF5 or CSV file."""
        suffix = path.suffix.lower()
        if suffix in {".hdf5", ".h5"}:
            return FilePopulationLoader._read_hdf5_catalogue(path, dataset_name=hdf5_dataset)
        if suffix == ".csv":
            return FilePopulationLoader._read_csv_catalogue(path)

        raise ValueError("Unsupported file format. Supported formats are: .hdf5, .h5, .csv.")

    @staticmethod
    def _read_hdf5_catalogue(path: Path, *, dataset_name: str) -> dict[str, Array]:
        """Read a structured HDF5 dataset into a catalogue mapping."""
        with h5py.File(path, "r") as handle:
            dataset = handle[dataset_name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"HDF5 object '{dataset_name}' is not a dataset.")
            data = np.atleast_1d(dataset[()])

        return FilePopulationLoader._structured_array_to_catalogue(data, file_format="HDF5")

    @staticmethod
    def _read_csv_catalogue(path: Path) -> dict[str, Array]:
        """Read a header-based CSV file into a catalogue mapping."""
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        data = np.atleast_1d(data)
        return FilePopulationLoader._structured_array_to_catalogue(data, file_format="CSV")

    @staticmethod
    def _structured_array_to_catalogue(data: np.ndarray, *, file_format: str) -> dict[str, Array]:
        """Convert a structured array with named columns into JAX arrays."""
        column_names = data.dtype.names
        if column_names is None:
            raise ValueError(f"{file_format} catalogue must provide named columns.")

        catalogue = {name: jnp.asarray(np.atleast_1d(data[name])) for name in column_names}
        if any(values.ndim != 1 for values in catalogue.values()):
            raise ValueError(f"{file_format} catalogue columns must be 1-D arrays.")
        return catalogue

    @staticmethod
    def _apply_column_map(catalogue: dict[str, Array], column_map: dict[str, str]) -> dict[str, Array]:
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
