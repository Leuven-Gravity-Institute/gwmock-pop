"""Concrete external population loader for local or cached file-backed catalogues."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gwmock_data import hdf5

from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.loaders._fetch import resolve_population_path
from gwmock_pop.loaders._validate import validate_population_catalogue
from gwmock_pop.provenance import (
    EngineDescription,
    external_engine_origin,
    provenance_sidecar_path,
    read_provenance,
    write_hdf5_provenance,
    write_provenance_sidecar,
)

if TYPE_CHECKING:
    from jax import Array

_HDF5_DATASET_NAME = "data"
_SUPPORTED_CATALOGUE_SUFFIXES = {".csv": "csv", ".hdf5": "hdf5", ".h5": "hdf5"}


def supported_population_formats() -> frozenset[str]:
    """Return the catalogue formats this package reads and writes.

    Returns:
        The format names, as reported by :func:`infer_population_file_format`.
    """
    return frozenset(_SUPPORTED_CATALOGUE_SUFFIXES.values())


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
    *,
    output_path: Path,
    population: Mapping[str, Array | np.ndarray],
    columns: Sequence[str],
    n_rows: int,
) -> None:
    """Persist a population as a header-based CSV file; its record, if any, is the caller's to write."""
    matrix = _population_matrix(population=population, columns=columns, n_rows=n_rows)
    np.savetxt(output_path, matrix, delimiter=",", header=",".join(columns), comments="")


def _write_population_hdf5(  # noqa: PLR0913  # one writer, one place: every knob the format has
    *,
    output_path: Path,
    population: Mapping[str, Array | np.ndarray],
    columns: Sequence[str],
    n_rows: int,
    provenance: Mapping[str, Any] | None = None,
    compression: str | None = None,
) -> None:
    """Persist a population as a structured HDF5 dataset named ``data``.

    An optional provenance record is embedded in a sibling ``metadata`` group,
    which the catalogue readers ignore.
    """
    structured = _population_structured_array(population=population, columns=columns, n_rows=n_rows)
    # A chunked dataset cannot have a zero-length chunk, so an empty catalogue is
    # written uncompressed rather than refused.
    dataset_compression = compression if n_rows > 0 else None
    root = hdf5.Group({_HDF5_DATASET_NAME: hdf5.Dataset(structured)})
    if provenance is not None:
        write_hdf5_provenance(root, dict(provenance))
    hdf5.write(root, output_path, compression=dataset_compression)


def write_population_catalogue(
    output_path: str | os.PathLike,
    population: Mapping[str, Array | np.ndarray],
    *,
    provenance: Mapping[str, Any] | None = None,
    compression: str | None = None,
) -> None:
    """Persist a population mapping using named-column CSV/HDF5 conventions.

    This is the only catalogue writer in the package, so every file it produces
    can carry the same record and none of them can drift into a private layout.

    Args:
        output_path: Destination ``.csv``, ``.h5``, or ``.hdf5`` file.
        population: Mapping from parameter name to a 1-D column. Insertion order
            is the output column order.
        provenance: Record describing how the catalogue was produced, built by
            :func:`gwmock_pop.provenance.build_provenance_record`. HDF5 files
            carry it internally; CSV files get a JSON sidecar next to them.
        compression: Optional HDF5 compression filter, such as ``"gzip"``.
            Ignored for CSV.

    Raises:
        ValueError: If the destination suffix names an unsupported format.
    """
    destination = Path(output_path)
    columns = _population_columns(population)
    n_rows = _population_size(population=population, columns=columns)
    file_format = infer_population_file_format(destination)

    # Written beside the destination and moved over it only once complete, so a
    # write that fails part-way (a refused dtype, an unknown compression filter)
    # leaves the previous catalogue, and the sidecar describing it, as they were.
    # A symlinked destination is written through, as opening it would: the file
    # it points to is replaced, not the link.
    target = destination.resolve() if destination.is_symlink() else destination
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        if file_format == "csv":
            _write_population_csv(output_path=partial, population=population, columns=columns, n_rows=n_rows)
        else:
            _write_population_hdf5(
                output_path=partial,
                population=population,
                columns=columns,
                n_rows=n_rows,
                provenance=provenance,
                compression=compression,
            )
        if target.exists():
            # A rewrite keeps the permissions of the catalogue it replaces.
            shutil.copymode(target, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)

    if provenance is None:
        # A catalogue written without a record must not inherit the record of
        # whatever it replaced. An HDF5 rewrite replaces the embedded record,
        # but a sidecar sits beside the file and would survive -- and
        # read_provenance falls back to it -- so it would resurface as a
        # description of samples that are no longer there.
        provenance_sidecar_path(destination).unlink(missing_ok=True)
    elif file_format == "csv":
        write_provenance_sidecar(destination, dict(provenance))


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

    def __init__(  # noqa: PLR0913
        self,
        source_type: str,
        path: str | os.PathLike,
        *,
        column_map: dict[str, str] | None = None,
        hdf5_dataset: str = _HDF5_DATASET_NAME,
        cache_dir: str | os.PathLike[str] | None = None,
        refresh: bool = False,
        token: str | None = None,
        credentials: Mapping[str, str] | Mapping[str, Mapping[str, str]] | None = None,
        download_timeout: int = 300,
    ) -> None:
        """Load and validate a population catalogue from disk or a remote URL.

        Args:
            source_type: Non-empty routing key used by downstream simulators,
                such as ``"bbh"`` or ``"bns"``.
            path: Path or supported URL to the input catalogue file. Supported
                local and remote formats are ``.hdf5``, ``.h5``, and ``.csv``.
            column_map: Optional mapping from file column names to canonical
                gwmock-pop parameter names. Keys must exist in the loaded
                catalogue.
            hdf5_dataset: Dataset name to read from HDF5 files.
            cache_dir: Optional cache directory for remotely fetched catalogues.
            refresh: Whether to force a re-download for remote URLs even when a
                cached copy already exists.
            token: Optional bearer token used for authenticated remote fetches.
            credentials: Optional mapping of request headers or scheme-specific
                credential fields for remote fetches.
            download_timeout: Timeout in seconds for HTTP(S) downloads.

        Raises:
            ValueError: If ``source_type`` is empty, the file format is
                unsupported, the file contents do not expose named columns, the
                column mapping is invalid, or the resulting catalogue is empty.
        """
        if not source_type:
            raise ValueError("source_type must be a non-empty string.")

        self._source_type = source_type

        fetch_result = resolve_population_path(
            path,
            cache_dir=cache_dir,
            refresh=refresh,
            token=token,
            credentials=credentials,
            timeout=download_timeout,
        )
        self._path = fetch_result.path

        try:
            raw_catalogue = read_population_catalogue(self._path, hdf5_dataset=hdf5_dataset)
            mapped_catalogue = apply_population_column_map(raw_catalogue, column_map)
            validated_catalogue = validate_population_catalogue(source_type, mapped_catalogue)
        except ValueError as exc:
            raise PopulationValidationError(str(exc)) from exc

        import jax.numpy as jnp  # noqa: PLC0415  # deferred JAX import

        self._catalogue = {name: jnp.asarray(values) for name, values in validated_catalogue.items()}
        if not self._catalogue:
            raise PopulationValidationError("Loaded catalogue is empty.")

        self._parameter_names = sorted(self._catalogue)
        self._metadata = {
            "fetch": fetch_result.metadata,
            "original_path": os.fspath(path),
            "resolved_path": str(self._path),
            "source_type": source_type,
        }

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

    @property
    def metadata(self) -> dict[str, Any]:
        """Return loader metadata, including remote cache details when relevant."""
        return self._metadata

    @property
    def provenance(self) -> dict[str, Any] | None:
        """Return the provenance record the loaded catalogue carries, if any.

        Returns:
            The record found in or beside the input file, or ``None`` when the
            catalogue arrived undocumented.
        """
        return read_provenance(self._path)

    def provenance_origin(self, engine: EngineDescription) -> dict[str, Any]:
        """Describe this loaded catalogue as the origin of a new one.

        The fetch details the loader already recorded for a remote catalogue --
        URL, cache key and ETag -- are folded in rather than restated, and the
        record the input file carried is chained rather than dropped.

        Args:
            engine: Identity of the engine that produced the input catalogue,
                including a pointer to that engine's own run record.

        Returns:
            An origin block for :func:`gwmock_pop.provenance.build_provenance_record`.
        """
        return external_engine_origin(
            engine=engine,
            input_path=self._metadata["original_path"],
            fetch=self._metadata["fetch"],
            upstream=self.provenance,
        )

    def simulate(self, n_samples: int | None = None, **kwargs: Any) -> Mapping[str, Array]:
        """Sample catalogue rows without replacement.

        Args:
            n_samples: Number of catalogue rows to draw.  Pass ``None`` to
                return all rows in the catalogue.
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
        if n_samples is None:
            n_samples = catalogue_size
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
        if n_samples > catalogue_size:
            raise ValueError(f"Requested {n_samples} samples from a catalogue with only {catalogue_size} rows.")

        import jax  # noqa: PLC0415  # deferred JAX import

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
        """Read an HDF5 catalogue from a compound dataset or group-of-datasets.

        ``dataset_name`` may be a path into groups, such as ``"run1/data"``; as in
        HDF5, empty and ``.`` components are ignored, so ``"/"`` names the file itself.
        """
        parts = [part for part in dataset_name.split("/") if part not in ("", ".")]
        # Only the top-level member on the path is read, not the rest of the file.
        hdf5_object: hdf5.Group | hdf5.Dataset | None = hdf5.read(path, members=parts[:1] or None)
        for part in parts:
            hdf5_object = hdf5_object.members.get(part) if isinstance(hdf5_object, hdf5.Group) else None
        if hdf5_object is None:
            raise KeyError(f"HDF5 catalogue {path} has no object '{dataset_name}'.")
        if isinstance(hdf5_object, hdf5.Dataset):
            data = np.atleast_1d(hdf5_object.data)
            return FilePopulationLoader._structured_array_to_catalogue(data, file_format="HDF5")
        group_catalogue: dict[str, np.ndarray] = {}
        expected_length: int | None = None
        for child_name, child_object in hdf5_object.members.items():
            if not isinstance(child_object, hdf5.Dataset):
                raise TypeError(f"HDF5 group '{dataset_name}' contains non-dataset member '{child_name}'.")
            column = np.atleast_1d(child_object.data)
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
