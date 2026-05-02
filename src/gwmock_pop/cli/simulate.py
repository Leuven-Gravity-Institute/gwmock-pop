"""Simulate populations from packaged presets or graph config files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer
from jax import Array

from gwmock_pop.cli.common import resolve_simulator

_HDF5_DATASET_NAME = "data"
_SUPPORTED_OUTPUT_SUFFIXES = {".csv": "csv", ".hdf5": "hdf5", ".h5": "hdf5"}


def _infer_output_format(output_path: Path) -> str:
    """Infer the persistence format from the output file suffix."""
    file_format = _SUPPORTED_OUTPUT_SUFFIXES.get(output_path.suffix.lower())
    if file_format is None:
        supported = ", ".join(sorted(_SUPPORTED_OUTPUT_SUFFIXES))
        raise ValueError(f"Unsupported output format for {output_path}. Supported suffixes: {supported}.")
    return file_format


def _population_columns(population: Mapping[str, Array]) -> list[str]:
    """Return the stable output column ordering for a sampled population."""
    return list(population)


def _population_size(population: Mapping[str, Array], columns: Sequence[str]) -> int:
    """Return the number of sampled rows in the population."""
    if not columns:
        return 0
    return int(np.asarray(population[columns[0]]).shape[0])


def _population_matrix(population: Mapping[str, Array], columns: Sequence[str], n_rows: int) -> np.ndarray:
    """Materialize the sampled population as a dense 2-D matrix."""
    matrix = np.empty((n_rows, len(columns)), dtype=float)
    for column_index, column_name in enumerate(columns):
        matrix[:, column_index] = np.asarray(population[column_name], dtype=float)
    return matrix


def _population_structured_array(population: Mapping[str, Array], columns: Sequence[str], n_rows: int) -> np.ndarray:
    """Materialize the sampled population as a structured array with named columns."""
    dtype = [(column_name, np.asarray(population[column_name]).dtype) for column_name in columns]
    structured = np.empty(n_rows, dtype=dtype)
    for column_name in columns:
        structured[column_name] = np.asarray(population[column_name])
    return structured


def _write_population_csv(
    output_path: Path, population: Mapping[str, Array], columns: Sequence[str], n_rows: int
) -> None:
    """Persist a population as a header-based CSV file."""
    matrix = _population_matrix(population=population, columns=columns, n_rows=n_rows)
    np.savetxt(output_path, matrix, delimiter=",", header=",".join(columns), comments="")


def _write_population_hdf5(
    output_path: Path, population: Mapping[str, Array], columns: Sequence[str], n_rows: int
) -> None:
    """Persist a population as a structured HDF5 dataset named ``data``."""
    structured = _population_structured_array(population=population, columns=columns, n_rows=n_rows)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(_HDF5_DATASET_NAME, data=structured)


def _write_population(output_path: Path, population: Mapping[str, Array]) -> None:
    """Persist a sampled population using named-column conventions."""
    columns = _population_columns(population)
    n_rows = _population_size(population=population, columns=columns)
    file_format = _infer_output_format(output_path)

    if file_format == "csv":
        _write_population_csv(output_path=output_path, population=population, columns=columns, n_rows=n_rows)
        return
    if file_format == "hdf5":
        _write_population_hdf5(output_path=output_path, population=population, columns=columns, n_rows=n_rows)
        return

    raise ValueError(f"Unsupported output format {file_format!r}.")


def simulate_command(
    config: Annotated[str, typer.Option("--config", help="Preset name or YAML/TOML config-file path.")],
    n: Annotated[int, typer.Option("--n", min=0, help="Number of events to sample.")],
    output: Annotated[Path, typer.Option("--output", help="Destination .csv, .h5, or .hdf5 file.")],
    seed: Annotated[int | None, typer.Option("--seed", help="Optional random seed.")] = None,
) -> None:
    """Simulate a population and persist it as a named-column catalogue."""
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        simulator = resolve_simulator(config=config, seed=seed)
        output_path = output.expanduser()
        _infer_output_format(output_path)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.error("Refusing to overwrite existing output file %s.", output_path)
        raise typer.Exit(1)

    try:
        population = simulator.simulate(n)
        _write_population(output_path=output_path, population=population)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    logger.info("Saved %s samples to %s", n, output_path)
