"""Convert named-column population catalogues between CSV and HDF5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from gwmock_pop.constants import CBC_PARAMETER_NAMES
from gwmock_pop.loaders.file_loader import (
    apply_population_column_map,
    infer_population_file_format,
    read_population_catalogue,
    write_population_catalogue,
)
from gwmock_pop.utils.yaml import read_yaml


def _load_column_map(path: Path) -> dict[str, str]:
    """Load a JSON or YAML source-to-canonical column mapping from disk."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        data = read_yaml(path)
    else:
        raise ValueError(f"Unsupported column-map format for {path}. Supported suffixes: .json, .yaml, .yml.")

    if not isinstance(data, dict):
        raise ValueError(f"Column map {path} must contain a top-level mapping.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in data.items()):
        raise ValueError(f"Column map {path} must map string source columns to string canonical names.")
    return data


def _canonical_parameter_names() -> set[str]:
    """Return the canonical gwmock-pop parameter names accepted by the converter."""
    return set(CBC_PARAMETER_NAMES)


def _validate_column_map_targets(column_map: dict[str, str], canonical_names: set[str]) -> None:
    """Reject column maps that point at unknown canonical target names."""
    invalid_targets = sorted({target for target in column_map.values() if target not in canonical_names})
    if invalid_targets:
        raise ValueError(
            f"Column map targets must be canonical gwmock-pop parameter names. Invalid targets: {invalid_targets}"
        )


def _filter_canonical_columns(
    raw_catalogue: dict[str, np.ndarray], column_map: dict[str, str], canonical_names: set[str]
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], list[str]]:
    """Keep only canonical columns after applying the optional renaming map."""
    renamed_catalogue = apply_population_column_map(raw_catalogue, column_map)

    converted: dict[str, np.ndarray] = {}
    renamed_columns: list[tuple[str, str]] = []
    skipped_columns: list[str] = []

    reverse_map = {v: k for k, v in column_map.items()}
    for target_name, values in renamed_catalogue.items():
        source_name = reverse_map.get(target_name, target_name)
        if target_name != source_name:
            renamed_columns.append((source_name, target_name))
        if target_name not in canonical_names:
            skipped_columns.append(source_name)
            continue
        converted[target_name] = values

    return converted, renamed_columns, skipped_columns


def _render_conversion_summary(
    output_path: Path,
    population: dict[str, np.ndarray],
    renamed_columns: list[tuple[str, str]],
    skipped_columns: list[str],
) -> str:
    """Render a human-readable conversion summary."""
    row_count = 0 if not population else len(next(iter(population.values())))
    lines = [
        f"Converted {row_count} rows to {output_path}.",
        "Retained columns: " + (", ".join(population) if population else "none"),
    ]
    if renamed_columns:
        lines.append("Renamed columns:")
        lines.extend(f"  {source} -> {target}" for source, target in renamed_columns)
    else:
        lines.append("Renamed columns: none")
    if skipped_columns:
        lines.append("Skipped columns: " + ", ".join(skipped_columns))
    else:
        lines.append("Skipped columns: none")
    return "\n".join(lines)


def convert_command(
    input_path_arg: Annotated[Path, typer.Option("--input", help="Source .csv, .h5, or .hdf5 population file.")],
    output: Annotated[Path, typer.Option("--output", help="Destination .csv, .h5, or .hdf5 population file.")],
    column_map: Annotated[
        Path | None,
        typer.Option("--column-map", help="Optional JSON/YAML mapping from source columns to canonical names."),
    ] = None,
) -> None:
    """Convert a named-column population file between CSV and HDF5 formats."""
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        input_path = input_path_arg.expanduser()
        output_path = output.expanduser()
        infer_population_file_format(input_path)
        infer_population_file_format(output_path)
        loaded_column_map = {} if column_map is None else _load_column_map(column_map.expanduser())
        canonical_names = _canonical_parameter_names()
        _validate_column_map_targets(loaded_column_map, canonical_names)
        raw_catalogue = read_population_catalogue(input_path)
        converted, renamed_columns, skipped_columns = _filter_canonical_columns(
            raw_catalogue=raw_catalogue,
            column_map=loaded_column_map,
            canonical_names=canonical_names,
        )
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    if not converted:
        logger.error("No canonical gwmock-pop columns remain after applying the optional column map.")
        raise typer.Exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.error("Refusing to overwrite existing output file %s.", output_path)
        raise typer.Exit(1)

    try:
        write_population_catalogue(output_path=output_path, population=converted)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    typer.echo(
        _render_conversion_summary(
            output_path=output_path,
            population=converted,
            renamed_columns=renamed_columns,
            skipped_columns=skipped_columns,
        )
    )
