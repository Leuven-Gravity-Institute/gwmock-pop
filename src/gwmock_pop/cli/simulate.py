"""Simulate populations from packaged presets or graph config files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from gwmock_pop.cli.common import ResolvedOutput, ResolvedRun, resolve_output, resolve_run
from gwmock_pop.loaders.file_loader import write_population_catalogue
from gwmock_pop.provenance import run_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array

_WRITER = "gwmock_pop.cli.simulate.simulate_command"


def _build_record(
    run: ResolvedRun,
    output: ResolvedOutput,
    population: Mapping[str, Array],
    n_samples: int,
) -> dict[str, Any]:
    """Describe a simulated catalogue well enough to reconstruct it.

    Args:
        run: The resolved run.
        output: The resolved destination.
        population: The sampled catalogue, whose key order is the column order.
        n_samples: Number of rows written.

    Returns:
        The provenance record to store with the catalogue.
    """
    return run.simulator.build_provenance_record(
        n_samples=n_samples,
        file_format=output.file_format,
        parameter_names=list(population),
        run=run_metadata(name=run.run_name, seed=run.seed, seed_source=run.seed_source),
        writer=_WRITER,
    )


def simulate_command(
    config: Annotated[str, typer.Option("--config", help="Preset name or YAML/TOML config-file path.")],
    output: Annotated[Path, typer.Option("--output", help="Destination .csv, .h5, or .hdf5 file.")],
    n: Annotated[
        int | None,
        typer.Option("--n", min=0, help="Number of events to sample. Defaults to a configured run.n_samples."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed. Defaults to a configured run.seed, otherwise one is drawn."),
    ] = None,
) -> None:
    """Simulate a population and persist it as a named-column catalogue.

    The catalogue is written with a provenance record: the package version and
    checkout state, the complete resolved configuration, the seed as actually
    used, and the column list in output order. HDF5 files carry the record in a
    ``metadata`` group; CSV files get a JSON sidecar. Set
    ``run.output.save_metadata`` to false to write the samples alone.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        run = resolve_run(config=config, seed=seed, n_samples=n)
        if run.n_samples is None:
            raise ValueError("No sample count given. Pass --n or set run.n_samples in the configuration.")
        resolved_output = resolve_output(output, run.configuration)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    output_path = resolved_output.path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not resolved_output.overwrite:
        logger.error("Refusing to overwrite existing output file %s.", output_path)
        raise typer.Exit(1)

    try:
        population = run.simulator.simulate(run.n_samples)
        record = (
            _build_record(run=run, output=resolved_output, population=population, n_samples=run.n_samples)
            if resolved_output.save_metadata
            else None
        )
        write_population_catalogue(
            output_path=output_path,
            population=population,
            provenance=record,
            compression=resolved_output.compression,
        )
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    logger.info("Saved %s samples to %s", run.n_samples, output_path)
