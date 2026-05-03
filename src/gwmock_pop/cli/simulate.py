"""Simulate populations from packaged presets or graph config files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gwmock_pop.cli.common import resolve_simulator
from gwmock_pop.loaders.file_loader import infer_population_file_format, write_population_catalogue


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
        infer_population_file_format(output_path)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.error("Refusing to overwrite existing output file %s.", output_path)
        raise typer.Exit(1)

    try:
        population = simulator.simulate(n)
        write_population_catalogue(output_path=output_path, population=population)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    logger.info("Saved %s samples to %s", n, output_path)
