"""Simulate populations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from gwmock_pop.config.main import MainConfiguration
from gwmock_pop.simulators.graph import GraphSimulator


def _get_output_path(config: MainConfiguration) -> Path:
    """Build the output path for a simulation run."""
    output_dir = Path(config.run.output.directory)
    suffixes = {"csv": ".csv", "hdf5": ".hdf5", "npz": ".npz"}
    return output_dir / f"{config.run.name}{suffixes[config.run.output.format]}"


def _build_output_metadata(config: MainConfiguration, simulator: GraphSimulator) -> dict[str, Any]:
    """Build generic output metadata for formats that can persist it."""
    return {
        "config_version": config.config_version,
        "run_name": config.run.name,
        "n_samples": config.run.n_samples,
        "seed": config.run.seed,
        "parameter_names": simulator.parameter_names,
    }


def simulate_command(filename: Annotated[str, typer.Argument(help="File name of the configuration file.")]) -> None:
    """Simulate a population from a configuration file.

    Args:
        filename: File name of the configuration file.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        config = MainConfiguration.from_file(filename)
    except (FileNotFoundError, ValidationError, ValueError) as error:
        logger.error("Failed to load configuration from %s: %s", filename, error)
        raise typer.Exit(1) from error

    if config.run.mode != "fixed_n_samples":
        logger.error("Only run.mode='fixed_n_samples' is supported by the CLI MVP.")
        raise typer.Exit(1)

    if not config.parameters:
        logger.error("No parameter graph configuration found under 'parameters'.")
        raise typer.Exit(1)

    output_path = _get_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not config.run.output.overwrite:
        logger.error(
            "Refusing to overwrite existing output file %s. Set run.output.overwrite=true to replace it.",
            output_path,
        )
        raise typer.Exit(1)

    simulator = GraphSimulator(config=config.parameters, seed=config.run.seed, source_type="population")
    population = simulator.simulate(n_samples=config.run.n_samples)

    metadata = None
    if config.run.output.save_metadata:
        metadata = _build_output_metadata(config, simulator)

    simulator.save(
        output_path=output_path,
        file_format=config.run.output.format,
        data=population,
        compression=config.run.output.compression,
        metadata=metadata,
    )
    sample_count = len(next(iter(population.values()))) if population else 0
    logger.info("Saved %s samples to %s", sample_count, output_path)
