"""Shared helpers for simulator-backed CLI commands.

The precedence rules here are the point of this module. Every setting a run
needs is settled in one place, so a value written in a configuration file is
either used or reported, never quietly ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gwmock_pop.config.main import MainConfiguration
from gwmock_pop.config.simulation import DEFAULT_SOURCE_TYPE, resolve_simulation_target
from gwmock_pop.loaders.file_loader import infer_population_file_format, supported_population_formats
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.graph import GraphSimulator


@dataclass(frozen=True)
class ResolvedRun:
    """A simulator and the run settings the CLI settled on for it.

    Attributes:
        simulator: The configured simulator.
        configuration: The simulation configuration the run was resolved from.
        seed: The seed the simulator's generator was actually initialized with.
        seed_source: Where that seed came from: ``"cli"``, ``"config"`` or
            ``"drawn"``.
        n_samples: The resolved sample count, or ``None`` when neither the command
            line nor the configuration named one.
        preset: Canonical name of the packaged preset behind the run, if any.
        config_path: Path of the configuration file behind the run, if any.
    """

    simulator: GraphSimulator
    configuration: MainConfiguration
    seed: int
    seed_source: str
    n_samples: int | None
    preset: str | None
    config_path: Path | None

    @property
    def run_name(self) -> str:
        """Return the run identifier.

        Returns:
            The configured run name.
        """
        return self.configuration.run.name


@dataclass(frozen=True)
class ResolvedOutput:
    """Where and how a catalogue is to be written.

    Attributes:
        path: Destination file.
        file_format: Format inferred from the destination suffix.
        compression: HDF5 compression filter to apply, if any.
        overwrite: Whether an existing destination may be replaced.
        save_metadata: Whether to store the provenance record.
    """

    path: Path
    file_format: str
    compression: str | None
    overwrite: bool
    save_metadata: bool


def _resolve_seed(cli_seed: int | None, configuration: MainConfiguration) -> tuple[int | None, str]:
    """Settle which seed a run starts from.

    ``--seed`` wins, then a seed the configuration explicitly declares, and
    failing both a seed is drawn. Only an explicitly declared ``run.seed`` counts:
    its schema default would otherwise turn every unseeded run into the same run.

    Args:
        cli_seed: Seed given on the command line, if any.
        configuration: The resolved configuration.

    Returns:
        The seed to build the generator with -- ``None`` to have one drawn -- and
        the name of where it came from.
    """
    if cli_seed is not None:
        return cli_seed, "cli"
    if "seed" in configuration.run.model_fields_set:
        return configuration.run.seed, "config"
    return None, "drawn"


def _resolve_sample_count(cli_n: int | None, configuration: MainConfiguration) -> int | None:
    """Settle how many samples a run draws.

    Args:
        cli_n: Count given on the command line, if any.
        configuration: The resolved configuration.

    Returns:
        The count, or ``None`` when neither source named one. As with the seed,
        only an explicitly declared ``run.n_samples`` counts, so a run never
        silently falls back to the schema's million samples.
    """
    if cli_n is not None:
        return cli_n
    if "n_samples" in configuration.run.model_fields_set:
        return configuration.run.n_samples
    return None


def resolve_run(config: str, *, seed: int | None = None, n_samples: int | None = None) -> ResolvedRun:
    """Build a simulator and settle the run settings that drive it.

    Args:
        config: Packaged preset name or path to a YAML/TOML configuration file.
        seed: Seed given on the command line, if any.
        n_samples: Sample count given on the command line, if any.

    Returns:
        The simulator and the settings it was built with.

    Raises:
        ValueError: If ``config`` names neither a preset nor a readable
            configuration file.
        FileNotFoundError: If it looks like a configuration path that is missing.
    """
    target = resolve_simulation_target(config)
    requested_seed, seed_source = _resolve_seed(cli_seed=seed, configuration=target.configuration)
    simulator = GraphSimulator.from_target(
        target,
        source_type=target.source_type or DEFAULT_SOURCE_TYPE,
        seed=requested_seed,
    )
    return ResolvedRun(
        simulator=simulator,
        configuration=target.configuration,
        # The generator is the one authority on the seed in use, so a drawn seed
        # is read back from it rather than drawn a second time here.
        seed=simulator.rng_manager.resolved_seed,
        seed_source=seed_source,
        n_samples=_resolve_sample_count(cli_n=n_samples, configuration=target.configuration),
        preset=target.preset,
        config_path=target.config_path,
    )


def resolve_output(output: Path, configuration: MainConfiguration) -> ResolvedOutput:
    """Settle where and how a catalogue is written.

    ``output.directory``, ``output.format`` and ``output.compression`` are
    honoured only when the configuration states them, because each of their
    schema defaults contradicts what ``--output`` on its own asks for.
    ``overwrite`` and ``save_metadata`` are read as they stand.

    Args:
        output: Destination given on the command line.
        configuration: The resolved configuration.

    Returns:
        The resolved destination and write options.

    Raises:
        ValueError: If the destination suffix names an unsupported format, or a
            declared ``output.format`` names a format this package does not write
            or contradicts the destination suffix.
    """
    output_config = configuration.run.output
    declared = output_config.model_fields_set

    path = output.expanduser()
    if "directory" in declared and not path.is_absolute():
        path = Path(output_config.directory).expanduser() / path

    file_format = infer_population_file_format(path)
    if "format" in declared and output_config.format not in supported_population_formats():
        supported = ", ".join(sorted(supported_population_formats()))
        raise ValueError(
            f"Configured output format {output_config.format!r} is not a catalogue format this package "
            f"writes. Supported formats: {supported}."
        )
    if "format" in declared and output_config.format != file_format:
        raise ValueError(
            f"Configured output format {output_config.format!r} contradicts the destination {path}, "
            f"which is a {file_format!r} file. Change one of them."
        )

    return ResolvedOutput(
        path=path,
        file_format=file_format,
        compression=output_config.compression if "compression" in declared else None,
        overwrite=output_config.overwrite,
        save_metadata=output_config.save_metadata,
    )


def resolve_simulator(config: str, seed: int | None) -> GWPopSimulator:
    """Build a simulator from either a packaged preset name or a config-file path.

    Args:
        config: Packaged preset name or path to a YAML/TOML configuration file.
        seed: Seed given on the command line, if any.

    Returns:
        The configured simulator.
    """
    return resolve_run(config=config, seed=seed).simulator
