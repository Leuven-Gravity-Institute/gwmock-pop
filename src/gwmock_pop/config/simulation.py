"""Resolve a preset name or a configuration file into the settings a run needs.

The CLI and :class:`~gwmock_pop.simulators.graph.GraphSimulator` both start from
one ``--config``-shaped value: a packaged preset name or a path to a YAML/TOML
file. Resolving it in one place is what keeps the structured configuration tree
attached to the code that runs: a file's ``run`` and ``output`` blocks are read
here, so they cannot be silently dropped on the way to the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path
from typing import Any

from gwmock_pop.config.main import MainConfiguration
from gwmock_pop.configs import get_packaged_preset, get_packaged_preset_resource
from gwmock_pop.graph.validation import load_validated_parameters_config
from gwmock_pop.utils.yaml import read_data_file

# Suffixes accepted for a configuration file.
CONFIG_FILE_SUFFIXES = frozenset({".yaml", ".yml", ".toml"})

# Source type assumed for a configuration file that declares none.
DEFAULT_SOURCE_TYPE = "bbh"

_GRAPH_ROOT_KEY = "parameters"


@dataclass(frozen=True)
class SimulationTarget:
    """Everything one ``--config`` value resolves to.

    Attributes:
        graph_config: The validated parameter graph.
        configuration: The simulation configuration declared alongside the graph,
            with every default filled in. Its ``parameters`` field is left empty:
            the graph is carried by ``graph_config`` so there is one copy of it.
        source_type: Source type declared by the preset or the file, or ``None``
            when none was declared.
        preset: Canonical name of the packaged preset, when the target is one.
        config_path: Path of the configuration file, when the target is one.
    """

    graph_config: dict[str, Any]
    configuration: MainConfiguration
    source_type: str | None
    preset: str | None = None
    config_path: Path | None = None


def _declared_root(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return the root mapping when it carries the full-configuration form.

    A file whose root *is* the parameter graph is not searched for configuration
    blocks: a graph node named ``run`` or ``advanced`` would otherwise be read as
    one.

    Args:
        raw: Parsed contents of the configuration file.

    Returns:
        The root mapping, or ``None`` when the root is the graph itself.
    """
    return raw if _GRAPH_ROOT_KEY in raw else None


def _configuration_from_root(root: dict[str, Any] | None) -> MainConfiguration:
    """Build the declared configuration from a config-file root mapping."""
    if root is None:
        return MainConfiguration()
    declared = {
        key: value for key, value in root.items() if key in MainConfiguration.model_fields and key != _GRAPH_ROOT_KEY
    }
    return MainConfiguration(**declared)


def _source_type_from_root(root: dict[str, Any] | None) -> str | None:
    """Return the source type declared at a config-file root, if any."""
    if root is None:
        return None
    declared = root.get("source_type")
    if isinstance(declared, str) and declared.strip():
        return declared
    return None


def simulation_target_from_file(
    config_path: str | Path,
    *,
    encoding: str = "utf-8",
) -> SimulationTarget:
    """Resolve a configuration file into a simulation target.

    Args:
        config_path: Path to a YAML/TOML configuration file.
        encoding: Encoding of the file.

    Returns:
        The resolved target.

    Raises:
        ConfigValidationError: If the parameter graph fails validation.
    """
    path = Path(config_path)
    graph_config, _ = load_validated_parameters_config(config_path=path, encoding=encoding)
    root = _declared_root(read_data_file(path, encoding=encoding))
    return SimulationTarget(
        graph_config=graph_config,
        configuration=_configuration_from_root(root),
        source_type=_source_type_from_root(root),
        config_path=path,
    )


def simulation_target_from_preset(preset_name: str, *, encoding: str = "utf-8") -> SimulationTarget:
    """Resolve a packaged preset name into a simulation target.

    Args:
        preset_name: Name or compatibility alias of a packaged preset.
        encoding: Encoding of the packaged file.

    Returns:
        The resolved target, carrying the preset's canonical name rather than the
        temporary path the packaged resource was read from.

    Raises:
        ValueError: If no packaged preset has that name.
    """
    preset = get_packaged_preset(preset_name)
    with as_file(get_packaged_preset_resource(preset_name)) as config_path:
        target = simulation_target_from_file(config_path, encoding=encoding)
    return SimulationTarget(
        graph_config=target.graph_config,
        configuration=target.configuration,
        source_type=preset.source_type,
        preset=preset.name,
        config_path=None,
    )


def resolve_simulation_target(config: str, *, encoding: str = "utf-8") -> SimulationTarget:
    """Resolve a preset name or configuration-file path into a simulation target.

    Args:
        config: Packaged preset name or path to a YAML/TOML configuration file.
        encoding: Encoding of a configuration file.

    Returns:
        The resolved target.

    Raises:
        ValueError: If the path is not a file, its suffix is unsupported, or the
            value names neither a preset nor an existing path.
        FileNotFoundError: If the value looks like a configuration-file path but
            no such file exists.
    """
    config_path = Path(config).expanduser()
    if config_path.is_file():
        if config_path.suffix.lower() not in CONFIG_FILE_SUFFIXES:
            supported = ", ".join(sorted(CONFIG_FILE_SUFFIXES))
            raise ValueError(f"Unsupported config-file suffix for {config_path}. Supported suffixes: {supported}.")
        return simulation_target_from_file(config_path, encoding=encoding)
    if config_path.exists():
        raise ValueError(f"Configuration path is not a file: {config_path}")
    try:
        return simulation_target_from_preset(config, encoding=encoding)
    except ValueError as error:
        if config_path.suffix.lower() in CONFIG_FILE_SUFFIXES:
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}") from error
        raise ValueError(f"Unknown preset or configuration path {config!r}.") from error
