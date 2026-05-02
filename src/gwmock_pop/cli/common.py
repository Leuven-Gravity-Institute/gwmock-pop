"""Shared helpers for simulator-backed CLI commands."""

from __future__ import annotations

from pathlib import Path

from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.bbh.base import BBHSimulator
from gwmock_pop.simulators.graph import GraphSimulator

_CONFIG_FILE_SUFFIXES = {".yaml", ".yml", ".toml"}


def resolve_simulator(config: str, seed: int | None) -> GWPopSimulator:
    """Build a simulator from either a packaged preset name or a config-file path."""
    config_path = Path(config).expanduser()
    if config_path.exists():
        return GraphSimulator.from_config_file(config_path, source_type="bbh", seed=seed)

    try:
        return BBHSimulator.from_preset(config, seed=seed)
    except ValueError as error:
        if config_path.suffix.lower() in _CONFIG_FILE_SUFFIXES:
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}") from error
        raise ValueError(f"Unknown preset or configuration path {config!r}.") from error
