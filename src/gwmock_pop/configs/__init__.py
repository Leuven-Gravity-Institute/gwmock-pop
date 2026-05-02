"""Packaged graph-simulator preset configurations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from gwmock_pop.utils.yaml import read_data_file

_SUPPORTED_CONFIG_SUFFIXES = {".yaml", ".yml", ".toml"}
_REQUIRED_PRESET_FIELDS = ("name", "source_type", "description", "parameters")


@dataclass(frozen=True)
class PackagedPreset:
    """Metadata for one packaged simulator preset."""

    name: str
    source_type: str
    description: str
    resource_name: str


def _iter_preset_resources() -> list[Traversable]:
    """Return packaged config resources that represent built-in presets."""
    return sorted(
        (
            resource
            for resource in files(__name__).iterdir()
            if resource.is_file() and Path(resource.name).suffix.lower() in _SUPPORTED_CONFIG_SUFFIXES
        ),
        key=lambda resource: resource.name,
    )


def iter_packaged_presets() -> list[PackagedPreset]:
    """Return metadata for all packaged presets."""
    presets: list[PackagedPreset] = []
    for resource in _iter_preset_resources():
        with as_file(resource) as config_path:
            config = read_data_file(config_path)

        missing = [field for field in _REQUIRED_PRESET_FIELDS if field not in config]
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(f"Packaged preset {resource.name} is missing required metadata fields: {missing_fields}.")

        presets.append(
            PackagedPreset(
                name=str(config["name"]),
                source_type=str(config["source_type"]),
                description=str(config["description"]),
                resource_name=resource.name,
            )
        )

    return sorted(presets, key=lambda preset: preset.name)


def get_packaged_preset_resource(preset_name: str) -> Traversable:
    """Return the packaged config resource for a named preset."""
    presets = iter_packaged_presets()
    for preset in presets:
        if preset.name == preset_name:
            return files(__name__).joinpath(preset.resource_name)

    available = ", ".join(p.name for p in presets)
    raise ValueError(f"Unknown preset {preset_name!r}. Available presets: {available}.")


__all__ = ["PackagedPreset", "get_packaged_preset_resource", "iter_packaged_presets"]
