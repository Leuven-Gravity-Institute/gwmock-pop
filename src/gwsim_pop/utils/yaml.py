"""Utility functions for handling YAML files."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from ruyaml import YAML


# Register the representers for PyYAML
def _yaml_enum_representer(dumper, data: Enum):
    """Define the representer for Enum.

    Args:
        dumper: YAML dumper.
        data: Enum data.

    Returns:
        Representer of Enum.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data.value))


_yaml_representers = ((Enum, _yaml_enum_representer),)

# Register it globally for PyYAML
for data_type, representer in _yaml_representers:
    yaml.add_multi_representer(data_type=data_type, multi_representer=representer)
    yaml.SafeDumper.add_multi_representer(data_type=data_type, representer=representer)


def read_yaml(filename: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Read from file.

    Args:
        filename: File name.
        encoding: File encoding.

    Returns:
        A dictionary of data.

    """
    with Path(filename).open("r", encoding=encoding) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level of '{filename}', got {type(data).__name__!r}.")
    return data


def write_yaml(
    filename: str | Path,
    data: dict[str, Any],
    encoding: str = "utf-8",
    round_trip: bool = False,
    **kwargs,
):
    """Write to file.

    Args:
        filename: File name.
        data: A dictionary of data.
        encoding: Encoding of the file.
        round_trip: If True, use ruyaml to preserve comments/order.
        **kwargs: Extra arguments passed to yaml.safe_dump (or ruyaml.YAML.dump).
    """
    path = Path(filename)

    path.parent.mkdir(parents=True, exist_ok=True)

    if round_trip:
        yaml_handler = YAML(typ="rt")
        yaml_handler.default_flow_style = False
        yaml_handler.indent(mapping=4, sequence=4, offset=2)
        for data_type, representer in _yaml_representers:
            yaml_handler.representer.add_multi_representer(data_type, representer)

        with path.open("w", encoding=encoding) as f:
            yaml_handler.dump(data, f, **kwargs)

    else:
        with path.open("w", encoding=encoding) as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, **kwargs)
