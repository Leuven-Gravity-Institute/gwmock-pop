"""Shared fixtures for the provenance test modules."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

MINIMAL_GRAPH_PARAMETERS = """
parameters:
  mass_1:
    sampler:
      function: log_uniform
      arguments:
        minimum: 5.0
        maximum: 100.0
  mass_2:
    transform:
      function: constant_like
      arguments:
        reference: '@mass_1'
        value: 1.4
""".lstrip()

DEFAULTED_GRAPH_PARAMETERS = """
parameters:
  distance:
    sampler:
      function: uniform_comoving_volume_distance
      arguments:
        d_max: 1000.0
""".lstrip()


@pytest.fixture
def write_graph_config(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes a graph config file into ``tmp_path``."""

    def _write(
        *,
        preamble: str = "",
        parameters: str = MINIMAL_GRAPH_PARAMETERS,
        name: str = "config.yaml",
    ) -> Path:
        config_path = tmp_path / name
        config_path.write_text(preamble + parameters, encoding="utf-8")
        return config_path

    return _write
