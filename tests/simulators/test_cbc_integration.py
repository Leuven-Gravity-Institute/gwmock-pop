"""Integration tests for CBC simulation via the GWPopSimulator protocol."""

from __future__ import annotations

from importlib.resources import as_file, files

import jax.numpy as jnp
import pytest

from gwmock_pop import GWPopSimulator
from gwmock_pop.simulators import GraphSimulator

pytestmark = pytest.mark.integration


def test_bbh_end_to_end_via_protocol() -> None:
    """Exercise the full BBH config-file path through the public protocol API."""
    n_samples = 100
    with as_file(files("gwmock_pop.configs").joinpath("bbh_gwtc4.yaml")) as config_path:
        sim: GWPopSimulator = GraphSimulator.from_config_file(config_path, source_type="bbh", seed=42)

        assert isinstance(sim, GWPopSimulator)
        assert sim.source_type == "bbh"

        result = sim.simulate(n_samples)

        assert list(result.keys()) == list(sim.parameter_names)
        assert set(result.keys()) == set(sim.parameter_names)
        assert all(values.shape == (n_samples,) for values in result.values())
        assert all(bool(jnp.all(jnp.isfinite(values))) for values in result.values())

        single = {key: float(values[0]) for key, values in result.items()}
        assert isinstance(single, dict)
        assert all(isinstance(key, str) for key in single)
        assert all(isinstance(value, float) for value in single.values())
