"""Tests for packaged BBH preset simulators."""

from __future__ import annotations

import math
from collections.abc import Callable
from importlib.resources import files

import jax.numpy as jnp
import numpy as np
import pytest

import gwmock_pop
from gwmock_pop.distributions import power_law_plus_peak_cdf
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.bbh.base import BBHSimulator
from gwmock_pop.simulators.graph import GraphSimulator

_PRESET_HYPERPARAMETERS = {
    "alpha": 2.63,
    "minimum": 4.59,
    "maximum": 86.22,
    "lambda_peak": 0.10,
    "peak_mean": 33.07,
    "peak_sigma": 5.69,
    "peak_maximum": 100.0,
}

_CANONICAL_PARAMETER_NAMES = [
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "eccentricity",
    "distance",
    "coa_phase",
    "inclination",
    "theta_jn",
    "long_asc_node",
    "mean_per_ano",
    "coa_time",
    "right_ascension",
    "declination",
    "polarization_angle",
    "redshift",
    "f_ref",
]


def _ks_p_value(samples: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """Return the asymptotic one-sample Kolmogorov-Smirnov p-value."""
    sample = np.sort(np.asarray(samples, dtype=float))
    n_samples = sample.size
    empirical_upper = np.arange(1, n_samples + 1) / n_samples
    empirical_lower = np.arange(0, n_samples) / n_samples
    theoretical = np.clip(cdf(sample), 0.0, 1.0)
    statistic = max(
        float(np.max(empirical_upper - theoretical)),
        float(np.max(theoretical - empirical_lower)),
    )
    lam = (math.sqrt(n_samples) + 0.12 + 0.11 / math.sqrt(n_samples)) * statistic
    p_value = 2.0 * sum(((-1) ** (term - 1)) * math.exp(-2.0 * (lam**2) * (term**2)) for term in range(1, 101))
    return max(0.0, min(1.0, p_value))


def test_from_preset_returns_protocol_conformant_simulator() -> None:
    """The packaged preset builds a graph-backed GWPopSimulator."""
    simulator = BBHSimulator.from_preset("power_law_plus_peak", seed=42)

    assert isinstance(simulator, GWPopSimulator)
    assert simulator.source_type == "bbh"

    result = simulator.simulate(128)

    assert list(result.keys()) == list(simulator.parameter_names)
    assert list(result.keys()) == ["detector_frame_mass_1", "detector_frame_mass_2"]
    assert all(array.shape == (128,) for array in result.values())
    assert bool(jnp.all(result["detector_frame_mass_2"] <= result["detector_frame_mass_1"]))


@pytest.mark.parametrize("preset_name", ["bbh_gwtc4", "bbh_flat"])
def test_named_bbh_presets_return_canonical_bbh_surface(preset_name: str) -> None:
    """Named BBH presets expose the full canonical BBH parameter set."""
    simulator = BBHSimulator.from_preset(preset_name, seed=42)

    result = simulator.simulate(64)

    assert isinstance(simulator, GWPopSimulator)
    assert simulator.source_type == "bbh"
    assert list(result.keys()) == list(simulator.parameter_names)
    assert len(result) == 21
    assert list(result.keys()) == _CANONICAL_PARAMETER_NAMES
    assert all(array.shape == (64,) for array in result.values())
    assert bool(jnp.all(jnp.isfinite(result["distance"])))
    assert bool(jnp.all(jnp.isfinite(result["redshift"])))
    assert bool(jnp.all(result["detector_frame_mass_1"] >= result["detector_frame_mass_2"]))


def test_gwtc4_alias_still_loads_canonical_bbh_preset() -> None:
    """The legacy gwtc4 preset name remains available as a compatibility alias."""
    alias_simulator = BBHSimulator.from_preset("gwtc4", seed=42)
    canonical_simulator = BBHSimulator.from_preset("bbh_gwtc4", seed=42)

    alias_result = alias_simulator.simulate(16)
    canonical_result = canonical_simulator.simulate(16)

    assert alias_simulator.source_type == "bbh"
    assert list(alias_result.keys()) == _CANONICAL_PARAMETER_NAMES
    assert list(canonical_result.keys()) == _CANONICAL_PARAMETER_NAMES
    for parameter_name in _CANONICAL_PARAMETER_NAMES:
        np.testing.assert_allclose(
            np.asarray(alias_result[parameter_name]), np.asarray(canonical_result[parameter_name])
        )


def test_bns_flat_preset_loads_as_graph_simulator() -> None:
    """The BNS flat preset is available through the generic graph preset loader."""
    simulator = GraphSimulator.from_preset("bns_flat", seed=42)
    result = simulator.simulate(64)

    assert isinstance(simulator, GWPopSimulator)
    assert simulator.source_type == "bns"
    assert list(result.keys()) == _CANONICAL_PARAMETER_NAMES
    assert all(array.shape == (64,) for array in result.values())
    assert bool(jnp.all(result["detector_frame_mass_1"] >= result["detector_frame_mass_2"]))
    assert bool(jnp.all(result["detector_frame_mass_1"] >= 1.0))
    assert bool(jnp.all(result["detector_frame_mass_2"] >= 1.0))
    assert bool(jnp.all(result["detector_frame_mass_1"] <= 3.0))
    assert bool(jnp.all(result["detector_frame_mass_2"] <= 3.0))


def test_from_preset_rejects_unknown_preset_name() -> None:
    """Unknown preset names raise a helpful error."""
    with pytest.raises(ValueError, match="Unknown preset"):
        BBHSimulator.from_preset("does_not_exist")


def test_list_presets_returns_expected_packaged_names() -> None:
    """Top-level preset discovery returns the expected canonical preset names."""
    preset_names = gwmock_pop.list_presets()

    assert "bbh_gwtc4" in preset_names
    assert "bbh_flat" in preset_names
    assert "bns_flat" in preset_names
    assert "power_law_plus_peak" in preset_names


def test_packaged_preset_resources_are_available_via_importlib_resources() -> None:
    """Built-in preset config files are exposed as package resources."""
    resource_names = {resource.name for resource in files("gwmock_pop.configs").iterdir()}

    assert "bbh_gwtc4.yaml" in resource_names
    assert "bbh_flat.yaml" in resource_names
    assert "bns_flat.yaml" in resource_names


def test_power_law_plus_peak_preset_passes_primary_mass_ks_check() -> None:
    """The preset's primary-mass samples match the configured reference CDF."""
    simulator = BBHSimulator.from_preset("power_law_plus_peak", seed=123)
    result = simulator.simulate(1_000)
    primary_mass = np.asarray(result["detector_frame_mass_1"])

    p_value = _ks_p_value(
        primary_mass,
        lambda x: np.asarray(power_law_plus_peak_cdf(mass=jnp.asarray(x), **_PRESET_HYPERPARAMETERS)),
    )

    assert p_value > 0.05, p_value
