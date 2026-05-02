"""Tests for packaged BBH preset simulators."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop.distributions import power_law_plus_peak_cdf
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.bbh.base import BBHSimulator

_PRESET_HYPERPARAMETERS = {
    "alpha": 2.63,
    "minimum": 4.59,
    "maximum": 86.22,
    "lambda_peak": 0.10,
    "peak_mean": 33.07,
    "peak_sigma": 5.69,
    "peak_maximum": 100.0,
}


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


def test_from_preset_rejects_unknown_preset_name() -> None:
    """Unknown preset names raise a helpful error."""
    with pytest.raises(ValueError, match="Unknown BBH preset"):
        BBHSimulator.from_preset("does_not_exist")


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
