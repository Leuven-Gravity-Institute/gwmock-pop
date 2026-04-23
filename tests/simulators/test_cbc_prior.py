"""Tests for the lightweight CBC prior simulator."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gwmock_pop import CBCPriorSimulator
from gwmock_pop.protocols import GWPopSimulator


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


def test_simulate_returns_expected_shape_and_keys() -> None:
    """CBCPriorSimulator returns the full canonical BBH parameter surface."""
    simulator = CBCPriorSimulator()

    result = simulator.simulate(1000, seed=42)

    assert list(result.keys()) == simulator.parameter_names
    assert all(array.shape == (1000,) for array in result.values())


def test_simulate_is_reproducible_for_fixed_seed() -> None:
    """The same seed produces identical CBC prior samples."""
    simulator = CBCPriorSimulator()

    result_a = simulator.simulate(256, seed=42)
    result_b = simulator.simulate(256, seed=42)

    for name in simulator.parameter_names:
        np.testing.assert_array_equal(np.asarray(result_a[name]), np.asarray(result_b[name]))


def test_simulator_satisfies_protocol() -> None:
    """CBCPriorSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(CBCPriorSimulator(), GWPopSimulator)


def test_requested_marginals_pass_ks_checks() -> None:
    """The requested RA, declination, distance, and mass marginals match their targets."""
    simulator = CBCPriorSimulator(m_min=10.0, m_max=80.0, d_max=2_000.0)
    result = simulator.simulate(10_000, seed=123)

    right_ascension = np.asarray(result["right_ascension"])
    declination = np.asarray(result["declination"])
    distance = np.asarray(result["distance"])
    mass_1 = np.asarray(result["detector_frame_mass_1"])

    p_ra = _ks_p_value(right_ascension, lambda x: x / (2.0 * math.pi))
    p_declination = _ks_p_value(declination, lambda x: 0.5 * (np.sin(x) + 1.0))
    p_distance = _ks_p_value(distance, lambda x: (x / simulator._d_max) ** 3)
    p_mass_1 = _ks_p_value(mass_1, lambda x: (x - simulator._m_min) / (simulator._m_max - simulator._m_min))

    assert p_ra > 0.01, p_ra
    assert p_declination > 0.01, p_declination
    assert p_distance > 0.01, p_distance
    assert p_mass_1 > 0.01, p_mass_1


def test_aligned_spins_zero_in_plane_components() -> None:
    """Aligned-spin mode zeroes the in-plane spin components."""
    simulator = CBCPriorSimulator(aligned_spins=True)

    result = simulator.simulate(128, seed=7)

    np.testing.assert_array_equal(np.asarray(result["spin_1x"]), np.zeros(128))
    np.testing.assert_array_equal(np.asarray(result["spin_1y"]), np.zeros(128))
    np.testing.assert_array_equal(np.asarray(result["spin_2x"]), np.zeros(128))
    np.testing.assert_array_equal(np.asarray(result["spin_2y"]), np.zeros(128))


def test_module_does_not_import_graph_simulator() -> None:
    """The lightweight simulator stays decoupled from graph-engine internals."""
    module_source = Path(CBCPriorSimulator.__module__.replace(".", "/"))
    del module_source
    source_text = Path("src/gwmock_pop/simulators/cbc_prior.py").read_text(encoding="utf-8")

    assert "gwmock_pop.simulators.graph" not in source_text
