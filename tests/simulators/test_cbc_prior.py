"""Tests for the lightweight CBC prior simulator."""

from __future__ import annotations

import importlib
import inspect
import math
import sys
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop import CBCPriorSimulator
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import cbc_prior as cbc_prior_module


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


def test_constructor_forwards_seed_to_random_mixin() -> None:
    """Constructor ``seed`` is passed through to ``RandomMixin`` / ``RNGManager``."""
    simulator = CBCPriorSimulator(seed=99)
    assert simulator._rng_manager._seed == 99


def test_requested_marginals_pass_ks_checks() -> None:
    """The requested RA, declination, distance, and mass marginals match their targets."""
    simulator = CBCPriorSimulator(m_min=10.0, m_max=80.0, d_max=2_000.0, accurate_cosmology=False)
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


def test_accurate_cosmology_mode_uses_lookup_sampler_and_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accurate cosmology mode delegates to the new sampler and transform nodes."""
    calls = {"sampler": 0, "transform": 0}

    def fake_sampler(*, key, n_samples: int, d_max: float):
        calls["sampler"] += 1
        assert d_max == 5_000.0
        return jnp.full((n_samples,), 1234.5)

    def fake_transform(distance):
        calls["transform"] += 1
        return jnp.full(distance.shape, 0.321)

    monkeypatch.setattr(cbc_prior_module, "uniform_comoving_volume_distance", fake_sampler)
    monkeypatch.setattr(cbc_prior_module, "comoving_distance_to_redshift", fake_transform)

    result = CBCPriorSimulator().simulate(8, seed=7)

    assert calls == {"sampler": 1, "transform": 1}
    np.testing.assert_array_equal(np.asarray(result["distance"]), np.full(8, 1234.5))
    np.testing.assert_allclose(np.asarray(result["redshift"]), np.full(8, 0.321))


def test_legacy_cosmology_mode_reproduces_low_redshift_hubble_law() -> None:
    """Legacy cosmology mode preserves the previous low-redshift approximation."""
    simulator = CBCPriorSimulator(d_max=2_000.0, accurate_cosmology=False)

    result = simulator.simulate(4096, seed=42)
    expected_redshift = np.asarray(result["distance"]) * (
        cbc_prior_module.PLANCK18_H0_KM_S_MPC / cbc_prior_module._SPEED_OF_LIGHT_KM_S
    )

    np.testing.assert_allclose(np.asarray(result["redshift"]), expected_redshift)


def test_aligned_spins_zero_in_plane_components() -> None:
    """Aligned-spin mode zeroes the in-plane spin components."""
    simulator = CBCPriorSimulator(aligned_spins=True)

    result = simulator.simulate(128, seed=7)

    np.testing.assert_array_equal(np.asarray(result["spin_1x"]), np.zeros(128))
    np.testing.assert_array_equal(np.asarray(result["spin_1y"]), np.zeros(128))
    np.testing.assert_array_equal(np.asarray(result["spin_2x"]), np.zeros(128))
    np.testing.assert_array_equal(np.asarray(result["spin_2y"]), np.zeros(128))


def test_theta_jn_depends_on_spin_alignment_mode() -> None:
    """theta_jn matches inclination only in aligned-spin mode."""
    aligned = CBCPriorSimulator(aligned_spins=True)
    aligned_result = aligned.simulate(256, seed=11)
    np.testing.assert_array_equal(
        np.asarray(aligned_result["theta_jn"]),
        np.asarray(aligned_result["inclination"]),
    )

    precessing = CBCPriorSimulator(aligned_spins=False)
    precessing_result = precessing.simulate(256, seed=11)
    assert not np.array_equal(
        np.asarray(precessing_result["theta_jn"]),
        np.asarray(precessing_result["inclination"]),
    )


def test_module_does_not_import_graph_simulator() -> None:
    """Public imports should not eagerly pull in the graph simulator."""
    sys.modules.pop("gwmock_pop.simulators", None)
    sys.modules.pop("gwmock_pop.simulators.cbc_prior", None)
    sys.modules.pop("gwmock_pop.simulators.graph", None)

    simulators_module = importlib.import_module("gwmock_pop.simulators")
    if simulators_module.__spec__ is None or simulators_module.__spec__.origin is None:
        source_text = inspect.getsource(simulators_module)
    else:
        source_text = Path(simulators_module.__spec__.origin).read_text(encoding="utf-8")
    assert "gwmock_pop.simulators.graph" not in source_text

    importlib.import_module("gwmock_pop.simulators.cbc_prior")
    assert "gwmock_pop.simulators.graph" not in sys.modules


def test_chi_max_enforces_closed_unit_interval() -> None:
    """chi_max must lie in [0, 1]."""
    with pytest.raises(ValueError, match=r"chi_max must be in \[0, 1\]"):
        CBCPriorSimulator(chi_max=1.01)
    with pytest.raises(ValueError, match=r"chi_max must be in \[0, 1\]"):
        CBCPriorSimulator(chi_max=-0.01)


def test_total_mass_max_must_be_strictly_greater_than_two_m_min() -> None:
    """total_mass_max == 2*m_min is rejected to avoid zero-measure acceptance."""
    with pytest.raises(ValueError, match="total_mass_max must be greater than 2 \\* m_min"):
        CBCPriorSimulator(m_min=10.0, total_mass_max=20.0)


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("m_min", {"m_min": math.inf}),
        ("m_max", {"m_max": math.nan}),
        ("d_max", {"d_max": math.inf}),
        ("chi_max", {"chi_max": math.nan}),
        ("gps_start", {"gps_start": math.inf}),
        ("gps_end", {"gps_end": math.nan}),
        ("f_ref", {"f_ref": math.inf}),
        ("total_mass_max", {"total_mass_max": math.nan}),
    ],
)
def test_numeric_configuration_values_must_be_finite(field: str, kwargs: dict[str, float]) -> None:
    """All numeric configuration bounds must be finite."""
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        CBCPriorSimulator(**kwargs)
