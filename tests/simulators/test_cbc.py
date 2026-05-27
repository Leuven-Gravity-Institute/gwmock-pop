"""Tests for the graph-backed CBC population simulator."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from gwmock_pop import CBC_PARAMETER_NAMES, CBCSimulator
from gwmock_pop.graph.validation import ConfigValidationError
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import CBCSimulator as SimulatorsCBCSimulator
from gwmock_pop.simulators.graph import GraphSimulator


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


def test_simulate_returns_canonical_keys_and_shape() -> None:
    """CBCSimulator returns the full canonical CBC parameter surface in order."""
    simulator = CBCSimulator(seed=42)

    result = simulator.simulate(1000)

    assert list(result.keys()) == list(simulator.parameter_names)
    assert set(result.keys()) == set(CBC_PARAMETER_NAMES)
    assert all(array.shape == (1000,) for array in result.values())


def test_is_a_graph_simulator() -> None:
    """CBCSimulator is backed by the graph engine."""
    assert isinstance(CBCSimulator(), GraphSimulator)


def test_default_tidal_params_are_zero() -> None:
    """The default CBC population has zero tidal deformability."""
    result = CBCSimulator(seed=0).simulate(16)
    np.testing.assert_array_equal(np.asarray(result["lambda_1"]), np.zeros(16))
    np.testing.assert_array_equal(np.asarray(result["lambda_2"]), np.zeros(16))


def test_parameter_names_are_subset_of_cbc_parameter_names() -> None:
    """Default CBC parameters stay within the public CBC contract."""
    assert set(CBCSimulator().parameter_names).issubset(CBC_PARAMETER_NAMES)


def test_simulate_is_reproducible_for_fixed_seed() -> None:
    """Two simulators built with the same seed produce identical samples."""
    result_a = CBCSimulator(seed=42).simulate(256)
    result_b = CBCSimulator(seed=42).simulate(256)

    for name in CBCSimulator().parameter_names:
        np.testing.assert_array_equal(np.asarray(result_a[name]), np.asarray(result_b[name]))


def test_simulator_satisfies_protocol() -> None:
    """CBCSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(CBCSimulator(), GWPopSimulator)


def test_constructor_forwards_seed_to_random_mixin() -> None:
    """Constructor ``seed`` is passed through to ``RandomMixin`` / ``RNGManager``."""
    simulator = CBCSimulator(seed=99)
    assert simulator._rng_manager._seed == 99


def test_public_import_surfaces_export_cbc_simulator() -> None:
    """The simulator is importable from the package and simulators namespace."""
    assert SimulatorsCBCSimulator is CBCSimulator


def test_requested_marginals_pass_ks_checks() -> None:
    """Default isotropic and uniform marginals match their analytic targets."""
    simulator = CBCSimulator(m_min=10.0, m_max=80.0, seed=123)
    result = simulator.simulate(10_000)

    right_ascension = np.asarray(result["right_ascension"])
    declination = np.asarray(result["declination"])
    inclination = np.asarray(result["inclination"])
    mass_1 = np.asarray(result["detector_frame_mass_1"])

    p_ra = _ks_p_value(right_ascension, lambda x: x / (2.0 * math.pi))
    p_declination = _ks_p_value(declination, lambda x: 0.5 * (np.sin(x) + 1.0))
    p_inclination = _ks_p_value(inclination, lambda x: 0.5 * (1.0 - np.cos(x)))
    p_mass_1 = _ks_p_value(mass_1, lambda x: (x - 10.0) / (80.0 - 10.0))

    assert p_ra > 0.01, p_ra
    assert p_declination > 0.01, p_declination
    assert p_inclination > 0.01, p_inclination
    assert p_mass_1 > 0.01, p_mass_1


def test_distance_respects_configured_bounds() -> None:
    """Sampled luminosity distances stay within the configured range."""
    result = CBCSimulator(d_min=100.0, d_max=2_000.0, seed=11).simulate(4096)
    distance = np.asarray(result["distance"])
    assert np.all(distance >= 100.0)
    assert np.all(distance <= 2_000.0)


def test_precessing_spin_magnitudes_respect_chi_max() -> None:
    """Isotropic spin magnitudes stay within the configured bound."""
    result = CBCSimulator(chi_max=0.5, seed=9).simulate(4096)
    for component in (1, 2):
        magnitude = np.sqrt(
            np.asarray(result[f"spin_{component}x"]) ** 2
            + np.asarray(result[f"spin_{component}y"]) ** 2
            + np.asarray(result[f"spin_{component}z"]) ** 2
        )
        assert np.all(magnitude <= 0.5 + 1e-6)


def test_aligned_spins_zero_in_plane_components() -> None:
    """Aligned-spin mode zeroes the in-plane spin components."""
    result = CBCSimulator(aligned_spins=True, seed=7).simulate(128)
    for name in ("spin_1x", "spin_1y", "spin_2x", "spin_2y"):
        np.testing.assert_array_equal(np.asarray(result[name]), np.zeros(128))


def test_theta_jn_depends_on_spin_alignment_mode() -> None:
    """theta_jn matches inclination only in aligned-spin mode."""
    aligned = CBCSimulator(aligned_spins=True, seed=11).simulate(256)
    np.testing.assert_array_equal(np.asarray(aligned["theta_jn"]), np.asarray(aligned["inclination"]))

    precessing = CBCSimulator(aligned_spins=False, seed=11).simulate(256)
    assert not np.array_equal(np.asarray(precessing["theta_jn"]), np.asarray(precessing["inclination"]))


def test_total_mass_max_is_respected() -> None:
    """The total-mass cap rejects pairs exceeding the configured limit."""
    result = CBCSimulator(total_mass_max=80.0, seed=4).simulate(4096)
    total = np.asarray(result["detector_frame_mass_1"]) + np.asarray(result["detector_frame_mass_2"])
    assert np.all(total <= 80.0)


def test_total_mass_max_below_minimum_pair_raises() -> None:
    """A total-mass cap that admits no pairs raises at simulate time."""
    simulator = CBCSimulator(m_min=10.0, total_mass_max=20.0)
    with pytest.raises(ValueError, match="total_mass_max must be greater than"):
        simulator.simulate(8)


def test_parameters_override_replaces_a_distribution() -> None:
    """A user override swaps the distribution of a single parameter."""
    simulator = CBCSimulator(
        seed=3,
        parameters={
            "detector_frame_mass_1": {
                "sampler": {
                    "function": "power_law_plus_peak",
                    "arguments": {
                        "alpha": 3.5,
                        "minimum": 5.0,
                        "maximum": 100.0,
                        "lambda_peak": 0.1,
                        "peak_mean": 35.0,
                        "peak_sigma": 5.0,
                        "peak_maximum": 100.0,
                    },
                }
            }
        },
    )
    result = simulator.simulate(5000)
    mass_1 = np.asarray(result["detector_frame_mass_1"])
    # A steep power law concentrates probability near the minimum mass.
    assert mass_1.mean() < 30.0
    assert set(result.keys()) == set(CBC_PARAMETER_NAMES)


def test_parameters_override_adds_a_new_output_node() -> None:
    """A user override can add a new parameter, extending the output keys."""
    simulator = CBCSimulator(
        seed=3,
        parameters={
            "snr_threshold": {"sampler": {"function": "uniform", "arguments": {"minimum": 0.0, "maximum": 1.0}}}
        },
    )
    assert "snr_threshold" in simulator.parameter_names
    result = simulator.simulate(64)
    assert "snr_threshold" in result
    assert result["snr_threshold"].shape == (64,)


def test_parameters_override_with_unknown_function_raises() -> None:
    """An override referencing an unknown sampler fails fast with a clear error."""
    with pytest.raises(ConfigValidationError):
        CBCSimulator(parameters={"distance": {"sampler": {"function": "does_not_exist", "arguments": {}}}})
