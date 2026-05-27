"""Tests for the lightweight BNS prior simulator."""

from __future__ import annotations

import numpy as np

from gwmock_pop import CBC_PARAMETER_NAMES, BNSPriorSimulator
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import BNSPriorSimulator as SimulatorsBNSPriorSimulator


def test_simulate_returns_expected_shape_and_keys() -> None:
    """BNSPriorSimulator returns the full canonical CBC parameter surface."""
    simulator = BNSPriorSimulator()

    result = simulator.simulate(1000, seed=42)

    assert list(result.keys()) == simulator.parameter_names
    assert all(array.shape == (1000,) for array in result.values())


def test_parameter_names_are_subset_of_cbc_parameter_names() -> None:
    """BNSPriorSimulator parameters stay within the public CBC contract."""
    simulator = BNSPriorSimulator()
    assert set(simulator.parameter_names).issubset(CBC_PARAMETER_NAMES)


def test_simulate_is_reproducible_for_fixed_seed() -> None:
    """The same seed produces identical BNS prior samples."""
    simulator = BNSPriorSimulator()

    result_a = simulator.simulate(256, seed=42)
    result_b = simulator.simulate(256, seed=42)

    for name in simulator.parameter_names:
        np.testing.assert_array_equal(np.asarray(result_a[name]), np.asarray(result_b[name]))


def test_simulator_satisfies_protocol() -> None:
    """BNSPriorSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(BNSPriorSimulator(), GWPopSimulator)


def test_default_source_type_is_bns() -> None:
    """The BNS prior routes to the BNS signal backend."""
    assert BNSPriorSimulator().source_type == "bns"


def test_component_masses_respect_bounds_ordering_and_total_mass_limit() -> None:
    """Mass samples stay in range, ordered, and obey an optional total-mass cap."""
    simulator = BNSPriorSimulator(total_mass_max=4.2)

    result = simulator.simulate(4096, seed=7)
    mass_1 = np.asarray(result["detector_frame_mass_1"])
    mass_2 = np.asarray(result["detector_frame_mass_2"])

    assert np.all(mass_2 >= 1.0)
    assert np.all(mass_1 <= 3.0)
    assert np.all(mass_1 >= mass_2)
    assert np.all(mass_1 + mass_2 <= 4.2)


def test_spin_magnitudes_respect_configured_low_spin_bounds() -> None:
    """Default aligned spins stay within the configured low-spin interval."""
    simulator = BNSPriorSimulator(chi_max=0.03)

    result = simulator.simulate(2048, seed=9)
    spin_1x = np.asarray(result["spin_1x"])
    spin_1y = np.asarray(result["spin_1y"])
    spin_1z = np.asarray(result["spin_1z"])
    spin_2x = np.asarray(result["spin_2x"])
    spin_2y = np.asarray(result["spin_2y"])
    spin_2z = np.asarray(result["spin_2z"])

    np.testing.assert_array_equal(spin_1x, np.zeros(2048))
    np.testing.assert_array_equal(spin_1y, np.zeros(2048))
    np.testing.assert_array_equal(spin_2x, np.zeros(2048))
    np.testing.assert_array_equal(spin_2y, np.zeros(2048))

    spin_1_mag = np.sqrt(spin_1x**2 + spin_1y**2 + spin_1z**2)
    spin_2_mag = np.sqrt(spin_2x**2 + spin_2y**2 + spin_2z**2)
    assert np.all(spin_1_mag <= 0.03)
    assert np.all(spin_2_mag <= 0.03)


def test_public_import_surfaces_export_bns_prior_simulator() -> None:
    """The simulator is importable from the package and simulators namespace."""
    assert SimulatorsBNSPriorSimulator is BNSPriorSimulator


def test_bns_prior_has_tidal_params_in_parameter_names() -> None:
    """lambda_1 and lambda_2 appear in BNSPriorSimulator parameter_names."""
    simulator = BNSPriorSimulator()
    assert "lambda_1" in simulator.parameter_names
    assert "lambda_2" in simulator.parameter_names


def test_bns_prior_tidal_values_are_non_negative_and_within_default_bound() -> None:
    """Sampled tidal params lie in [0, lambda_max] with the default bound of 3000."""
    result = BNSPriorSimulator().simulate(1000, seed=5)
    lambda_1 = np.asarray(result["lambda_1"])
    lambda_2 = np.asarray(result["lambda_2"])
    assert np.all(lambda_1 >= 0.0)
    assert np.all(lambda_1 <= 3000.0)
    assert np.all(lambda_2 >= 0.0)
    assert np.all(lambda_2 <= 3000.0)


def test_bns_prior_lambda_max_clamps_tidal_output() -> None:
    """BNSPriorSimulator(lambda_max=500) produces tidal values at most 500."""
    result = BNSPriorSimulator(lambda_max=500.0).simulate(1000, seed=6)
    assert np.all(np.asarray(result["lambda_1"]) <= 500.0)
    assert np.all(np.asarray(result["lambda_2"]) <= 500.0)


def test_bns_prior_d_min_enforces_lower_distance_bound() -> None:
    """Sampled distances respect d_min when set."""
    sim = BNSPriorSimulator(d_min=100.0, d_max=500.0)
    result = sim.simulate(2048, seed=11)
    distance = np.asarray(result["distance"])
    assert np.all(distance >= 100.0), f"min distance {distance.min():.2f} < 100"
    assert np.all(distance <= 500.0), f"max distance {distance.max():.2f} > 500"


def test_bns_prior_d_min_default_zero_preserves_behaviour() -> None:
    """Default d_min=0 produces the same results as before."""
    result_default = BNSPriorSimulator(d_max=1000.0).simulate(256, seed=42)
    result_explicit = BNSPriorSimulator(d_min=0.0, d_max=1000.0).simulate(256, seed=42)
    np.testing.assert_array_equal(
        np.asarray(result_default["distance"]),
        np.asarray(result_explicit["distance"]),
    )
