"""Tests for the lightweight NSBH prior simulator."""

from __future__ import annotations

import numpy as np

from gwmock_pop import CBC_PARAMETER_NAMES, NSBHPriorSimulator
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import NSBHPriorSimulator as SimulatorsNSBHPriorSimulator


def test_simulator_satisfies_protocol() -> None:
    """NSBHPriorSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(NSBHPriorSimulator(), GWPopSimulator)


def test_simulate_returns_expected_shape_and_keys() -> None:
    """NSBHPriorSimulator returns the full canonical CBC parameter surface."""
    simulator = NSBHPriorSimulator()

    result = simulator.simulate(1000, seed=42)

    assert list(result.keys()) == simulator.parameter_names
    assert all(array.shape == (1000,) for array in result.values())


def test_parameter_names_are_subset_of_cbc_parameter_names() -> None:
    """NSBHPriorSimulator parameters stay within the public CBC contract."""
    simulator = NSBHPriorSimulator()
    assert set(simulator.parameter_names).issubset(CBC_PARAMETER_NAMES)


def test_simulate_is_reproducible_for_fixed_seed() -> None:
    """The same seed produces identical NSBH prior samples."""
    simulator = NSBHPriorSimulator()

    result_a = simulator.simulate(256, seed=42)
    result_b = simulator.simulate(256, seed=42)

    for name in simulator.parameter_names:
        np.testing.assert_array_equal(np.asarray(result_a[name]), np.asarray(result_b[name]))


def test_default_source_type_is_nsbh() -> None:
    """Default ``source_type`` identifies neutron star - black hole systems."""
    assert NSBHPriorSimulator().source_type == "nsbh"


def test_component_masses_respect_nsbh_bounds_and_ordering() -> None:
    """Primary masses use BH bounds and secondary masses use NS bounds."""
    simulator = NSBHPriorSimulator(
        bh_mass_min=4.0,
        bh_mass_max=20.0,
        ns_mass_min=1.2,
        ns_mass_max=2.5,
        total_mass_max=22.0,
    )

    result = simulator.simulate(4096, seed=7)
    mass_1 = np.asarray(result["detector_frame_mass_1"])
    mass_2 = np.asarray(result["detector_frame_mass_2"])

    assert np.all(mass_1 >= 4.0)
    assert np.all(mass_1 <= 20.0)
    assert np.all(mass_2 >= 1.2)
    assert np.all(mass_2 <= 2.5)
    assert np.all(mass_1 >= mass_2)
    assert np.all(mass_1 + mass_2 <= 22.0)


def test_component_spin_magnitudes_respect_independent_bounds() -> None:
    """BH and NS spins use different configured magnitude caps."""
    simulator = NSBHPriorSimulator(bh_chi_max=0.8, ns_chi_max=0.03)

    result = simulator.simulate(2048, seed=9)
    spin_1_mag = np.sqrt(
        np.asarray(result["spin_1x"]) ** 2 + np.asarray(result["spin_1y"]) ** 2 + np.asarray(result["spin_1z"]) ** 2
    )
    spin_2_mag = np.sqrt(
        np.asarray(result["spin_2x"]) ** 2 + np.asarray(result["spin_2y"]) ** 2 + np.asarray(result["spin_2z"]) ** 2
    )

    assert np.all(spin_1_mag <= 0.8)
    assert np.all(spin_2_mag <= 0.03)


def test_public_import_surfaces_export_nsbh_prior_simulator() -> None:
    """The simulator is importable from the package and simulators namespace."""
    assert SimulatorsNSBHPriorSimulator is NSBHPriorSimulator


def test_nsbh_prior_has_tidal_params_in_parameter_names() -> None:
    """lambda_1 and lambda_2 appear in NSBHPriorSimulator parameter_names."""
    simulator = NSBHPriorSimulator()
    assert "lambda_1" in simulator.parameter_names
    assert "lambda_2" in simulator.parameter_names


def test_nsbh_prior_bh_tidal_is_zero() -> None:
    """BH primary tidal deformability (lambda_1) is always zero for NSBH."""
    result = NSBHPriorSimulator().simulate(500, seed=10)
    np.testing.assert_array_equal(np.asarray(result["lambda_1"]), np.zeros(500))


def test_nsbh_prior_ns_tidal_is_non_negative_and_within_default_bound() -> None:
    """NS secondary tidal deformability (lambda_2) is in [0, ns_lambda_max]."""
    result = NSBHPriorSimulator().simulate(1000, seed=11)
    lambda_2 = np.asarray(result["lambda_2"])
    assert np.all(lambda_2 >= 0.0)
    assert np.all(lambda_2 <= 3000.0)


def test_nsbh_prior_ns_lambda_max_clamps_ns_tidal_output() -> None:
    """NSBHPriorSimulator(ns_lambda_max=100) produces lambda_2 values at most 100."""
    result = NSBHPriorSimulator(ns_lambda_max=100.0).simulate(1000, seed=12)
    assert np.all(np.asarray(result["lambda_2"]) <= 100.0)
