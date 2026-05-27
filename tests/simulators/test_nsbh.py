"""Tests for the graph-backed NSBH population simulator."""

from __future__ import annotations

import numpy as np

from gwmock_pop import CBC_PARAMETER_NAMES, NSBHSimulator
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import NSBHSimulator as SimulatorsNSBHSimulator


def test_simulator_satisfies_protocol() -> None:
    """NSBHSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(NSBHSimulator(), GWPopSimulator)


def test_simulate_returns_canonical_keys_and_shape() -> None:
    """NSBHSimulator returns the full canonical CBC parameter surface in order."""
    simulator = NSBHSimulator(seed=42)

    result = simulator.simulate(1000)

    assert list(result.keys()) == list(simulator.parameter_names)
    assert set(result.keys()) == set(CBC_PARAMETER_NAMES)
    assert all(array.shape == (1000,) for array in result.values())


def test_parameter_names_are_subset_of_cbc_parameter_names() -> None:
    """NSBH parameters stay within the public CBC contract."""
    assert set(NSBHSimulator().parameter_names).issubset(CBC_PARAMETER_NAMES)


def test_simulate_is_reproducible_for_fixed_seed() -> None:
    """Two simulators built with the same seed produce identical samples."""
    result_a = NSBHSimulator(seed=42).simulate(256)
    result_b = NSBHSimulator(seed=42).simulate(256)

    for name in NSBHSimulator().parameter_names:
        np.testing.assert_array_equal(np.asarray(result_a[name]), np.asarray(result_b[name]))


def test_default_source_type_is_nsbh() -> None:
    """Default ``source_type`` identifies neutron star - black hole systems."""
    assert NSBHSimulator().source_type == "nsbh"


def test_public_import_surfaces_export_nsbh_simulator() -> None:
    """The simulator is importable from the package and simulators namespace."""
    assert SimulatorsNSBHSimulator is NSBHSimulator


def test_component_masses_respect_nsbh_bounds_and_ordering() -> None:
    """Primary masses use BH bounds and secondary masses use NS bounds."""
    simulator = NSBHSimulator(
        bh_mass_min=4.0,
        bh_mass_max=20.0,
        ns_mass_min=1.2,
        ns_mass_max=2.5,
        total_mass_max=22.0,
        seed=7,
    )

    result = simulator.simulate(4096)
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
    result = NSBHSimulator(bh_chi_max=0.8, ns_chi_max=0.03, seed=9).simulate(2048)
    spin_1_mag = np.sqrt(
        np.asarray(result["spin_1x"]) ** 2 + np.asarray(result["spin_1y"]) ** 2 + np.asarray(result["spin_1z"]) ** 2
    )
    spin_2_mag = np.sqrt(
        np.asarray(result["spin_2x"]) ** 2 + np.asarray(result["spin_2y"]) ** 2 + np.asarray(result["spin_2z"]) ** 2
    )

    assert np.all(spin_1_mag <= 0.8 + 1e-6)
    assert np.all(spin_2_mag <= 0.03 + 1e-6)


def test_bh_tidal_is_zero_and_ns_tidal_within_bound() -> None:
    """BH primary tidal deformability is zero; NS secondary is within bound."""
    result = NSBHSimulator(seed=10).simulate(1000)
    np.testing.assert_array_equal(np.asarray(result["lambda_1"]), np.zeros(1000))
    lambda_2 = np.asarray(result["lambda_2"])
    assert np.all(lambda_2 >= 0.0)
    assert np.all(lambda_2 <= 3000.0)


def test_ns_lambda_max_clamps_ns_tidal_output() -> None:
    """NSBHSimulator(ns_lambda_max=100) produces lambda_2 values at most 100."""
    result = NSBHSimulator(ns_lambda_max=100.0, seed=12).simulate(1000)
    assert np.all(np.asarray(result["lambda_2"]) <= 100.0)
