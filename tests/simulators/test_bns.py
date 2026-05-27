"""Tests for the graph-backed BNS population simulator."""

from __future__ import annotations

import numpy as np

from gwmock_pop import CBC_PARAMETER_NAMES, BNSSimulator
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import BNSSimulator as SimulatorsBNSSimulator


def test_simulate_returns_canonical_keys_and_shape() -> None:
    """BNSSimulator returns the full canonical CBC parameter surface in order."""
    simulator = BNSSimulator(seed=42)

    result = simulator.simulate(1000)

    assert list(result.keys()) == list(simulator.parameter_names)
    assert set(result.keys()) == set(CBC_PARAMETER_NAMES)
    assert all(array.shape == (1000,) for array in result.values())


def test_parameter_names_are_subset_of_cbc_parameter_names() -> None:
    """BNS parameters stay within the public CBC contract."""
    assert set(BNSSimulator().parameter_names).issubset(CBC_PARAMETER_NAMES)


def test_simulate_is_reproducible_for_fixed_seed() -> None:
    """Two simulators built with the same seed produce identical samples."""
    result_a = BNSSimulator(seed=42).simulate(256)
    result_b = BNSSimulator(seed=42).simulate(256)

    for name in BNSSimulator().parameter_names:
        np.testing.assert_array_equal(np.asarray(result_a[name]), np.asarray(result_b[name]))


def test_simulator_satisfies_protocol() -> None:
    """BNSSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(BNSSimulator(), GWPopSimulator)


def test_default_source_type_is_bns() -> None:
    """The BNS simulator routes to the BNS signal backend."""
    assert BNSSimulator().source_type == "bns"


def test_public_import_surfaces_export_bns_simulator() -> None:
    """The simulator is importable from the package and simulators namespace."""
    assert SimulatorsBNSSimulator is BNSSimulator


def test_component_masses_respect_bounds_ordering_and_total_mass_limit() -> None:
    """Mass samples stay in range, ordered, and obey an optional total-mass cap."""
    result = BNSSimulator(total_mass_max=4.2, seed=7).simulate(4096)
    mass_1 = np.asarray(result["detector_frame_mass_1"])
    mass_2 = np.asarray(result["detector_frame_mass_2"])

    assert np.all(mass_2 >= 1.0)
    assert np.all(mass_1 <= 3.0)
    assert np.all(mass_1 >= mass_2)
    assert np.all(mass_1 + mass_2 <= 4.2)


def test_spin_magnitudes_respect_configured_low_spin_bounds() -> None:
    """Default aligned spins stay within the configured low-spin interval."""
    result = BNSSimulator(chi_max=0.03, seed=9).simulate(2048)

    for name in ("spin_1x", "spin_1y", "spin_2x", "spin_2y"):
        np.testing.assert_array_equal(np.asarray(result[name]), np.zeros(2048))

    for component in (1, 2):
        magnitude = np.sqrt(
            np.asarray(result[f"spin_{component}x"]) ** 2
            + np.asarray(result[f"spin_{component}y"]) ** 2
            + np.asarray(result[f"spin_{component}z"]) ** 2
        )
        assert np.all(magnitude <= 0.03)


def test_tidal_params_in_parameter_names_and_within_default_bound() -> None:
    """lambda_1 and lambda_2 appear and lie within [0, lambda_max]."""
    simulator = BNSSimulator(seed=5)
    assert "lambda_1" in simulator.parameter_names
    assert "lambda_2" in simulator.parameter_names

    result = simulator.simulate(1000)
    for name in ("lambda_1", "lambda_2"):
        values = np.asarray(result[name])
        assert np.all(values >= 0.0)
        assert np.all(values <= 3000.0)


def test_lambda_max_clamps_tidal_output() -> None:
    """BNSSimulator(lambda_max=500) produces tidal values at most 500."""
    result = BNSSimulator(lambda_max=500.0, seed=6).simulate(1000)
    assert np.all(np.asarray(result["lambda_1"]) <= 500.0)
    assert np.all(np.asarray(result["lambda_2"]) <= 500.0)


def test_d_min_enforces_lower_distance_bound() -> None:
    """Sampled distances respect d_min when set."""
    result = BNSSimulator(d_min=100.0, d_max=500.0, seed=11).simulate(2048)
    distance = np.asarray(result["distance"])
    assert np.all(distance >= 100.0)
    assert np.all(distance <= 500.0)
