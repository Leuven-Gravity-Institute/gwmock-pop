"""The spin-angle conversion against LALSimulation, which it ports."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_pop.catalogue.spins import spin_angles_to_components

pytestmark = pytest.mark.unit


def _random_inputs(count: int, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mass_1 = rng.uniform(1.0, 300.0, count)
    return {
        "theta_jn": np.arccos(rng.uniform(-1, 1, count)),
        "phi_jl": rng.uniform(0, 2 * np.pi, count),
        "tilt_1": np.arccos(rng.uniform(-1, 1, count)),
        "tilt_2": np.arccos(rng.uniform(-1, 1, count)),
        "phi_12": rng.uniform(0, 2 * np.pi, count),
        "spin_magnitude_1": rng.uniform(0, 0.99, count),
        "spin_magnitude_2": rng.uniform(0, 0.99, count),
        "mass_1": mass_1,
        "mass_2": mass_1 * rng.uniform(0.05, 1.0, count),
        "reference_frequency": rng.choice([2.0, 5.0, 20.0], count),
        "reference_phase": rng.uniform(0, 2 * np.pi, count),
    }


def test_it_reproduces_lalsimulation():
    """Every output, for random binaries over wide mass and spin ranges, to 1e-10."""
    lal = pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    inputs = _random_inputs(500)
    ours = np.array(spin_angles_to_components(**inputs))
    for index in range(500):
        expected = lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
            inputs["theta_jn"][index],
            inputs["phi_jl"][index],
            inputs["tilt_1"][index],
            inputs["tilt_2"][index],
            inputs["phi_12"][index],
            inputs["spin_magnitude_1"][index],
            inputs["spin_magnitude_2"][index],
            inputs["mass_1"][index] * lal.MSUN_SI,
            inputs["mass_2"][index] * lal.MSUN_SI,
            inputs["reference_frequency"][index],
            inputs["reference_phase"][index],
        )
        np.testing.assert_allclose(ours[:, index], expected, rtol=0.0, atol=1e-10)


def test_without_spin_the_inclination_is_theta_jn():
    """With no spin, J is along L, so the line of sight makes the same angle with both."""
    inputs = _random_inputs(50)
    inputs["spin_magnitude_1"] = np.zeros(50)
    inputs["spin_magnitude_2"] = np.zeros(50)
    inclination, *spins = spin_angles_to_components(**inputs)
    np.testing.assert_allclose(inclination, inputs["theta_jn"], rtol=0.0, atol=1e-12)
    assert all(np.all(component == 0.0) for component in spins)


def test_spin_magnitudes_and_tilts_are_kept():
    """A rotation moves the spins but keeps each magnitude and its angle to L."""
    inputs = _random_inputs(200)
    _, s1x, s1y, s1z, s2x, s2y, s2z = spin_angles_to_components(**inputs)
    np.testing.assert_allclose(np.sqrt(s1x**2 + s1y**2 + s1z**2), inputs["spin_magnitude_1"], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.sqrt(s2x**2 + s2y**2 + s2z**2), inputs["spin_magnitude_2"], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(s1z, inputs["spin_magnitude_1"] * np.cos(inputs["tilt_1"]), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(s2z, inputs["spin_magnitude_2"] * np.cos(inputs["tilt_2"]), rtol=0.0, atol=1e-12)
