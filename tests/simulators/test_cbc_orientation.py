"""Orientation and reference-frequency columns of the graph-backed CBC simulators.

For a non-precessing binary the total angular momentum ``J`` is parallel to the
orbital angular momentum ``L``, so ``theta_jn`` is the inclination itself, and no
column depends on a reference frequency. For a precessing binary ``theta_jn``
follows from the inclination, the spins and the reference frequency; it is not
an independent angle, so the simulators do not write one.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_pop import BBHSimulator, BNSSimulator, CBCSimulator, NSBHSimulator

_CLASSES = (CBCSimulator, BBHSimulator, BNSSimulator, NSBHSimulator)
_SEEDS = (0, 1, 7, 42)
_N_SAMPLES = 256
_IN_PLANE_SPINS = ("spin_1x", "spin_1y", "spin_2x", "spin_2y")


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_aligned_spin_draws_write_theta_jn_equal_to_inclination(simulator_class: type, seed: int) -> None:
    """An aligned-spin draw writes ``theta_jn`` equal to ``inclination``."""
    result = simulator_class(aligned_spins=True, seed=seed).simulate(_N_SAMPLES)

    for name in _IN_PLANE_SPINS:
        np.testing.assert_array_equal(np.asarray(result[name]), np.zeros(_N_SAMPLES))
    np.testing.assert_array_equal(np.asarray(result["theta_jn"]), np.asarray(result["inclination"]))


@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_aligned_spin_draws_write_no_reference_frequency(simulator_class: type) -> None:
    """An aligned-spin draw has nothing defined at a reference frequency, so writes no ``f_ref``."""
    simulator = simulator_class(aligned_spins=True, seed=0)

    assert "f_ref" not in simulator.parameter_names
    assert "f_ref" not in simulator.simulate(8)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_precessing_draws_write_no_independent_theta_jn(simulator_class: type, seed: int) -> None:
    """A precessing draw writes no ``theta_jn``: an independent angle would describe another binary."""
    simulator = simulator_class(aligned_spins=False, seed=seed)
    result = simulator.simulate(_N_SAMPLES)

    assert any(bool(np.any(np.asarray(result[name]) != 0.0)) for name in _IN_PLANE_SPINS)
    assert "inclination" in result
    assert "theta_jn" not in simulator.parameter_names
    assert "theta_jn" not in result


@pytest.mark.parametrize("f_ref", [5.0, 20.0])
@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_precessing_draws_write_the_configured_reference_frequency(simulator_class: type, f_ref: float) -> None:
    """A precessing draw writes the reference frequency its spins are defined at."""
    result = simulator_class(aligned_spins=False, f_ref=f_ref, seed=0).simulate(16)

    np.testing.assert_array_equal(np.asarray(result["f_ref"]), np.full(16, f_ref))
