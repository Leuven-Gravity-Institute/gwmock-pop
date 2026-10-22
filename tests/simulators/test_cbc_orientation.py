"""Orientation and reference-frequency columns of the graph-backed CBC simulators.

The orientation of a binary is written as ``inclination`` alone, in both spin
modes. ``theta_jn`` follows from the inclination, the spins and the reference
frequency; with anti-aligned spins whose angular momentum exceeds the orbital
one, ``J`` is anti-parallel to ``L`` and ``theta_jn`` is ``pi - inclination``.
It is not written. Only precessing draws depend on a reference frequency, so
only they write ``f_ref``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_pop import BBHSimulator, BNSSimulator, CBCSimulator, NSBHSimulator

_CLASSES = (CBCSimulator, BBHSimulator, BNSSimulator, NSBHSimulator)
_N_SAMPLES = 256
_IN_PLANE_SPINS = ("spin_1x", "spin_1y", "spin_2x", "spin_2y")


@pytest.mark.parametrize("aligned_spins", [True, False], ids=["aligned", "precessing"])
@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_draws_write_no_theta_jn(simulator_class: type, aligned_spins: bool) -> None:
    """Neither spin mode writes ``theta_jn``; ``inclination`` is the orientation."""
    simulator = simulator_class(aligned_spins=aligned_spins, seed=0)
    result = simulator.simulate(_N_SAMPLES)

    assert "inclination" in result
    assert "theta_jn" not in simulator.parameter_names
    assert "theta_jn" not in result


@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_aligned_spin_draws_write_no_reference_frequency(simulator_class: type) -> None:
    """An aligned-spin draw has nothing defined at a reference frequency, so writes no ``f_ref``."""
    simulator = simulator_class(aligned_spins=True, seed=0)

    assert "f_ref" not in simulator.parameter_names
    assert "f_ref" not in simulator.simulate(8)


@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_aligned_spin_draws_have_no_in_plane_spin_and_precessing_draws_do(simulator_class: type) -> None:
    """The spin modes the reference-frequency tests rely on are the ones drawn."""
    aligned = simulator_class(aligned_spins=True, seed=0).simulate(_N_SAMPLES)
    precessing = simulator_class(aligned_spins=False, seed=0).simulate(_N_SAMPLES)

    for name in _IN_PLANE_SPINS:
        np.testing.assert_array_equal(np.asarray(aligned[name]), np.zeros(_N_SAMPLES))
    assert any(bool(np.any(np.asarray(precessing[name]) != 0.0)) for name in _IN_PLANE_SPINS)


@pytest.mark.parametrize("f_ref", [5.0, 20.0])
@pytest.mark.parametrize("simulator_class", _CLASSES, ids=lambda cls: cls.__name__)
def test_precessing_draws_write_the_configured_reference_frequency(simulator_class: type, f_ref: float) -> None:
    """A precessing draw writes the reference frequency its spins are defined at."""
    result = simulator_class(aligned_spins=False, f_ref=f_ref, seed=0).simulate(16)

    np.testing.assert_array_equal(np.asarray(result["f_ref"]), np.full(16, f_ref))
