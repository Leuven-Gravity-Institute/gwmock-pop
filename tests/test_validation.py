"""Tests for physical sample validation helpers."""

from __future__ import annotations

import math

import jax.numpy as jnp

from gwmock_pop import CBCSimulator, validate_sample


def test_validate_sample_accepts_cbc_prior_output() -> None:
    """CBCSimulator outputs satisfy the physical-consistency checks."""
    sample = CBCSimulator(seed=0).simulate(1000)

    assert validate_sample(sample) == []


def test_validate_sample_reports_m1_m2_ordering_violation() -> None:
    """Mass-ordering failures mention both m1 and m2."""
    violations = validate_sample({"m1": jnp.asarray([10.0]), "m2": jnp.asarray([20.0])})

    assert any("m1" in violation and "m2" in violation for violation in violations)


def test_validate_sample_reports_spin_1z_violation() -> None:
    """Spin-z bound failures mention the offending parameter."""
    violations = validate_sample({"spin_1z": jnp.asarray([1.2])})

    assert any("spin_1z" in violation for violation in violations)


def test_validate_sample_reports_all_supported_constraint_failures() -> None:
    """Each requested physical constraint produces a readable violation."""
    sample = {
        "m1": jnp.asarray([0.0]),
        "m2": jnp.asarray([-1.0]),
        "spin_1z": jnp.asarray([1.1]),
        "spin_2z": jnp.asarray([-1.2]),
        "distance": jnp.asarray([0.0]),
        "redshift": jnp.asarray([-0.1]),
        "right_ascension": jnp.asarray([2.0 * math.pi + 0.1]),
        "declination": jnp.asarray([0.5 * math.pi + 0.1]),
    }

    violations = validate_sample(sample)

    assert any("m1" in violation and "must be > 0" in violation for violation in violations)
    assert any("m2" in violation and "must be > 0" in violation for violation in violations)
    assert any("spin_1z" in violation for violation in violations)
    assert any("spin_2z" in violation for violation in violations)
    assert any("distance" in violation for violation in violations)
    assert any("redshift" in violation for violation in violations)
    assert any("right_ascension" in violation for violation in violations)
    assert any("declination" in violation for violation in violations)


def test_validate_sample_is_importable_from_top_level_package() -> None:
    """The validation helper is exported from ``gwmock_pop``."""
    assert callable(validate_sample)
