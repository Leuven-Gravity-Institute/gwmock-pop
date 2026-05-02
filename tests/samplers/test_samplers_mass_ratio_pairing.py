"""Tests for the conditional mass-ratio pairing sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gwmock_pop.samplers import mass_ratio_pairing


def test_mass_ratio_pairing_returns_one_ratio_per_primary_mass() -> None:
    """Array-valued primary masses produce one conditional ratio each."""
    primary_mass = jnp.array([10.0, 20.0, 40.0, 80.0])

    mass_ratio = mass_ratio_pairing(
        key=jax.random.PRNGKey(42),
        primary_mass=primary_mass,
        beta=1.26,
        secondary_minimum=4.59,
    )

    secondary_mass = primary_mass * mass_ratio

    assert mass_ratio.shape == primary_mass.shape
    assert bool(jnp.all(mass_ratio <= 1.0))
    assert bool(jnp.all(secondary_mass >= 4.59))


def test_mass_ratio_pairing_supports_scalar_primary_mass_with_n_samples() -> None:
    """Scalar primary masses can be expanded with ``n_samples``."""
    mass_ratio = mass_ratio_pairing(
        key=jax.random.PRNGKey(11),
        primary_mass=jnp.asarray(30.0),
        beta=1.26,
        secondary_minimum=4.59,
        n_samples=128,
    )

    assert mass_ratio.shape == (128,)
    assert bool(jnp.all(mass_ratio >= 4.59 / 30.0))
    assert bool(jnp.all(mass_ratio <= 1.0))
