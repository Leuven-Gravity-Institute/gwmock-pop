"""Basic transform helpers for graph-based simulator configs."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gwmock_pop.cosmology.flat_lambda_cdm import (
    DEFAULT_LOOKUP_GRID_SIZE,
    DEFAULT_MAX_REDSHIFT,
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
    compute_redshift_from_luminosity_distance,
)


def constant_like(reference: Array, value: float) -> Array:
    """Return a constant array with the same shape as ``reference``.

    Args:
        reference: Array providing the target shape.
        value: Scalar value to broadcast across the output.

    Returns:
        An array with the same shape as ``reference`` filled with ``value``.
    """
    dtype = jnp.result_type(reference, jnp.asarray(value))
    return jnp.full(shape=reference.shape, fill_value=value, dtype=dtype)


def multiply(left: Array, right: Array) -> Array:
    """Multiply two array-like inputs elementwise.

    Args:
        left: Left multiplicand.
        right: Right multiplicand.

    Returns:
        Elementwise product of ``left`` and ``right``.
    """
    return jnp.asarray(left) * jnp.asarray(right)


def comoving_distance_to_redshift(
    distance: Array,
    *,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    max_redshift: float = DEFAULT_MAX_REDSHIFT,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Convert luminosity distance to redshift with a flat-ΛCDM lookup.

    The historical transform name is retained for graph-config compatibility even
    though the input quantity is luminosity distance.

    Args:
        distance: Luminosity distance in Mpc.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift supported by the lookup table.
        n_grid: Number of tabulation points.

    Returns:
        Redshift inferred from ``distance``.
    """
    return compute_redshift_from_luminosity_distance(
        luminosity_distance=distance,
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
