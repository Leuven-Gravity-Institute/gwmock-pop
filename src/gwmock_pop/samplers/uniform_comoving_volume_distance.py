"""Distance samplers for cosmological priors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.cosmology.flat_lambda_cdm import (
    DEFAULT_LOOKUP_GRID_SIZE,
    DEFAULT_MAX_REDSHIFT,
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
    build_distance_lookup,
)


def uniform_comoving_volume_distance(  # noqa: PLR0913
    key: Array,
    n_samples: int,
    d_max: float,
    *,
    d_min: float = 0.0,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    max_redshift: float = DEFAULT_MAX_REDSHIFT,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Draw luminosity distances uniformly in comoving volume.

    Args:
        key: JAX PRNG key.
        n_samples: Number of distances to sample.
        d_max: Maximum luminosity distance in Mpc.
        d_min: Minimum luminosity distance in Mpc (default 0).
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift supported by the lookup table.
        n_grid: Number of tabulation points.

    Returns:
        Sampled luminosity distances in Mpc.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if d_min < 0.0:
        raise ValueError("d_min must be non-negative.")
    if d_max <= 0.0:
        raise ValueError("d_max must be positive.")
    if d_min >= d_max:
        raise ValueError("d_min must be less than d_max.")

    redshift_grid, comoving_distance_grid, luminosity_distance_grid = build_distance_lookup(
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
    max_supported_distance = float(luminosity_distance_grid[-1])
    if d_max > max_supported_distance:
        raise ValueError(f"d_max={d_max} exceeds the luminosity-distance lookup range for max_redshift={max_redshift}.")

    comoving_distance_max = jnp.interp(jnp.asarray(d_max), luminosity_distance_grid, comoving_distance_grid)
    comoving_distance_min = jnp.interp(jnp.asarray(d_min), luminosity_distance_grid, comoving_distance_grid)
    u = jax.random.uniform(key, shape=(n_samples,))
    sampled_comoving_distance = jnp.cbrt(
        u * (comoving_distance_max**3 - comoving_distance_min**3) + comoving_distance_min**3
    )
    sampled_redshift = jnp.interp(sampled_comoving_distance, comoving_distance_grid, redshift_grid)
    return sampled_comoving_distance * (1.0 + sampled_redshift)
