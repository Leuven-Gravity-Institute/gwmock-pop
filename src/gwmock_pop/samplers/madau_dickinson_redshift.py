"""Sampler for the Madau-Dickinson rate-weighted source redshift distribution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.cosmology.flat_lambda_cdm import (
    DEFAULT_LOOKUP_GRID_SIZE,
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
)
from gwmock_pop.distributions.madau_dickinson import (
    MADAU_DICKINSON_GAMMA,
    MADAU_DICKINSON_KAPPA,
    MADAU_DICKINSON_Z_PEAK,
    _redshift_weight,
)


def madau_dickinson_redshift(  # noqa: PLR0913
    key: Array,
    n_samples: int,
    z_max: float,
    *,
    z_min: float = 0.0,
    gamma: float = MADAU_DICKINSON_GAMMA,
    kappa: float = MADAU_DICKINSON_KAPPA,
    z_peak: float = MADAU_DICKINSON_Z_PEAK,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Draw source redshifts from the Madau-Dickinson rate-weighted distribution.

    Samples ``p(z) proportional to psi(z) / (1 + z) * dVc/dz`` on ``[z_min, z_max]``
    by inverse-transform sampling on a redshift grid, where ``psi(z)`` is the
    Madau-like rate shape and the single ``1 / (1 + z)`` factor converts the
    source-frame rate to the detector frame.

    Args:
        key: JAX PRNG key.
        n_samples: Number of redshifts to sample.
        z_max: Maximum redshift.
        z_min: Minimum redshift (default 0).
        gamma: Low-redshift power-law slope.
        kappa: High-redshift fall-off exponent offset.
        z_peak: Redshift of the rate peak.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        n_grid: Number of grid points for the inverse-CDF lookup.

    Returns:
        Sampled redshifts of shape ``(n_samples,)``.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if z_min < 0.0:
        raise ValueError("z_min must be non-negative.")
    if z_max <= z_min:
        raise ValueError("z_max must be strictly greater than z_min.")

    redshift_grid = jnp.linspace(z_min, z_max, n_grid)
    weight_grid = _redshift_weight(
        redshift_grid,
        gamma=gamma,
        kappa=kappa,
        z_peak=z_peak,
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        n_grid=n_grid,
    )

    delta_redshift = jnp.diff(redshift_grid)
    trapezoids = 0.5 * (weight_grid[1:] + weight_grid[:-1]) * delta_redshift
    cumulative = jnp.concatenate([jnp.zeros(1, dtype=trapezoids.dtype), jnp.cumsum(trapezoids)])
    cdf_grid = cumulative / cumulative[-1]

    uniform = jax.random.uniform(key, shape=(n_samples,))
    return jnp.interp(uniform, cdf_grid, redshift_grid)
