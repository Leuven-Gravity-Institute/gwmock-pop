"""Functions to compute cosmological quantities under flat lambda CDM model."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array

from gwmock_pop.constants import SPEED_OF_LIGHT

PLANCK18_H0_KM_S_MPC = 67.66
PLANCK18_OMEGA_M = 0.3111
DEFAULT_MAX_REDSHIFT = 10.0
DEFAULT_LOOKUP_GRID_SIZE = 4096
MIN_LOOKUP_GRID_SIZE = 2


def compute_normalized_hubble_parameter(redshift: Array, omega_m: Array) -> Array:
    """Compute the normalized Hubble parameter.

    Args:
        redshift: Redshift.
        omega_m: Matter density.

    Returns:
        Normalized Hubble parameter.
    """
    return jnp.sqrt(omega_m * (1.0 + redshift) ** 3 + (1.0 - omega_m))


def compute_hubble_parameter(redshift: Array, hubble_constant: Array, omega_m: Array) -> Array:
    """Compute the Hubble parameter.

    Args:
        redshift: Redshift.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.

    Returns:
        Hubble parameter in km / s/ Mpc.
    """
    return hubble_constant * compute_normalized_hubble_parameter(redshift=redshift, omega_m=omega_m)


def compute_comoving_distance(redshift: Array, hubble_constant: Array, omega_m: Array, n_grid: int = 1000) -> Array:
    """Compute the comoving distance.

    Args:
        redshift: Redshift to evaluate the comoving distance.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        n_grid: Number of grid points to perform the numerical integration.

    Returns:
        Comoving distance in Mpc.
    """

    def compute_integrand(redshift_array: Array) -> Array:
        """Compute the integrand for calculating the comoving distance.

        Args:
            redshift_array: An array of redshift.

        Returns:
            Integrand.
        """
        return 1.0 / compute_normalized_hubble_parameter(redshift=redshift_array, omega_m=omega_m)

    z_grid = jnp.linspace(start=0.0, stop=redshift, num=n_grid)

    integrand = compute_integrand(z_grid)

    return SPEED_OF_LIGHT / 1000 / hubble_constant * jscipy.integrate.trapezoid(y=integrand, x=z_grid, axis=0)


def compute_differential_comoving_volume(
    redshift: Array, hubble_constant: Array, omega_m: Array, n_grid: int = 1000
) -> Array:
    """Compute the differential comoving volume.

    Args:
        redshift: Redshift.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        n_grid: Number of grid points to perform the numerical integration.

    Returns:
        Differential comoving volume in Mpc^{3}.
    """
    comoving_distance = compute_comoving_distance(
        redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=n_grid
    )
    hubble_parameter = compute_hubble_parameter(redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m)
    return 4 * jnp.pi * comoving_distance**2 / hubble_parameter * SPEED_OF_LIGHT / 1000


def compute_luminosity_distance(redshift: Array, hubble_constant: Array, omega_m: Array, n_grid: int = 1000) -> Array:
    """Compute the luminosity distance.

    Args:
        redshift: Redshift to evaluate the luminosity distance.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        n_grid: Number of grid points to perform the numerical integration.

    Returns:
        Luminosity distance in Mpc.
    """
    return (1.0 + redshift) * compute_comoving_distance(
        redshift=redshift,
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        n_grid=n_grid,
    )


def build_distance_lookup(
    *,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    max_redshift: float = DEFAULT_MAX_REDSHIFT,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> tuple[Array, Array, Array]:
    """Build flat-ΛCDM lookup tables for redshift, comoving distance, and luminosity distance.

    Args:
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift tabulated by the lookup.
        n_grid: Number of tabulation points.

    Returns:
        Tuple ``(redshift_grid, comoving_distance_grid, luminosity_distance_grid)``.
    """
    if max_redshift <= 0.0:
        raise ValueError("max_redshift must be positive.")
    if n_grid < MIN_LOOKUP_GRID_SIZE:
        raise ValueError(f"n_grid must be at least {MIN_LOOKUP_GRID_SIZE}.")

    redshift_grid = jnp.linspace(0.0, max_redshift, n_grid)
    inv_e = 1.0 / compute_normalized_hubble_parameter(
        redshift=redshift_grid,
        omega_m=jnp.asarray(omega_m),
    )
    delta_redshift = jnp.diff(redshift_grid)
    trapezoids = 0.5 * (inv_e[1:] + inv_e[:-1]) * delta_redshift
    integral = jnp.concatenate([jnp.zeros(1, dtype=trapezoids.dtype), jnp.cumsum(trapezoids)])
    comoving_distance_grid = SPEED_OF_LIGHT / 1000 / jnp.asarray(hubble_constant) * integral
    luminosity_distance_grid = (1.0 + redshift_grid) * comoving_distance_grid
    return redshift_grid, comoving_distance_grid, luminosity_distance_grid


def compute_redshift_from_luminosity_distance(
    luminosity_distance: Array,
    *,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    max_redshift: float = DEFAULT_MAX_REDSHIFT,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Invert a luminosity distance to redshift via a flat-ΛCDM lookup table.

    Args:
        luminosity_distance: Luminosity distance in Mpc.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift tabulated by the lookup.
        n_grid: Number of tabulation points.

    Returns:
        Redshift inferred from ``luminosity_distance``.
    """
    redshift_grid, _, luminosity_distance_grid = build_distance_lookup(
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
    return jnp.interp(
        jnp.asarray(luminosity_distance),
        luminosity_distance_grid,
        redshift_grid,
        left=0.0,
        right=max_redshift,
    )
