"""Functions to compute cosmological quantities under flat lambda CDM model."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array

from gwsim_pop.constants import SPEED_OF_LIGHT


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
