"""Functions to compute cosmological quantities under flat lambda CDM model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array

from gwmock_pop.constants import SPEED_OF_LIGHT

PLANCK18_H0_KM_S_MPC = 67.66
PLANCK18_OMEGA_M = 0.3111
DEFAULT_MAX_REDSHIFT = 10.0
DEFAULT_LOOKUP_GRID_SIZE = 4096
MIN_LOOKUP_GRID_SIZE = 3
MIN_LOOKUP_NONZERO_REDSHIFT = 1e-6


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

    The redshift grid is ``0`` followed by ``n_grid - 1`` points spaced uniformly in
    ``log z`` from ``MIN_LOOKUP_NONZERO_REDSHIFT`` to ``max_redshift``. A grid uniform
    in ``z`` would leave a first interval so wide that linear interpolation misstates
    ``d_L`` there by about ``(1 - q0) / 2 * dz``; the logarithmic grid keeps the
    relative interpolation error small down to ``MIN_LOOKUP_NONZERO_REDSHIFT``.

    Args:
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift tabulated by the lookup.
        n_grid: Number of tabulation points.

    Returns:
        Tuple ``(redshift_grid, comoving_distance_grid, luminosity_distance_grid)``.
    """
    if max_redshift <= MIN_LOOKUP_NONZERO_REDSHIFT:
        raise ValueError(f"max_redshift must be greater than {MIN_LOOKUP_NONZERO_REDSHIFT}.")
    if n_grid < MIN_LOOKUP_GRID_SIZE:
        raise ValueError(f"n_grid must be at least {MIN_LOOKUP_GRID_SIZE}.")

    redshift_grid = jnp.concatenate(
        [jnp.zeros(1), jnp.geomspace(MIN_LOOKUP_NONZERO_REDSHIFT, max_redshift, n_grid - 1)]
    )
    # geomspace can round its endpoint (e.g. to 9.999999999999998 for 10), which would
    # put z = max_redshift itself outside the table.
    redshift_grid = redshift_grid.at[-1].set(max_redshift)
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


def interpolate_within_lookup(values: Array, grid: Array, table: Array, name: str) -> Array:
    """Linearly interpolate ``table`` over ``grid`` at ``values``, refusing to extrapolate.

    Interpolating outside ``[0, grid[-1]]`` would silently return the table edge.
    Concrete (eager) input outside that range raises ``ValueError``. Traced input
    (under ``jax.jit``, ``jax.vmap``, ``jax.lax.scan``, ...) cannot raise on a
    data-dependent condition, so there each out-of-range element is returned as
    NaN instead; it does not raise. In-range elements are unaffected either way.

    Args:
        values: Values to be looked up.
        grid: Increasing tabulation points, starting at 0.
        table: Tabulated values at ``grid``.
        name: Name of the looked-up quantity, used in the error message.

    Returns:
        Interpolated values, NaN where a traced value is out of range.

    Raises:
        ValueError: If concrete ``values`` contain an element below 0 or above ``grid[-1]``.
    """
    values = jnp.asarray(values)
    upper = grid[-1]
    out_of_range = (values < 0.0) | (values > upper)
    if not isinstance(out_of_range, jax.core.Tracer) and jnp.any(out_of_range):
        raise ValueError(
            f"{name} outside the lookup range [0, {float(upper)}]; got values in "
            f"[{float(jnp.min(values))}, {float(jnp.max(values))}]. Raise max_redshift to extend the table."
        )
    return jnp.where(out_of_range, jnp.nan, jnp.interp(values, grid, table))


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

    Raises:
        ValueError: If any concrete distance lies outside ``[0, d_L(max_redshift)]``.
            Traced out-of-range distances are returned as NaN instead; see
            :func:`interpolate_within_lookup`.
    """
    redshift_grid, _, luminosity_distance_grid = build_distance_lookup(
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
    return interpolate_within_lookup(
        luminosity_distance, luminosity_distance_grid, redshift_grid, "luminosity_distance"
    )
