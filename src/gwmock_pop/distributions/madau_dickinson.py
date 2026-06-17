r"""Madau-Dickinson star-formation / merger-rate redshift model.

The rate shape follows the flexible "Madau-like" parametrization widely used in
gravitational-wave population analysis (e.g. arXiv:2111.03634), which reduces to
the Madau & Dickinson (2014, arXiv:1403.0007) cosmic star-formation history for
the default parameters:

.. math::

    \psi(z) = \mathcal{C}\,
        \frac{(1+z)^{\gamma}}{1 + \left(\frac{1+z}{1+z_p}\right)^{\gamma+\kappa}},
    \qquad
    \mathcal{C} = 1 + (1+z_p)^{-(\gamma+\kappa)},

so that ``psi(0) = 1`` and the absolute source-frame merger-rate density is
``R(z) = R_0 * psi(z)`` for a local rate ``R_0``. The redshift distribution of
sources observed over a fixed detector-frame interval carries a single factor of
``1 / (1 + z)`` (cosmological time dilation):

.. math::

    p(z) \propto \frac{\psi(z)}{1+z}\,\frac{\mathrm{d}V_c}{\mathrm{d}z}.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gwmock_pop.cosmology.flat_lambda_cdm import (
    DEFAULT_LOOKUP_GRID_SIZE,
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
    compute_differential_comoving_volume,
)

# Madau & Dickinson (2014) cosmic star-formation-rate parameters: psi ~
# (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6], i.e. gamma=2.7, gamma+kappa=5.6 (kappa=2.9),
# 1+z_peak=2.9 (z_peak=1.9).
MADAU_DICKINSON_GAMMA = 2.7
MADAU_DICKINSON_KAPPA = 2.9
MADAU_DICKINSON_Z_PEAK = 1.9


def madau_dickinson_rate(
    redshift: Array,
    gamma: float = MADAU_DICKINSON_GAMMA,
    kappa: float = MADAU_DICKINSON_KAPPA,
    z_peak: float = MADAU_DICKINSON_Z_PEAK,
) -> Array:
    """Return the dimensionless Madau-like rate shape ``psi(z)`` normalised to ``psi(0) = 1``.

    Args:
        redshift: Redshift(s) at which to evaluate the rate shape.
        gamma: Low-redshift power-law slope.
        kappa: High-redshift fall-off; the denominator exponent is ``gamma + kappa``.
        z_peak: Redshift of the rate peak (the turnover scale ``1 + z_peak``).

    Returns:
        The rate shape ``psi(z)``; multiply by a local rate ``R_0`` for the
        absolute source-frame merger-rate density.
    """
    one_plus_z = 1.0 + jnp.asarray(redshift)
    exponent = gamma + kappa
    normalization = 1.0 + (1.0 + z_peak) ** (-exponent)
    return normalization * one_plus_z**gamma / (1.0 + (one_plus_z / (1.0 + z_peak)) ** exponent)


def _redshift_weight(  # noqa: PLR0913
    redshift: Array,
    *,
    gamma: float,
    kappa: float,
    z_peak: float,
    hubble_constant: float,
    omega_m: float,
    n_grid: int,
) -> Array:
    """Return the unnormalised source redshift weight ``psi(z) / (1+z) * dVc/dz``."""
    rate = madau_dickinson_rate(redshift, gamma=gamma, kappa=kappa, z_peak=z_peak)
    differential_comoving_volume = compute_differential_comoving_volume(
        redshift=jnp.asarray(redshift), hubble_constant=hubble_constant, omega_m=omega_m, n_grid=n_grid
    )
    return rate / (1.0 + jnp.asarray(redshift)) * differential_comoving_volume


def madau_dickinson_redshift_pdf(  # noqa: PLR0913
    redshift: Array,
    *,
    z_max: float,
    z_min: float = 0.0,
    gamma: float = MADAU_DICKINSON_GAMMA,
    kappa: float = MADAU_DICKINSON_KAPPA,
    z_peak: float = MADAU_DICKINSON_Z_PEAK,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Return the normalised source redshift density on ``[z_min, z_max]``.

    The density is ``p(z) proportional to psi(z) / (1 + z) * dVc/dz``, normalised
    to integrate to one over ``[z_min, z_max]`` and zero outside it.

    Args:
        redshift: Redshift(s) at which to evaluate the density.
        z_max: Upper redshift bound of the population.
        z_min: Lower redshift bound of the population.
        gamma: Low-redshift power-law slope.
        kappa: High-redshift fall-off exponent offset.
        z_peak: Redshift of the rate peak.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        n_grid: Number of grid points used to normalise the density.

    Returns:
        Normalised probability density evaluated at ``redshift``.
    """
    if z_max <= z_min:
        raise ValueError("z_max must be strictly greater than z_min.")

    grid = jnp.linspace(z_min, z_max, n_grid)
    weight_grid = _redshift_weight(
        grid, gamma=gamma, kappa=kappa, z_peak=z_peak, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=n_grid
    )
    normalization = jnp.trapezoid(y=weight_grid, x=grid)

    redshift_array = jnp.asarray(redshift)
    weight = _redshift_weight(
        redshift_array,
        gamma=gamma,
        kappa=kappa,
        z_peak=z_peak,
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        n_grid=n_grid,
    )
    in_support = (redshift_array >= z_min) & (redshift_array <= z_max)
    return jnp.where(in_support, weight / normalization, 0.0)
