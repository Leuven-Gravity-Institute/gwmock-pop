"""Conditional ratio power law distribution with Planck tapering."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gwsim_pop.distributions.smoothing import log_planck_tapering_function
from gwsim_pop.integrators.trapezoid import log_trapezoidal_cumsum


def planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
    x: Array, denominator: Array, beta: float, numerator_minimum: float, taper_range: float
) -> Array:
    """Compute the unnormalized logpdf of Planck tapered conditional ratio power law distribution.

    Args:
        x: An array of values to evaluate the unnormalized logpdf.
        denominator: Denominator.
        beta: Spectral index of the ratio.
        numerator_minimum: Minimum of the numerator.
        taper_range: Range of the tapering.

    Returns:
        Unnormalized logpdf.
    """
    return beta * jnp.log(x) + log_planck_tapering_function(
        x=denominator * x, x_min=numerator_minimum, delta=taper_range
    )


def planck_tapered_conditional_ratio_power_law_cdf(  # noqa: PLR0913
    denominator: Array,
    beta: float,
    numerator_minimum: float,
    taper_range: float,
    minimum: float,
    maximum: float,
    n_grids: int,
) -> tuple[Array, Array]:
    """Compute the cdf of Planck tapered conditional ratio power law distribution.

    Args:
        denominator: Denominator.
        beta: Spectral index of the ratio.
        numerator_minimum: Minimum of the numerator.
        taper_range: Range of the tapering.
        minimum: Minimum of the ratio.
        maximum: Maximum of the ratio.
        n_grids: Number of grid points to evaluate the CDF.

    Returns:
        x: Grid points to evaluate the CDF.
        cdf: Cumulative distribution function.
    """
    # Create an array to evaluate the
    x = jnp.linspace(start=minimum, stop=maximum, num=n_grids)

    # Compute the unnormalized logpdf
    unnormalized_logpdf = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
        x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
    )

    # Compute the logcdf
    logcdf = log_trapezoidal_cumsum(log_y=unnormalized_logpdf, x=x)

    # Compute the cdf
    return x, jnp.exp(logcdf - logcdf[-1])
