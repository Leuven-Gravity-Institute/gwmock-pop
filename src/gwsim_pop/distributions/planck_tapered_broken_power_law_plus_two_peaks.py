"""Broken power law plus two peaks distribution with Planck tapering."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.nn import logsumexp
from jax.scipy.stats.norm import logpdf as norm_logpdf

from gwsim_pop.distributions.broken_power_law import broken_power_law_unnormalized_logpdf
from gwsim_pop.distributions.smoothing import log_planck_tapering_function
from gwsim_pop.integrators.trapezoid import log_trapezoidal_cumsum


def planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(  # noqa: PLR0913
    x: Array,
    alpha_1: float,
    alpha_2: float,
    transition: float,
    minimum: float,
    maximum: float,
    mean_1: float,
    sigma_1: float,
    mean_2: float,
    sigma_2: float,
    taper_range: float,
    lambda_0: float,
    lambda_1: float,
) -> Array:
    """Compute the unnormalized logpdf of Planck tapered broken power law plus two peaks distribution.

    Args:
        x: An array of values to evaluate the unnormalized pdf.
        alpha_1: Spectral index of the first power law.
        alpha_2: Spectral index of the second power law.
        transition: Transition point of the broken power law.
        minimum: Minimum of the distribution.
        maximum: Maximum of the distribution.
        mean_1: Mean of the first peak.
        sigma_1: Standard deviation of the first peak.
        mean_2: Mean of the second peak.
        sigma_2: Standard deviation of the second peak.
        taper_range: Range of the tapering.
        lambda_0: Mixing fraction of the power law.
        lambda_1: Mixing fraction of the first peak.

    Returns:
        Unnormalized logpdf.
    """
    # Evaluate the logpdf of the broken power low
    broken_power_law_logpdf = broken_power_law_unnormalized_logpdf(
        x=x, x_min=minimum, x_max=maximum, transition=transition, alpha_1=alpha_1, alpha_2=alpha_2
    )

    # Evaluate the logpdf of the two normal peaks
    first_peak_logpdf = norm_logpdf(x=x, loc=mean_1, scale=sigma_1)
    second_peak_logpdf = norm_logpdf(x=x, loc=mean_2, scale=sigma_2)

    # Tapering function
    log_smoothing = log_planck_tapering_function(x=x, x_min=minimum, delta=taper_range)

    # Weights
    checkify.check(lambda_0 >= 0.0, "Invalid mixture weights: require lambda_0 >=0.")
    checkify.check(lambda_1 >= 0.0, "Invalid mixture weights: require lambda_1 >=0.")
    checkify.check(lambda_0 + lambda_1 <= 1, "Invalid mixture weights: require lambda_0 + lambda_1 <= 1.")
    weights = jnp.array([lambda_0, lambda_1, 1.0 - lambda_0 - lambda_1])

    # Stacked logpdf of the components
    stacked_logpdf = jnp.stack([broken_power_law_logpdf, first_peak_logpdf, second_peak_logpdf])

    return logsumexp(a=stacked_logpdf, b=weights[:, None], axis=0) + log_smoothing


def planck_tapered_broken_power_law_plus_two_peaks_cdf(  # noqa: PLR0913
    alpha_1: float,
    alpha_2: float,
    transition: float,
    minimum: float,
    maximum: float,
    mean_1: float,
    sigma_1: float,
    mean_2: float,
    sigma_2: float,
    taper_range: float,
    lambda_0: float,
    lambda_1: float,
    n_grids: int,
) -> tuple[Array, Array]:
    """Compute the cdf of Planck tapered broken power law plus two peaks distribution.

    Args:
        alpha_1: Spectral index of the first power law.
        alpha_2: Spectral index of the second power law.
        transition: Transition point of the broken power law.
        minimum: Minimum of the distribution.
        maximum: Maximum of the distribution.
        mean_1: Mean of the first peak.
        sigma_1: Standard deviation of the first peak.
        mean_2: Mean of the second peak.
        sigma_2: Standard deviation of the second peak.
        taper_range: Range of the tapering.
        lambda_0: Mixing fraction of the power law.
        lambda_1: Mixing fraction of the first peak.
        n_grids: Number of grid points to evaluate the CDF.

    Returns:
        x: Grid points to evaluate the CDF.
        cdf: Cumulative distribution function.
    """
    # Create an array of grid points to evaluate the CDF
    x = jnp.linspace(start=minimum, stop=maximum, num=n_grids)

    # Compute the unnormalized logpdf
    unnormalized_logpdf = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
        x=x,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        transition=transition,
        minimum=minimum,
        maximum=maximum,
        mean_1=mean_1,
        sigma_1=sigma_1,
        mean_2=mean_2,
        sigma_2=sigma_2,
        taper_range=taper_range,
        lambda_0=lambda_0,
        lambda_1=lambda_1,
    )

    # Compute the logcdf
    logcdf = log_trapezoidal_cumsum(log_y=unnormalized_logpdf, x=x)

    # Compute the cdf
    return x, jnp.exp(logcdf - logcdf[-1])
