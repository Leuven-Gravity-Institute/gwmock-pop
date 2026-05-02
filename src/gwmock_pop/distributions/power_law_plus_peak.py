"""Power-law-plus-peak primary-mass distribution helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.special import erf

_SQRT_TWO = jnp.sqrt(2.0)
_SQRT_TWO_PI = jnp.sqrt(2.0 * jnp.pi)


def _standard_normal_cdf(x: Array) -> Array:
    """Return the standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / _SQRT_TWO))


def _power_law_pdf(mass: Array, alpha: float, minimum: float, maximum: float) -> Array:
    """Return the normalized power-law density on ``[minimum, maximum]``."""
    support = (mass >= minimum) & (mass <= maximum)
    if jnp.isclose(alpha, 1.0):
        normalization = jnp.log(maximum / minimum)
        density = 1.0 / (mass * normalization)
    else:
        exponent = 1.0 - alpha
        normalization = maximum**exponent - minimum**exponent
        density = exponent * mass ** (-alpha) / normalization
    return jnp.where(support, density, 0.0)


def _power_law_cdf(mass: Array, alpha: float, minimum: float, maximum: float) -> Array:
    """Return the normalized power-law CDF on ``[minimum, maximum]``."""
    clipped_mass = jnp.clip(mass, minimum, maximum)
    if jnp.isclose(alpha, 1.0):
        normalization = jnp.log(maximum / minimum)
        cdf = jnp.log(clipped_mass / minimum) / normalization
    else:
        exponent = 1.0 - alpha
        normalization = maximum**exponent - minimum**exponent
        cdf = (clipped_mass**exponent - minimum**exponent) / normalization
    return jnp.where(mass < minimum, 0.0, jnp.where(mass > maximum, 1.0, cdf))


def _truncated_normal_pdf(mass: Array, mean: float, sigma: float, minimum: float, maximum: float) -> Array:
    """Return the normalized truncated-normal density on ``[minimum, maximum]``."""
    support = (mass >= minimum) & (mass <= maximum)
    lower = (minimum - mean) / sigma
    upper = (maximum - mean) / sigma
    normalization = _standard_normal_cdf(upper) - _standard_normal_cdf(lower)
    z = (mass - mean) / sigma
    density = jnp.exp(-0.5 * z**2) / (sigma * _SQRT_TWO_PI * normalization)
    return jnp.where(support, density, 0.0)


def _truncated_normal_cdf(mass: Array, mean: float, sigma: float, minimum: float, maximum: float) -> Array:
    """Return the normalized truncated-normal CDF on ``[minimum, maximum]``."""
    clipped_mass = jnp.clip(mass, minimum, maximum)
    lower = (minimum - mean) / sigma
    upper = (maximum - mean) / sigma
    normalization = _standard_normal_cdf(upper) - _standard_normal_cdf(lower)
    z = (clipped_mass - mean) / sigma
    cdf = (_standard_normal_cdf(z) - _standard_normal_cdf(lower)) / normalization
    return jnp.where(mass < minimum, 0.0, jnp.where(mass > maximum, 1.0, cdf))


def power_law_plus_peak_pdf(  # noqa: PLR0913
    mass: Array,
    alpha: float,
    minimum: float,
    maximum: float,
    lambda_peak: float,
    peak_mean: float,
    peak_sigma: float,
    peak_maximum: float,
) -> Array:
    """Return the normalized Talbot-Thrane-style power-law-plus-peak density."""
    checkify.check(minimum < maximum, "minimum must be strictly less than maximum.")
    checkify.check(minimum < peak_maximum, "minimum must be strictly less than peak_maximum.")
    checkify.check(lambda_peak >= 0.0, "lambda_peak must be non-negative.")
    checkify.check(lambda_peak <= 1.0, "lambda_peak must be at most 1.")
    checkify.check(peak_sigma > 0.0, "peak_sigma must be strictly positive.")

    power_law_component = _power_law_pdf(mass=mass, alpha=alpha, minimum=minimum, maximum=maximum)
    peak_component = _truncated_normal_pdf(
        mass=mass,
        mean=peak_mean,
        sigma=peak_sigma,
        minimum=minimum,
        maximum=peak_maximum,
    )
    return (1.0 - lambda_peak) * power_law_component + lambda_peak * peak_component


def power_law_plus_peak_cdf(  # noqa: PLR0913
    mass: Array,
    alpha: float,
    minimum: float,
    maximum: float,
    lambda_peak: float,
    peak_mean: float,
    peak_sigma: float,
    peak_maximum: float,
) -> Array:
    """Return the normalized Talbot-Thrane-style power-law-plus-peak CDF."""
    checkify.check(minimum < maximum, "minimum must be strictly less than maximum.")
    checkify.check(minimum < peak_maximum, "minimum must be strictly less than peak_maximum.")
    checkify.check(lambda_peak >= 0.0, "lambda_peak must be non-negative.")
    checkify.check(lambda_peak <= 1.0, "lambda_peak must be at most 1.")
    checkify.check(peak_sigma > 0.0, "peak_sigma must be strictly positive.")

    power_law_component = _power_law_cdf(mass=mass, alpha=alpha, minimum=minimum, maximum=maximum)
    peak_component = _truncated_normal_cdf(
        mass=mass,
        mean=peak_mean,
        sigma=peak_sigma,
        minimum=minimum,
        maximum=peak_maximum,
    )
    return (1.0 - lambda_peak) * power_law_component + lambda_peak * peak_component
