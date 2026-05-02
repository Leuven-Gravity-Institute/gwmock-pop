"""Sampler for the Talbot-Thrane power-law-plus-peak primary-mass model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify


def power_law_plus_peak(  # noqa: PLR0913
    key: Array,
    n_samples: int,
    alpha: float,
    minimum: float,
    maximum: float,
    lambda_peak: float,
    peak_mean: float,
    peak_sigma: float,
    peak_maximum: float,
) -> Array:
    """Draw primary masses from a power law plus truncated Gaussian peak."""
    checkify.check(n_samples > 0, "n_samples must be strictly positive.")
    checkify.check(minimum < maximum, "minimum must be strictly less than maximum.")
    checkify.check(minimum < peak_maximum, "minimum must be strictly less than peak_maximum.")
    checkify.check(lambda_peak >= 0.0, "lambda_peak must be non-negative.")
    checkify.check(lambda_peak <= 1.0, "lambda_peak must be at most 1.")
    checkify.check(peak_sigma > 0.0, "peak_sigma must be strictly positive.")

    component_key, power_key, peak_key = jax.random.split(key, 3)
    choose_peak = jax.random.bernoulli(component_key, p=lambda_peak, shape=(n_samples,))
    power_samples = _sample_power_law(power_key, n_samples=n_samples, alpha=alpha, minimum=minimum, maximum=maximum)
    peak_samples = _sample_truncated_normal(
        peak_key,
        n_samples=n_samples,
        mean=peak_mean,
        sigma=peak_sigma,
        minimum=minimum,
        maximum=peak_maximum,
    )
    return jnp.where(choose_peak, peak_samples, power_samples)


def _sample_power_law(key: Array, n_samples: int, alpha: float, minimum: float, maximum: float) -> Array:
    """Sample a normalized power law on ``[minimum, maximum]`` analytically."""
    uniform = jax.random.uniform(key, shape=(n_samples,))
    if jnp.isclose(alpha, 1.0):
        return minimum * (maximum / minimum) ** uniform

    exponent = 1.0 - alpha
    lower = minimum**exponent
    upper = maximum**exponent
    return (lower + uniform * (upper - lower)) ** (1.0 / exponent)


def _sample_truncated_normal(  # noqa: PLR0913
    key: Array,
    n_samples: int,
    mean: float,
    sigma: float,
    minimum: float,
    maximum: float,
) -> Array:
    """Sample a truncated normal using JAX's standard-normal helper."""
    lower = (minimum - mean) / sigma
    upper = (maximum - mean) / sigma
    standard_normal = jax.random.truncated_normal(key, lower=lower, upper=upper, shape=(n_samples,))
    return mean + sigma * standard_normal
