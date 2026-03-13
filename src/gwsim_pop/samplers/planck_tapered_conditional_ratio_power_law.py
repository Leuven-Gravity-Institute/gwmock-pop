"""Conditional ratio power law sampler with Planck tapering."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from gwsim_pop.distributions.planck_tapered_conditional_ratio_power_law import (
    planck_tapered_conditional_ratio_power_law_cdf,
)


def planck_tapered_conditional_ratio_power_law(  # noqa: PLR0913
    key: Array,
    n_samples: int,
    denominator: Array,
    beta: float,
    numerator_minimum: float,
    taper_range: float,
    minimum: float,
    maximum: float,
    n_grids: int,
) -> Array:
    """Sample mass ratios with conditional power law and Planck tapering.

    Args:
        key: JAX random key.
        n_samples: Number of samples to generate.
        denominator: Denominator (scalar or array).
        beta: Spectral index of the ratio.
        numerator_minimum: Minimum of the numerator.
        taper_range: Range of the tapering.
        minimum: Minimum of the ratio.
        maximum: Maximum of the ratio.
        n_grids: Number of grid points to evaluate the CDF.

    Returns:
        Array of sampled mass ratios.
    """
    # Handle array denominators by generating samples for each element
    denominator_shape = jnp.shape(denominator)
    is_array = len(denominator_shape) > 0 and denominator_shape[0] > 0

    if is_array:
        # Denominator is an array - generate one sample per element
        n_denom = denominator_shape[0]
        keys = jax.random.split(key, n_denom)
        samples_list = []
        for _i, (d, k) in enumerate(zip(denominator, keys, strict=True)):
            x, cdf = planck_tapered_conditional_ratio_power_law_cdf(
                denominator=d,
                beta=beta,
                numerator_minimum=numerator_minimum,
                taper_range=taper_range,
                minimum=minimum,
                maximum=maximum,
                n_grids=n_grids,
            )
            uniform_sample = jax.random.uniform(key=k)
            sample = jnp.interp(x=uniform_sample, xp=cdf, fp=x)
            samples_list.append(sample)
        return jnp.array(samples_list)
    else:
        # Denominator is scalar - original behavior
        x, cdf = planck_tapered_conditional_ratio_power_law_cdf(
            denominator=denominator,
            beta=beta,
            numerator_minimum=numerator_minimum,
            taper_range=taper_range,
            minimum=minimum,
            maximum=maximum,
            n_grids=n_grids,
        )

        # Draw uniform samples
        uniform_samples = jax.random.uniform(key=key, shape=(n_samples,))
        return jnp.interp(x=uniform_samples, xp=cdf, fp=x)
