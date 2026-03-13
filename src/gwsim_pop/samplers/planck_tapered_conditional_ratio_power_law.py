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
        denominator: Denominator.
        beta: Spectral index of the ratio.
        numerator_minimum: Minimum of the numerator.
        taper_range: Range of the tapering.
        minimum: Minimum of the ratio.
        maximum: Maximum of the ratio.
        n_grids: Number of grid points to evaluate the CDF.

    Returns:
        Array of sampled mass ratios.
    """
    # Compute the cdf
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
