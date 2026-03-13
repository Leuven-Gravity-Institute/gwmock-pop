"""Broken power law plus two peaks distribution with Planck tapering."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from gwsim_pop.distributions.planck_tapered_broken_power_law_plus_two_peaks import (
    planck_tapered_broken_power_law_plus_two_peaks_cdf,
)


def planck_tapered_broken_power_law_plus_two_peaks(  # noqa: PLR0913
    key: Array,
    n_samples: int,
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
    n_grids: int = 1000,
) -> Array:
    """Draw samples from the Planck tapered broker power law plus two peaks distribution.

    Args:
        key: Random key.
        n_samples: Number of samples.
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
        An array of samples.
    """
    # Compute the cdf
    x, cdf = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
        n_grids=n_grids,
    )

    # Draw uniform samples
    uniform_samples = jax.random.uniform(key=key, shape=(n_samples,))
    return jnp.interp(x=uniform_samples, xp=cdf, fp=x)
