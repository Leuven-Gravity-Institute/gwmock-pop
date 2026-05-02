"""Sampler for the conditional mass-ratio pairing function."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify


def mass_ratio_pairing(  # noqa: PLR0913
    key: Array,
    primary_mass: Array,
    beta: float,
    secondary_minimum: float,
    minimum: float = 0.0,
    n_samples: int | None = None,
) -> Array:
    """Draw one conditional mass ratio per primary mass."""
    primary_mass_array = jnp.asarray(primary_mass)

    if primary_mass_array.ndim == 0:
        if n_samples is None:
            raise ValueError("n_samples is required when primary_mass is scalar.")
        target_shape = (n_samples,)
        primary_mass_array = jnp.full(shape=target_shape, fill_value=primary_mass_array)
    else:
        target_shape = primary_mass_array.shape
        if n_samples is not None:
            checkify.check(target_shape[0] == n_samples, "n_samples must match the leading dimension of primary_mass.")

    lower = jnp.maximum(minimum, secondary_minimum / primary_mass_array)
    checkify.check(jnp.all(lower < 1.0), "mass-ratio support is empty for at least one primary mass.")

    uniform = jax.random.uniform(key, shape=target_shape)
    return _inverse_mass_ratio_pairing_cdf(uniform=uniform, beta=beta, lower=lower)


def _inverse_mass_ratio_pairing_cdf(uniform: Array, beta: float, lower: Array) -> Array:
    """Invert the normalized pairing CDF analytically."""
    if jnp.isclose(beta, -1.0):
        return lower * jnp.exp(uniform * jnp.log(1.0 / lower))

    exponent = beta + 1.0
    return (lower**exponent + uniform * (1.0 - lower**exponent)) ** (1.0 / exponent)
