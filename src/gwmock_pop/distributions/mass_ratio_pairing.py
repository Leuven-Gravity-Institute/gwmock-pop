"""Conditional mass-ratio pairing distribution helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify


def _effective_minimum_mass_ratio(primary_mass: Array, secondary_minimum: float, minimum: float) -> Array:
    """Return the support lower bound for the conditional mass ratio."""
    primary_mass_array = jnp.asarray(primary_mass)
    checkify.check(jnp.all(primary_mass_array > 0.0), "primary_mass must be strictly positive.")
    return jnp.maximum(minimum, secondary_minimum / primary_mass_array)


def mass_ratio_pairing_pdf(
    mass_ratio: Array,
    primary_mass: Array,
    beta: float,
    secondary_minimum: float,
    minimum: float = 0.0,
) -> Array:
    """Return the normalized conditional pairing density ``p(q | m1)``."""
    lower = _effective_minimum_mass_ratio(
        primary_mass=primary_mass,
        secondary_minimum=secondary_minimum,
        minimum=minimum,
    )
    checkify.check(jnp.all(lower < 1.0), "mass-ratio support is empty for at least one primary mass.")
    return _power_law_mass_ratio_pdf(mass_ratio=mass_ratio, beta=beta, lower=lower)


def mass_ratio_pairing_cdf(
    mass_ratio: Array,
    primary_mass: Array,
    beta: float,
    secondary_minimum: float,
    minimum: float = 0.0,
) -> Array:
    """Return the normalized conditional pairing CDF ``P(q' <= q | m1)``."""
    lower = _effective_minimum_mass_ratio(
        primary_mass=primary_mass,
        secondary_minimum=secondary_minimum,
        minimum=minimum,
    )
    checkify.check(jnp.all(lower < 1.0), "mass-ratio support is empty for at least one primary mass.")

    mass_ratio_array, lower_array = jnp.broadcast_arrays(jnp.asarray(mass_ratio), lower)
    clipped = jnp.clip(mass_ratio_array, lower_array, 1.0)
    support = (mass_ratio_array >= lower_array) & (mass_ratio_array <= 1.0)
    if jnp.isclose(beta, -1.0):
        numerator = jnp.log(clipped / lower_array)
        denominator = jnp.log(1.0 / lower_array)
    else:
        exponent = beta + 1.0
        numerator = clipped**exponent - lower_array**exponent
        denominator = 1.0 - lower_array**exponent
    cdf = numerator / denominator
    return jnp.where(
        mass_ratio_array < lower_array,
        0.0,
        jnp.where(mass_ratio_array > 1.0, 1.0, jnp.where(support, cdf, 0.0)),
    )


def _power_law_mass_ratio_pdf(mass_ratio: Array, beta: float, lower: Array) -> Array:
    """Return the normalized ``q**beta`` density on ``[lower, 1]``."""
    mass_ratio_array, lower_array = jnp.broadcast_arrays(jnp.asarray(mass_ratio), lower)
    support = (mass_ratio_array >= lower_array) & (mass_ratio_array <= 1.0)
    if jnp.isclose(beta, -1.0):
        normalization = jnp.log(1.0 / lower_array)
        density = 1.0 / (mass_ratio_array * normalization)
    else:
        exponent = beta + 1.0
        normalization = 1.0 - lower_array**exponent
        density = exponent * mass_ratio_array**beta / normalization
    return jnp.where(support, density, 0.0)
