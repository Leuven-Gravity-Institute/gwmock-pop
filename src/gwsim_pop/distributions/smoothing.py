"""Functions to smooth distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def planck_tapering_function(x: Array, x_min: float, delta: float) -> Array:
    """Planck tapering function to smoothly transition between 0 and 1.

    References:
    - GWTC-4.0: Population Properties of Merging Compact Binaries: https://arxiv.org/abs/2508.18083

    Args:
        x: The input value.
        x_min: The minimum value of the transition region.
        delta: The width of the transition region.

    Returns:
        An array with the value of the tapering function at x.

    """

    def planck_rise(x: Array, x_min: float, delta: float) -> Array:
        """Planck rise function from 0 to 1 over the interval [x_min, x_min + delta].

        Args:
            x: The input value.
            x_min: The minimum value of the transition region.
            delta: The width of the transition region.

        Returns:
            The value of the rise function at x.

        """
        a = jnp.exp(delta / (x - x_min) + delta / (x - (x_min + delta)))
        return 1.0 / (1.0 + a)

    return jnp.where(
        x < x_min,
        0.0,
        jnp.where(
            x < x_min + delta,
            planck_rise(x, x_min, delta),
            1.0,
        ),
    )


def log_planck_tapering_function(x: Array, x_min: float, delta: float) -> Array:
    """Log of the Planck tapering function to smoothly transition between 0 and 1.

    References:
    - GWTC-4.0: Population Properties of Merging Compact Binaries: https://arxiv.org/abs/2508.18083

    Args:
        x: The input value.
        x_min: The minimum value of the transition region.
        delta: The width of the transition region.

    Returns:
        An array with the value of the log tapering function at x.

    """

    def log_planck_rise(x: Array, x_min: float, delta: float) -> Array:
        """Log of the Planck rise function from -inf to 0 over the interval [x_min, x_min + delta].

        Args:
            x: The input value.
            x_min: The minimum value of the transition region.
            delta: The width of the transition region.

        Returns:
            The log of the rise function value at x, in the range (-inf, 0].

        """
        exponent = delta / (x - x_min) + delta / (x - (x_min + delta))
        return -jax.nn.softplus(exponent)

    return jnp.where(
        x < x_min,
        -jnp.inf,
        jnp.where(
            x < x_min + delta,
            log_planck_rise(x, x_min, delta),
            0.0,
        ),
    )
