"""Functions to smooth distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def planck_tapering_function(x: jax.Array, x_min: float, delta: float) -> jax.Array:
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

    def planck_rise(x: jax.Array, x_min: float, delta: float) -> jax.Array:
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
