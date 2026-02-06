"""Broken power law distribution implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def broken_power_law_unnormalized_logpdf(  # noqa: PLR0913
    x: jax.Array,
    x_min: float,
    x_max: float,
    transition: float,
    alpha_1: float,
    alpha_2: float,
) -> jax.Array:
    """Unnormalized log probability density function of a broken power law distribution.

    References:
    - GWTC-4.0: Population Properties of Merging Compact Binaries: https://arxiv.org/abs/2508.18083

    Args:
        x: The input value.
        x_min: The minimum value of the distribution.
        x_max: The maximum value of the distribution.
        transition: The transition point where the power law changes.
        alpha_1: The power law index for x < transition.
        alpha_2: The power law index for x >= transition.

    Returns:
        An array with the unnormalized log probability density at x.

    """
    return jnp.where(
        x < x_min,
        -jnp.inf,
        jnp.where(
            x < transition,
            -alpha_1 * jnp.log(x / transition),
            jnp.where(
                x < x_max,
                -alpha_2 * jnp.log(x / transition),
                -jnp.inf,
            ),
        ),
    )


def broken_power_law_log_normalization_constant(
    x_min: float,
    x_max: float,
    transition: float,
    alpha_1: float,
    alpha_2: float,
) -> jax.Array:
    """Log normalization constant of a broken power law distribution.

    References:
    - GWTC-4.0: Population Properties of Merging Compact Binaries: https://arxiv.org/abs/2508.18083

    Args:
        x_min: The minimum value of the distribution.
        x_max: The maximum value of the distribution.
        transition: The transition point where the power law changes.
        alpha_1: The power law index for x < transition.
        alpha_2: The power law index for x >= transition.

    Returns:
        The log normalization constant of the distribution.

    """
    return jnp.log(transition) + jnp.log(
        (1 - (x_min / transition) ** (1 - alpha_1)) / (1 - alpha_1)
        + ((x_max / transition) ** (1 - alpha_2) - 1) / (1 - alpha_2)
    )


def broken_power_law_logpdf(  # noqa: PLR0913
    x: jax.Array, x_min: float, x_max: float, transition: float, alpha_1: float, alpha_2: float
) -> jax.Array:
    """Log probability density function of a broken power law distribution.

    References:
    - GWTC-4.0: Population Properties of Merging Compact Binaries: https://arxiv.org/abs/2508.18083

    Args:
        x: The input value.
        x_min: The minimum value of the distribution.
        x_max: The maximum value of the distribution.
        transition: The transition point where the power law changes.
        alpha_1: The power law index for x < transition.
        alpha_2: The power law index for x >= transition.

    Returns:
        An array with the log probability density at x.

    """
    return broken_power_law_unnormalized_logpdf(
        x=x, x_min=x_min, x_max=x_max, transition=transition, alpha_1=alpha_1, alpha_2=alpha_2
    ) - broken_power_law_log_normalization_constant(
        x_min=x_min, x_max=x_max, transition=transition, alpha_1=alpha_1, alpha_2=alpha_2
    )
