"""Log-trapezoidal integrator."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.lax import cumlogsumexp
from jax.nn import logsumexp


def log_trapezoidal_integrand_dx(log_y: Array, x: Array) -> Array:
    r"""Compute the log integrand dx of the following integral using the trapezoidal rule.

    $$
    \log z = \log \int \exp(\log y(x)) \mathrm{d}x
    $$

    Args:
        log_y: Log of y(x).
        x: Argument of the function y.

    Return:
        Log of the integrand dx.
    """
    # Log of the average of adjacent exp(log_y) values
    log_midpoints = logsumexp(jnp.stack([log_y[:-1], log_y[1:]]), axis=0) - jnp.log(2.0)

    # Log of the widths
    log_dx = jnp.log(jnp.diff(x))

    return log_midpoints + log_dx


def log_trapezoidal_integrate(log_y: Array, x: Array) -> Array:
    r"""Compute the following integral using the trapezoidal rule.

    $$
    \log z = \log \int \exp(\log y(x)) \mathrm{d}x
    $$

    Args:
        log_y: Log of y(x).
        x: Argument of the function y.

    Return:
        Log of the integral.
    """
    return logsumexp(log_trapezoidal_integrand_dx(log_y=log_y, x=x))


def log_trapezoidal_cumsum(log_y: Array, x: Array) -> Array:
    r"""Compute the log of the cumulative sum of the following integral using the trapezoidal rule.

    $$
    \log z(x) = \log \int^{x} \exp(\log y(x')) \mathrm{d}x'
    $$

    Args:
        log_y: Log of y(x).
        x: Argument of the function y.

    Return:
        Log of the cumulative sum.
    """
    return jnp.concatenate([jnp.array([-jnp.inf]), cumlogsumexp(log_trapezoidal_integrand_dx(log_y=log_y, x=x))])
