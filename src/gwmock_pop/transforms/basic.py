"""Basic transform helpers for graph-based simulator configs."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def constant_like(reference: Array, value: float) -> Array:
    """Return a constant array with the same shape as ``reference``.

    Args:
        reference: Array providing the target shape.
        value: Scalar value to broadcast across the output.

    Returns:
        A 1-D array with the same shape as ``reference`` filled with ``value``.
    """
    dtype = jnp.result_type(reference, jnp.asarray(value))
    return jnp.full(shape=reference.shape, fill_value=value, dtype=dtype)


def multiply(left: Array, right: Array) -> Array:
    """Multiply two array-like inputs elementwise.

    Args:
        left: Left multiplicand.
        right: Right multiplicand.

    Returns:
        Elementwise product of ``left`` and ``right``.
    """
    return jnp.asarray(left) * jnp.asarray(right)
