"""Uniform sampler utilities for graph presets."""

from __future__ import annotations

import jax
from jax import Array


def uniform(*, key: Array, n_samples: int, minimum: float, maximum: float) -> Array:
    """Draw samples uniformly on ``[minimum, maximum]``."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if maximum < minimum:
        raise ValueError(f"maximum must be >= minimum, got {maximum} < {minimum}.")
    return jax.random.uniform(key, shape=(n_samples,), minval=minimum, maxval=maximum)
