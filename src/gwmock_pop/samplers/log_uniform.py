"""Log-uniform sampler utilities for graph presets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def log_uniform(*, key: Array, n_samples: int, minimum: float, maximum: float) -> Array:
    """Draw samples uniformly in log-space on ``[minimum, maximum]``."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if minimum <= 0.0:
        raise ValueError(f"minimum must be > 0 for log-uniform sampling, got {minimum}.")
    if maximum <= minimum:
        raise ValueError(f"maximum must be > minimum, got {maximum} <= {minimum}.")

    log_samples = jax.random.uniform(
        key,
        shape=(n_samples,),
        minval=jnp.log(minimum),
        maxval=jnp.log(maximum),
    )
    return jnp.exp(log_samples)
