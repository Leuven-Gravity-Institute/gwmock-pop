"""Joint uniform component-mass sampler with optional total-mass rejection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def joint_uniform_mass_pair(  # noqa: PLR0913
    *,
    key: Array,
    n_samples: int,
    m1_min: float,
    m1_max: float,
    m2_min: float,
    m2_max: float,
    total_mass_max: float | None = None,
    ordered: bool = False,
) -> Array:
    """Draw a pair of component masses uniformly from independent ranges.

    The two component masses are drawn independently and uniformly on
    ``[m1_min, m1_max]`` and ``[m2_min, m2_max]`` respectively. When
    ``total_mass_max`` is set, pairs with ``m1 + m2 > total_mass_max`` are
    rejected and resampled until ``n_samples`` accepted pairs are collected
    (matching the legacy prior-simulator behaviour). When ``ordered`` is
    ``True`` the first row holds the larger mass and the second the smaller.

    Args:
        key: JAX PRNG key.
        n_samples: Number of mass pairs to sample.
        m1_min: Lower bound for the first component mass.
        m1_max: Upper bound for the first component mass.
        m2_min: Lower bound for the second component mass.
        m2_max: Upper bound for the second component mass.
        total_mass_max: Optional upper bound on ``m1 + m2``.
        ordered: If ``True``, reorder each pair so row 0 >= row 1.

    Returns:
        Array of shape ``(2, n_samples)`` with component masses stacked along
        the leading axis.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if m1_max < m1_min:
        raise ValueError(f"m1_max must be >= m1_min, got {m1_max} < {m1_min}.")
    if m2_max < m2_min:
        raise ValueError(f"m2_max must be >= m2_min, got {m2_max} < {m2_min}.")
    if total_mass_max is not None and total_mass_max <= m1_min + m2_min:
        raise ValueError("total_mass_max must be greater than m1_min + m2_min to admit any samples.")

    if total_mass_max is None:
        key_1, key_2 = jax.random.split(key)
        mass_1 = jax.random.uniform(key_1, shape=(n_samples,), minval=m1_min, maxval=m1_max)
        mass_2 = jax.random.uniform(key_2, shape=(n_samples,), minval=m2_min, maxval=m2_max)
        return _order_pair(mass_1, mass_2, ordered=ordered)

    if n_samples == 0:
        empty = jnp.empty((0,))
        return _order_pair(empty, empty, ordered=ordered)

    accepted_mass_1: list[Array] = []
    accepted_mass_2: list[Array] = []
    remaining = n_samples
    current_key = key
    while remaining > 0:
        current_key, key_1, key_2 = jax.random.split(current_key, 3)
        batch_size = max(remaining * 4, 256)
        batch_mass_1 = jax.random.uniform(key_1, shape=(batch_size,), minval=m1_min, maxval=m1_max)
        batch_mass_2 = jax.random.uniform(key_2, shape=(batch_size,), minval=m2_min, maxval=m2_max)
        accepted = (batch_mass_1 + batch_mass_2) <= total_mass_max
        if not bool(jnp.any(accepted)):
            continue
        accepted_mass_1.append(batch_mass_1[accepted][:remaining])
        accepted_mass_2.append(batch_mass_2[accepted][:remaining])
        remaining -= int(accepted_mass_1[-1].shape[0])

    return _order_pair(jnp.concatenate(accepted_mass_1), jnp.concatenate(accepted_mass_2), ordered=ordered)


def _order_pair(mass_1: Array, mass_2: Array, *, ordered: bool) -> Array:
    """Stack two mass arrays, optionally enforcing row 0 >= row 1."""
    if ordered:
        return jnp.stack((jnp.maximum(mass_1, mass_2), jnp.minimum(mass_1, mass_2)))
    return jnp.stack((mass_1, mass_2))
