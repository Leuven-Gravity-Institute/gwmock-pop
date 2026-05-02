"""Tests for the cosmological distance sampler."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from gwmock_pop.samplers import uniform_comoving_volume_distance
from gwmock_pop.transforms import comoving_distance_to_redshift


def _ks_p_value(samples: np.ndarray, cdf) -> float:
    """Return the asymptotic one-sample Kolmogorov-Smirnov p-value."""
    sample = np.sort(np.asarray(samples, dtype=float))
    n_samples = sample.size
    empirical_upper = np.arange(1, n_samples + 1) / n_samples
    empirical_lower = np.arange(0, n_samples) / n_samples
    theoretical = np.clip(cdf(sample), 0.0, 1.0)
    statistic = max(
        float(np.max(empirical_upper - theoretical)),
        float(np.max(theoretical - empirical_lower)),
    )
    lam = (math.sqrt(n_samples) + 0.12 + 0.11 / math.sqrt(n_samples)) * statistic
    p_value = 2.0 * sum(((-1) ** (term - 1)) * math.exp(-2.0 * (lam**2) * (term**2)) for term in range(1, 101))
    return max(0.0, min(1.0, p_value))


def test_uniform_comoving_volume_distance_draws_within_requested_range() -> None:
    """The sampler returns luminosity distances in ``[0, d_max]``."""
    result = uniform_comoving_volume_distance(jax.random.PRNGKey(0), 1024, 40_000.0)
    assert result.shape == (1024,)
    assert np.all(np.asarray(result) >= 0.0)
    assert np.all(np.asarray(result) <= 40_000.0)


def test_uniform_comoving_volume_distance_is_uniform_in_comoving_volume() -> None:
    """The implied comoving-distance CDF follows ``(d_c / d_c,max)^3``."""
    d_max = 40_000.0
    distance = uniform_comoving_volume_distance(jax.random.PRNGKey(123), 10_000, d_max)
    redshift = comoving_distance_to_redshift(distance)
    comoving_distance = np.asarray(distance) / (1.0 + np.asarray(redshift))
    comoving_distance_max = d_max / (1.0 + float(comoving_distance_to_redshift(np.asarray(d_max))))

    p_value = _ks_p_value(comoving_distance, lambda x: np.clip((x / comoving_distance_max) ** 3, 0.0, 1.0))
    assert p_value > 0.01, p_value


def test_uniform_comoving_volume_distance_rejects_out_of_lookup_range() -> None:
    """The sampler raises when ``d_max`` exceeds the configured lookup support."""
    with pytest.raises(ValueError, match="exceeds the luminosity-distance lookup range"):
        uniform_comoving_volume_distance(jax.random.PRNGKey(0), 16, 200_000.0)
