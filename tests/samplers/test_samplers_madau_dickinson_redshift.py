"""Tests for the Madau-Dickinson rate-weighted redshift sampler."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop.distributions.madau_dickinson import madau_dickinson_redshift_pdf
from gwmock_pop.samplers import madau_dickinson_redshift


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


def _theoretical_cdf(z_min: float, z_max: float):
    """Build a CDF callable by integrating the analytic redshift density."""
    grid = np.asarray(jnp.linspace(z_min, z_max, 8192))
    density = np.asarray(madau_dickinson_redshift_pdf(jnp.asarray(grid), z_min=z_min, z_max=z_max))
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cumulative /= cumulative[-1]
    return lambda x: np.interp(x, grid, cumulative, left=0.0, right=1.0)


def test_sampler_draws_within_requested_range() -> None:
    """The sampler returns redshifts in ``[z_min, z_max]``."""
    result = madau_dickinson_redshift(jax.random.PRNGKey(0), 1024, 2.0)
    assert result.shape == (1024,)
    assert np.all(np.asarray(result) >= 0.0)
    assert np.all(np.asarray(result) <= 2.0)


def test_sampler_matches_analytic_distribution() -> None:
    """The empirical redshift CDF follows the analytic Madau-Dickinson density."""
    z_max = 2.0
    samples = madau_dickinson_redshift(jax.random.PRNGKey(123), 20_000, z_max)
    p_value = _ks_p_value(np.asarray(samples), _theoretical_cdf(0.0, z_max))
    assert p_value > 0.01, p_value


def test_sampler_respects_z_min() -> None:
    """A non-zero ``z_min`` truncates the lower redshift tail."""
    samples = madau_dickinson_redshift(jax.random.PRNGKey(7), 4096, 3.0, z_min=0.5)
    assert np.all(np.asarray(samples) >= 0.5)
    assert np.all(np.asarray(samples) <= 3.0)


def test_sampler_is_deterministic_for_fixed_key() -> None:
    """Identical keys reproduce identical draws."""
    a = madau_dickinson_redshift(jax.random.PRNGKey(42), 256, 2.0)
    b = madau_dickinson_redshift(jax.random.PRNGKey(42), 256, 2.0)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_sampler_rejects_degenerate_range() -> None:
    """The sampler raises when ``z_max`` does not exceed ``z_min``."""
    with pytest.raises(ValueError, match="z_max must be strictly greater"):
        madau_dickinson_redshift(jax.random.PRNGKey(0), 16, 1.0, z_min=1.5)


def test_sampler_rejects_negative_sample_count() -> None:
    """The sampler raises for a negative ``n_samples``."""
    with pytest.raises(ValueError, match="n_samples must be >= 0"):
        madau_dickinson_redshift(jax.random.PRNGKey(0), -1, 2.0)
