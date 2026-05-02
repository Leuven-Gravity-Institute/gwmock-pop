"""Tests for the power-law-plus-peak primary-mass sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gwmock_pop.samplers import power_law_plus_peak


def test_power_law_plus_peak_sampler_returns_expected_shape_and_support() -> None:
    """The sampler returns one primary mass per requested sample within support."""
    samples = power_law_plus_peak(
        key=jax.random.PRNGKey(42),
        n_samples=1_000,
        alpha=2.63,
        minimum=4.59,
        maximum=86.22,
        lambda_peak=0.10,
        peak_mean=33.07,
        peak_sigma=5.69,
        peak_maximum=100.0,
    )

    assert samples.shape == (1_000,)
    assert bool(jnp.all(samples >= 4.59))
    assert bool(jnp.all(samples <= 100.0))


def test_power_law_plus_peak_sampler_is_reproducible_for_fixed_seed() -> None:
    """The sampler is deterministic for a fixed JAX key."""
    kwargs = {
        "n_samples": 256,
        "alpha": 2.63,
        "minimum": 4.59,
        "maximum": 86.22,
        "lambda_peak": 0.10,
        "peak_mean": 33.07,
        "peak_sigma": 5.69,
        "peak_maximum": 100.0,
    }

    sample_a = power_law_plus_peak(key=jax.random.PRNGKey(7), **kwargs)
    sample_b = power_law_plus_peak(key=jax.random.PRNGKey(7), **kwargs)

    assert jnp.allclose(sample_a, sample_b)
