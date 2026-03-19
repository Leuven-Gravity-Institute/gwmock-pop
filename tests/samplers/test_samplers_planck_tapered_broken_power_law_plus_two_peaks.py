"""Tests for the Planck tapered broken power law plus two peaks sampler."""

import jax
import jax.numpy as jnp

from gwmock_pop.samplers import planck_tapered_broken_power_law_plus_two_peaks


class TestPlanckTaperedBrokenPowerLawPlusTwoPeaksSampler:
    """Tests for the Planck tapered broken power law plus two peaks sampler."""

    def test_sampler_output_shape(self):
        """Test that the sampler returns the correct shape."""
        key = jax.random.PRNGKey(42)
        n_samples = 1000
        alpha_1 = 2.0
        alpha_2 = 1.5
        transition = 1e9
        minimum = 1e8
        maximum = 1e10
        mean_1 = 1e9
        sigma_1 = 1e8
        mean_2 = 5e9
        sigma_2 = 1e9
        taper_range = 1e9
        lambda_0 = 0.3
        lambda_1 = 0.4

        samples = planck_tapered_broken_power_law_plus_two_peaks(
            key=key,
            n_samples=n_samples,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            transition=transition,
            minimum=minimum,
            maximum=maximum,
            mean_1=mean_1,
            sigma_1=sigma_1,
            mean_2=mean_2,
            sigma_2=sigma_2,
            taper_range=taper_range,
            lambda_0=lambda_0,
            lambda_1=lambda_1,
        )

        assert jnp.shape(samples) == (n_samples,)
        assert jnp.all(samples >= minimum)
        assert jnp.all(samples <= maximum)

    def test_sampler_with_different_keys(self):
        """Test that different random keys produce different samples."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        n_samples = 1000
        alpha_1 = 2.0
        alpha_2 = 1.5
        transition = 1e9
        minimum = 1e8
        maximum = 1e10
        mean_1 = 1e9
        sigma_1 = 1e8
        mean_2 = 5e9
        sigma_2 = 1e9
        taper_range = 1e9
        lambda_0 = 0.3
        lambda_1 = 0.4

        samples1 = planck_tapered_broken_power_law_plus_two_peaks(
            key=key1,
            n_samples=n_samples,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            transition=transition,
            minimum=minimum,
            maximum=maximum,
            mean_1=mean_1,
            sigma_1=sigma_1,
            mean_2=mean_2,
            sigma_2=sigma_2,
            taper_range=taper_range,
            lambda_0=lambda_0,
            lambda_1=lambda_1,
        )

        # Test that different keys produce different samples
        samples2 = planck_tapered_broken_power_law_plus_two_peaks(
            key=key2,
            n_samples=n_samples,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            transition=transition,
            minimum=minimum,
            maximum=maximum,
            mean_1=mean_1,
            sigma_1=sigma_1,
            mean_2=mean_2,
            sigma_2=sigma_2,
            taper_range=taper_range,
            lambda_0=lambda_0,
            lambda_1=lambda_1,
        )

        # Samples should be different when keys are different
        # Note: There's a small probability they could be the same, but it's very unlikely
        # with different keys, so we don't expect all elements to be different
        assert not jnp.allclose(samples1, samples2)

    def test_sampler_with_same_seed_same_output(self):
        """Test that same seeds produce same outputs."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)

        n_samples = 1000
        alpha_1 = 2.0
        alpha_2 = 1.5
        transition = 1e9
        minimum = 1e8
        maximum = 1e10
        mean_1 = 1e9
        sigma_1 = 1e8
        mean_2 = 5e9
        sigma_2 = 1e9
        taper_range = 1e9
        lambda_0 = 0.3
        lambda_1 = 0.4

        samples1 = planck_tapered_broken_power_law_plus_two_peaks(
            key=key1,
            n_samples=n_samples,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            transition=transition,
            minimum=minimum,
            maximum=maximum,
            mean_1=mean_1,
            sigma_1=sigma_1,
            mean_2=mean_2,
            sigma_2=sigma_2,
            taper_range=taper_range,
            lambda_0=lambda_0,
            lambda_1=lambda_1,
        )

        samples2 = planck_tapered_broken_power_law_plus_two_peaks(
            key=key2,
            n_samples=n_samples,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            transition=transition,
            minimum=minimum,
            maximum=maximum,
            mean_1=mean_1,
            sigma_1=sigma_1,
            mean_2=mean_2,
            sigma_2=sigma_2,
            taper_range=taper_range,
            lambda_0=lambda_0,
            lambda_1=lambda_1,
        )

        assert jnp.allclose(samples1, samples2)

    def test_sampler_with_different_sample_sizes(self):
        """Test sampler with different sample sizes."""
        key = jax.random.PRNGKey(42)
        alpha_1 = 2.0
        alpha_2 = 1.0
        transition = 1e9
        minimum = 1e8
        maximum = 1e10
        mean_1 = 1e9
        sigma_1 = 1e8
        mean_2 = 5e9
        sigma_2 = 1e9
        taper_range = 1e9
        lambda_0 = 0.3
        lambda_1 = 0.4

        sample_sizes = [1, 10, 100, 1000]
        for size in sample_sizes:
            samples = planck_tapered_broken_power_law_plus_two_peaks(
                key=key,
                n_samples=size,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                transition=transition,
                minimum=minimum,
                maximum=maximum,
                mean_1=mean_1,
                sigma_1=sigma_1,
                mean_2=mean_2,
                sigma_2=sigma_2,
                taper_range=taper_range,
                lambda_0=lambda_0,
                lambda_1=lambda_1,
            )
            assert jnp.shape(samples) == (size,)
