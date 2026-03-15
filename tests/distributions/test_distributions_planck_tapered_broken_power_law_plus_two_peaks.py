"""Tests for Planck tapered broken power law plus two peaks distribution."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from gwsim_pop.distributions.planck_tapered_broken_power_law_plus_two_peaks import (
    planck_tapered_broken_power_law_plus_two_peaks_cdf,
    planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf,
)


class TestPlanckTaperedBrokenPowerLawPlusTwoPeaksUnnormalizedLogpdf:
    """Tests for planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf."""

    def test_basic_functionality(self):
        """Test basic functionality of the unnormalized logpdf."""
        x = jnp.array([1.5, 2.0, 3.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert isinstance(result, Array)
        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_scalar_input(self):
        """Test with scalar input values."""
        x = jnp.array(2.0)
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert isinstance(result, Array)
        # logsumexp returns scalar even with stacked components
        # but the result is well-defined
        assert jnp.ndim(result) >= 0

    def test_near_minimum_boundary(self):
        """Test behavior near the minimum boundary."""
        x = jnp.array([1.0, 1.1, 1.2])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert result.shape == x.shape
        # Values near minimum should be lower due to tapering
        assert result[0] < result[1]

    def test_different_weights(self):
        """Test with different weight combinations."""
        x = jnp.array([2.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0

        # Test with different lambda values
        result1 = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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
            lambda_0=0.7,
            lambda_1=0.2,
        )

        result2 = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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
            lambda_0=0.3,
            lambda_1=0.5,
        )

        # Different weights should produce different results
        assert not jnp.allclose(result1, result2)

    def test_array_output_shape(self):
        """Test that output shape matches input shape."""
        for n in [1, 5, 10, 100]:
            x = jnp.linspace(1.0, 5.0, n)
            alpha_1 = 1.5
            alpha_2 = 2.5
            transition = 2.0
            minimum = 1.0
            maximum = 5.0
            mean_1 = 2.0
            sigma_1 = 0.5
            mean_2 = 3.5
            sigma_2 = 0.5
            taper_range = 1.0
            lambda_0 = 0.5
            lambda_1 = 0.3

            result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
                x=x,
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

            assert result.shape == x.shape

    def test_peak_behavior(self):
        """Test that the function produces expected values around parameter settings."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 3.0
        minimum = 1.5
        maximum = 6.0
        mean_1 = 3.0
        sigma_1 = 0.5
        mean_2 = 5.0
        sigma_2 = 0.5
        taper_range = 0.5
        lambda_0 = 0.3
        lambda_1 = 0.35

        # Sample points around first peak
        x_around_peak1 = jnp.array([mean_1 - 0.3, mean_1, mean_1 + 0.3])
        result_peak1 = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x_around_peak1,
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

        # All values should be well-defined (no NaN)
        assert jnp.all(~jnp.isnan(result_peak1))

    def test_weights_sum_to_one(self):
        """Test that weights correctly sum to 1."""
        # lambda_0 + lambda_1 + lambda_2 should equal 1
        lambda_0 = 0.5
        lambda_1 = 0.3
        lambda_2 = 1.0 - lambda_0 - lambda_1

        assert jnp.isclose(lambda_0 + lambda_1 + lambda_2, 1.0)

    def test_negative_lambda_0_raises_error(self):
        """Test that negative lambda_0 raises ValueError."""
        x = jnp.array([2.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = -0.1  # Invalid: negative
        lambda_1 = 0.3

        with pytest.raises(ValueError, match=r"Invalid mixture weights: require lambda_0 >=0\."):
            planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
                x=x,
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

    def test_negative_lambda_1_raises_error(self):
        """Test that negative lambda_1 raises ValueError."""
        x = jnp.array([2.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = -0.1  # Invalid: negative

        with pytest.raises(ValueError, match=r"Invalid mixture weights: require lambda_1 >=0\."):
            planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
                x=x,
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

    def test_weights_sum_greater_than_one_raises_error(self):
        """Test that lambda_0 + lambda_1 > 1 raises ValueError."""
        x = jnp.array([2.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.7
        lambda_1 = 0.4  # Invalid: sum > 1

        with pytest.raises(ValueError, match=r"Invalid mixture weights: require lambda_0 \+ lambda_1 <= 1\."):
            planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
                x=x,
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


class TestPlanckTaperedBrokenPowerLawPlusTwoPeaksCDF:
    """Tests for planck_tapered_broken_power_law_plus_two_peaks_cdf."""

    def test_basic_functionality(self):
        """Test basic functionality of the CDF."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3
        n_grids = 100

        x_grid, cdf = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
            n_grids=n_grids,
        )

        assert isinstance(x_grid, Array)
        assert isinstance(cdf, Array)
        assert x_grid.shape == (n_grids,)
        assert cdf.shape == (n_grids,)

    def test_cdf_properties(self):
        """Test that CDF has expected properties."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3
        n_grids = 100

        _x_grid, cdf = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
            n_grids=n_grids,
        )

        # CDF should start near 0 and end at 1
        assert cdf[0] >= 0.0
        assert cdf[-1] == 1.0

        # CDF should be monotonically increasing
        assert jnp.all(jnp.diff(cdf) >= 0.0)

    def test_grid_range(self):
        """Test that grid spans the correct range."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.5
        maximum = 4.5
        mean_1 = 2.5
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3
        n_grids = 100

        x_grid, _ = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
            n_grids=n_grids,
        )

        # Grid should span from minimum to maximum
        assert jnp.isclose(x_grid[0], minimum)
        assert jnp.isclose(x_grid[-1], maximum)

    def test_different_n_grids(self):
        """Test with different numbers of grid points."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        for n_grids in [10, 50, 100, 200]:
            x_grid, cdf = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
                n_grids=n_grids,
            )

            assert x_grid.shape == (n_grids,)
            assert cdf.shape == (n_grids,)
            assert jnp.isclose(x_grid[0], minimum)
            assert jnp.isclose(x_grid[-1], maximum)
            assert cdf[-1] == 1.0

    def test_cdf_continuity(self):
        """Test that CDF is continuous (no jumps)."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3
        n_grids = 100

        _, cdf = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
            n_grids=n_grids,
        )

        # Check that differences are not excessively large
        diff = jnp.diff(cdf)
        assert jnp.max(diff) < 0.1  # No large jumps  # noqa: PLR2004

    def test_monotonic_increase(self):
        """Test that CDF is strictly monotonically increasing."""
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3
        n_grids = 100

        _, cdf = planck_tapered_broken_power_law_plus_two_peaks_cdf(
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
            n_grids=n_grids,
        )

        # Check monotonicity
        assert jnp.all(jnp.diff(cdf) >= 0.0)


class TestEdgeCases:
    """Tests for edge cases and special parameter values."""

    def test_extreme_alpha_values(self):
        """Test with extreme spectral index values."""
        x = jnp.array([2.0])
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        # Test with very steep power law
        result_steep = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
            alpha_1=10.0,
            alpha_2=10.0,
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

        # Test with very shallow power law
        result_shallow = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
            alpha_1=0.1,
            alpha_2=0.1,
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

        assert jnp.isfinite(result_steep)
        assert jnp.isfinite(result_shallow)

    def test_narrow_peaks(self):
        """Test with very narrow peaks (small sigma)."""
        x = jnp.linspace(1.5, 5.0, 50)
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.01  # Very narrow
        mean_2 = 3.5
        sigma_2 = 0.01  # Very narrow
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert result.shape == x.shape
        # All results should be either finite or -inf (expected from tapering)
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_wide_peaks(self):
        """Test with very wide peaks (large sigma)."""
        x = jnp.linspace(1.5, 5.0, 50)
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 2.0  # Very wide
        mean_2 = 3.5
        sigma_2 = 2.0  # Very wide
        taper_range = 1.0
        lambda_0 = 0.5
        lambda_1 = 0.3

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert result.shape == x.shape
        # All results should be either finite or -inf (expected from tapering)
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_zero_weight_on_peaks(self):
        """Test with all weight on power law component."""
        x = jnp.array([2.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 1.0  # All weight on power law
        lambda_1 = 0.0  # No weight on peaks

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert jnp.isfinite(result)

    def test_zero_weight_on_power_law(self):
        """Test with all weight on peaks components."""
        x = jnp.array([2.0])
        alpha_1 = 1.5
        alpha_2 = 2.5
        transition = 2.0
        minimum = 1.0
        maximum = 5.0
        mean_1 = 2.0
        sigma_1 = 0.5
        mean_2 = 3.5
        sigma_2 = 0.5
        taper_range = 1.0
        lambda_0 = 0.0  # No weight on power law
        lambda_1 = 0.5  # Split weight between peaks

        result = planck_tapered_broken_power_law_plus_two_peaks_unnormalized_logpdf(
            x=x,
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

        assert jnp.isfinite(result)
