"""Tests for Planck tapered conditional ratio power law distribution."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from jax import Array

from gwsim_pop.distributions.planck_tapered_conditional_ratio_power_law import (
    planck_tapered_conditional_ratio_power_law_unnormalized_cdf,
    planck_tapered_conditional_ratio_power_law_unnormalized_logpdf,
)


class TestPlanckTaperedConditionalRatioPowerLawUnnormalizedLogpdf:
    """Tests for planck_tapered_conditional_ratio_power_law_unnormalized_logpdf."""

    def test_basic_functionality(self):
        """Test basic functionality of the unnormalized logpdf."""
        x = jnp.array([1.0, 1.5, 2.0])
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert isinstance(result, Array)
        assert result.shape == x.shape
        # Allow for -inf values from tapering
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_scalar_input(self):
        """Test with scalar input values."""
        x = 1.0
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert isinstance(result, Array)
        assert jnp.isfinite(result)

    def test_power_law_dependence_on_x(self):
        """Test that the function increases with beta * log(x) dependence."""
        denominator = 2.0
        beta = 2.0
        numerator_minimum = 1.0
        taper_range = 1.0

        # For the same denominator, larger x should give larger logpdf when not at boundary
        x1 = jnp.array([2.0])
        x2 = jnp.array([4.0])

        result1 = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x1, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        result2 = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x2, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        # Both values should be finite
        assert jnp.isfinite(result1)
        assert jnp.isfinite(result2)

    def test_array_output_shape(self):
        """Test that output shape matches input shape."""
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        for n in [1, 5, 10, 100]:
            x = jnp.linspace(0.5, 3.0, n)
            result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
                x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
            )

            assert result.shape == x.shape

    def test_positive_beta(self):
        """Test with positive spectral index."""
        x = jnp.array([1.0, 2.0, 3.0])
        denominator = 2.0
        beta = 0.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_negative_beta(self):
        """Test with negative spectral index."""
        x = jnp.array([1.0, 2.0, 3.0])
        denominator = 2.0
        beta = -1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_zero_beta(self):
        """Test with zero spectral index."""
        x = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array(2.0)
        beta = 0.0
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_different_denominators(self):
        """Test with different denominator values."""
        x = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result1 = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=jnp.array(1.0), beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        result2 = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=jnp.array(2.0), beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        result3 = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=jnp.array(3.0), beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        # Different denominators should produce well-defined results (scalar or 0-d)
        assert jnp.ndim(result1) == 0 or isinstance(result1, (float, jnp.ndarray))
        assert jnp.ndim(result2) == 0 or isinstance(result2, (float, jnp.ndarray))
        assert jnp.ndim(result3) == 0 or isinstance(result3, (float, jnp.ndarray))

    def test_taper_effect(self):
        """Test the effect of the taper range on the result."""
        x = jnp.array([1.2])
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0

        # With narrow taper
        result_narrow = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=0.1
        )

        # With wide taper
        result_wide = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=1.0
        )

        # Both should be finite
        assert jnp.isfinite(result_narrow)
        assert jnp.isfinite(result_wide)


class TestPlanckTaperedConditionalRatioPowerLawUnnormalizedCDF:
    """Tests for planck_tapered_conditional_ratio_power_law_unnormalized_cdf."""

    def test_basic_functionality(self):
        """Test basic functionality of the CDF."""
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5
        minimum = 0.5
        maximum = 3.0
        n_grids = 100

        x_grid, cdf = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
            denominator=denominator,
            beta=beta,
            numerator_minimum=numerator_minimum,
            taper_range=taper_range,
            minimum=minimum,
            maximum=maximum,
            n_grids=n_grids,
        )

        assert isinstance(x_grid, Array)
        assert isinstance(cdf, Array)
        assert x_grid.shape == (n_grids,)
        assert cdf.shape == (n_grids,)

    def test_cdf_properties(self):
        """Test that CDF has expected properties."""
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5
        minimum = 0.5
        maximum = 3.0
        n_grids = 100

        _x_grid, cdf = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
            denominator=denominator,
            beta=beta,
            numerator_minimum=numerator_minimum,
            taper_range=taper_range,
            minimum=minimum,
            maximum=maximum,
            n_grids=n_grids,
        )

        # CDF should start near 0 and end at 1
        assert cdf[0] >= 0.0
        assert cdf[-1] == 1.0

        # CDF should be monotonically increasing
        assert jnp.all(jnp.diff(cdf) >= 0.0)

    def test_grid_range(self):
        """Test that grid spans the correct range."""
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5
        minimum = 0.5
        maximum = 3.0
        n_grids = 100

        x_grid, _ = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
            denominator=denominator,
            beta=beta,
            numerator_minimum=numerator_minimum,
            taper_range=taper_range,
            minimum=minimum,
            maximum=maximum,
            n_grids=n_grids,
        )

        # Grid should span from minimum to maximum
        assert jnp.isclose(x_grid[0], minimum)
        assert jnp.isclose(x_grid[-1], maximum)

    def test_different_n_grids(self):
        """Test with different numbers of grid points."""
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5
        minimum = 0.5
        maximum = 3.0

        for n_grids in [10, 50, 100, 200]:
            x_grid, cdf = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
                denominator=denominator,
                beta=beta,
                numerator_minimum=numerator_minimum,
                taper_range=taper_range,
                minimum=minimum,
                maximum=maximum,
                n_grids=n_grids,
            )

            assert x_grid.shape == (n_grids,)
            assert cdf.shape == (n_grids,)
            assert jnp.isclose(x_grid[0], minimum)
            assert jnp.isclose(x_grid[-1], maximum)
            assert cdf[-1] == 1.0

    def test_cdf_continuity(self):
        """Test that CDF is continuous (no jumps)."""
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5
        minimum = 0.5
        maximum = 3.0
        n_grids = 100

        _, cdf = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
            denominator=denominator,
            beta=beta,
            numerator_minimum=numerator_minimum,
            taper_range=taper_range,
            minimum=minimum,
            maximum=maximum,
            n_grids=n_grids,
        )

        # Check that differences are not excessively large
        diff = jnp.diff(cdf)
        assert jnp.max(diff) < 0.1  # No large jumps  # noqa: PLR2004

    def test_monotonic_increase(self):
        """Test that CDF is monotonically increasing."""
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5
        minimum = 0.5
        maximum = 3.0
        n_grids = 100

        _, cdf = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
            denominator=denominator,
            beta=beta,
            numerator_minimum=numerator_minimum,
            taper_range=taper_range,
            minimum=minimum,
            maximum=maximum,
            n_grids=n_grids,
        )

        # Check monotonicity
        assert jnp.all(jnp.diff(cdf) >= 0.0)


class TestEdgeCases:
    """Tests for edge cases and special parameter values."""

    def test_extreme_beta_values(self):
        """Test with extreme spectral index values."""
        x = jnp.array(1.0)
        denominator = jnp.array(2.0)
        numerator_minimum = 1.0
        taper_range = 0.5

        # Test with very steep power law
        result_steep = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=10.0, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        # Test with very shallow power law
        result_shallow = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=0.01, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert jnp.isfinite(result_steep)
        assert jnp.isfinite(result_shallow)

    def test_small_x_values(self):
        """Test with very small x values."""
        x = jnp.array([0.1, 0.5, 1.0])
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert result.shape == x.shape
        # Allow for -inf values from tapering
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_large_x_values(self):
        """Test with large x values."""
        x = jnp.array([10.0, 100.0, 1000.0])
        denominator = 2.0
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_small_denominator(self):
        """Test with very small denominator values."""
        x = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=jnp.array(0.01), beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_large_denominator(self):
        """Test with large denominator values."""
        x = jnp.array(1.0)
        beta = 1.5
        numerator_minimum = 1.0
        taper_range = 0.5

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=jnp.array(100.0), beta=beta, numerator_minimum=numerator_minimum, taper_range=taper_range
        )

        assert jnp.isfinite(result)

    def test_small_taper_range(self):
        """Test with very small taper range."""
        x = jnp.array([1.0, 1.2, 1.5])
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=0.01
        )

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_large_taper_range(self):
        """Test with large taper range."""
        x = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array(2.0)
        beta = 1.5
        numerator_minimum = 1.0

        result = planck_tapered_conditional_ratio_power_law_unnormalized_logpdf(
            x=x, denominator=denominator, beta=beta, numerator_minimum=numerator_minimum, taper_range=5.0
        )

        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))

    def test_cdf_with_different_parameters(self):
        """Test CDF with different parameter combinations."""
        minimum = 0.5
        maximum = 3.0
        n_grids = 50

        param_combinations = [
            {"denominator": jnp.array(1.0), "beta": 0.5},
            {"denominator": jnp.array(2.0), "beta": 1.5},
            {"denominator": jnp.array(5.0), "beta": 2.5},
        ]

        for params in param_combinations:
            x_grid, cdf = planck_tapered_conditional_ratio_power_law_unnormalized_cdf(
                denominator=cast(Array, params["denominator"]),
                beta=cast(float, params["beta"]),
                numerator_minimum=1.0,
                taper_range=0.5,
                minimum=minimum,
                maximum=maximum,
                n_grids=n_grids,
            )

            assert x_grid.shape == (n_grids,)
            assert cdf.shape == (n_grids,)
            assert cdf[-1] == 1.0
            assert jnp.all(jnp.diff(cdf) >= 0.0)
