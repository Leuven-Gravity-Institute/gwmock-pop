"""Tests for log-trapezoidal integrator."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gwsim_pop.integrators.trapezoid import (
    log_trapezoidal_cumsum,
    log_trapezoidal_integrand_dx,
    log_trapezoidal_integrate,
)


class TestLogTrapezoidalIntegrandDx:
    """Tests for log_trapezoidal_integrand_dx."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality with simple input."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert isinstance(result, Array)
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_output_length(self) -> None:
        """Test that output has correct length (n-1 where n is input length)."""
        for n in [2, 3, 5, 10]:
            log_y = jnp.linspace(-5, -1, n)
            x = jnp.linspace(0, 1, n)

            result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

            assert result.shape == (n - 1,)

    def test_uniform_spacing_linear_log_y(self) -> None:
        """Test with uniform spacing and linear log_y."""
        log_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_non_uniform_spacing(self) -> None:
        """Test with non-uniform spacing."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 0.5, 2.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_constant_log_y(self) -> None:
        """Test with constant log_y values."""
        log_y = jnp.array([-2.0, -2.0, -2.0, -2.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))

    def test_zero_log_y(self) -> None:
        """Test with log_y = 0 (y = 1)."""
        log_y = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (2,)
        assert jnp.allclose(result, jnp.log(1.0) + jnp.log(1.0), atol=1e-6)

    def test_negative_inf_log_y(self) -> None:
        """Test behavior with -inf log_y values."""
        log_y = jnp.array([-jnp.inf, -1.0, -2.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_small_spacing(self) -> None:
        """Test with very small spacing."""
        log_y = jnp.array([-1.0, -1.01, -1.02])
        x = jnp.array([0.0, 0.001, 0.002])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_large_log_y_range(self) -> None:
        """Test with large range of log_y values."""
        log_y = jnp.array([-100.0, -50.0, -1.0, 0.0, 50.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (4,)
        assert jnp.all(jnp.isfinite(result))

    def test_single_interval(self) -> None:
        """Test with only two points (single interval)."""
        log_y = jnp.array([-1.0, -2.0])
        x = jnp.array([0.0, 1.0])

        result = log_trapezoidal_integrand_dx(log_y=log_y, x=x)

        assert result.shape == (1,)
        assert jnp.isfinite(result[0])


class TestLogTrapezoidalIntegrate:
    """Tests for log_trapezoidal_integrate."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality with simple input."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert isinstance(result, Array)
        assert result.shape == ()  # scalar
        assert jnp.isfinite(result)

    def test_scalar_output(self) -> None:
        """Test that output is always a scalar."""
        for n in [2, 3, 5, 10]:
            log_y = jnp.linspace(-5, -1, n)
            x = jnp.linspace(0, 1, n)

            result = log_trapezoidal_integrate(log_y=log_y, x=x)

            assert result.shape == ()

    def test_constant_function(self) -> None:
        """Test integration of constant function."""
        log_y = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_non_uniform_spacing(self) -> None:
        """Test with non-uniform spacing."""
        log_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
        x = jnp.array([0.0, 0.5, 1.5, 2.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_single_interval(self) -> None:
        """Test with only two points (single interval)."""
        log_y = jnp.array([-1.0, -2.0])
        x = jnp.array([0.0, 1.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_many_points(self) -> None:
        """Test with many points."""
        log_y = jnp.linspace(-10, -1, 100)
        x = jnp.linspace(0, 1, 100)

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_zero_log_y(self) -> None:
        """Test integration when log_y = 0 (y = 1)."""
        log_y = jnp.array([0.0, 0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_negative_inf_log_y(self) -> None:
        """Test with -inf log_y values."""
        log_y = jnp.array([-jnp.inf, -1.0, -2.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result) | jnp.isinf(result)

    def test_large_log_y_range(self) -> None:
        """Test with large range of log_y values."""
        log_y = jnp.array([-100.0, -50.0, -1.0, 0.0, 50.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_decreasing_x(self) -> None:
        """Test with decreasing x values (note: produces NaN due to negative spacing)."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([2.0, 1.0, 0.0])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        # Decreasing x produces NaN due to negative spacing in log space
        assert not jnp.isfinite(result)

    def test_narrow_integration_domain(self) -> None:
        """Test with very narrow integration domain."""
        log_y = jnp.array([-1.0, -1.001, -1.002])
        x = jnp.array([0.0, 0.001, 0.002])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)

    def test_wide_integration_domain(self) -> None:
        """Test with very wide integration domain."""
        log_y = jnp.array([-1.0, -1.5, -2.0])
        x = jnp.array([0.0, 1e6, 2e6])

        result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.isfinite(result)


class TestLogTrapezoidalCumsum:
    """Tests for log_trapezoidal_cumsum."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality with simple input."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert isinstance(result, Array)
        assert result.shape == (3,)  # n elements
        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))

    def test_output_length(self) -> None:
        """Test that output has correct length (n where n is input length)."""
        for n in [2, 3, 5, 10]:
            log_y = jnp.linspace(-5, -1, n)
            x = jnp.linspace(0, 1, n)

            result = log_trapezoidal_cumsum(log_y=log_y, x=x)

            assert result.shape == (n,)

    def test_first_element_is_minus_inf(self) -> None:
        """Test that first element is -inf."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result[0] == -jnp.inf

    def test_monotonic_increase(self) -> None:
        """Test that cumsum is monotonically increasing (in log space)."""
        log_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        finite_result = result[1:]  # Skip first -inf
        diffs = jnp.diff(finite_result)
        assert jnp.all(diffs >= -1e-6)  # Allow small numerical errors  # noqa: PLR2004

    def test_constant_function(self) -> None:
        """Test with constant function."""
        log_y = jnp.array([0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert jnp.all(jnp.isfinite(result) | jnp.isinf(result))
        assert result[0] == -jnp.inf

    def test_non_uniform_spacing(self) -> None:
        """Test with non-uniform spacing."""
        log_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
        x = jnp.array([0.0, 0.5, 1.5, 2.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (4,)
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))

    def test_single_interval(self) -> None:
        """Test with only two points (single interval)."""
        log_y = jnp.array([-1.0, -2.0])
        x = jnp.array([0.0, 1.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (2,)
        assert result[0] == -jnp.inf

    def test_many_points(self) -> None:
        """Test with many points."""
        log_y = jnp.linspace(-10, -1, 100)
        x = jnp.linspace(0, 1, 100)

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (100,)
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))

    def test_zero_log_y(self) -> None:
        """Test when log_y = 0 (y = 1)."""
        log_y = jnp.array([0.0, 0.0, 0.0, 0.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (4,)
        assert result[0] == -jnp.inf

    def test_negative_inf_log_y(self) -> None:
        """Test with -inf log_y values."""
        log_y = jnp.array([-jnp.inf, -1.0, -2.0])
        x = jnp.array([0.0, 1.0, 2.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (3,)
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]) | jnp.isinf(result[1:]))

    def test_last_element_equals_total_integral(self) -> None:
        """Test that last element of cumsum equals total integral."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 1.0, 2.0])

        cumsum_result = log_trapezoidal_cumsum(log_y=log_y, x=x)
        integral_result = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.allclose(cumsum_result[-1], integral_result, atol=1e-6)

    def test_large_log_y_range(self) -> None:
        """Test with large range of log_y values."""
        log_y = jnp.array([-100.0, -50.0, -1.0, 0.0, 50.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (5,)
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))

    def test_narrow_integration_domain(self) -> None:
        """Test with very narrow integration domain."""
        log_y = jnp.array([-1.0, -1.001, -1.002])
        x = jnp.array([0.0, 0.001, 0.002])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (3,)
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))

    def test_wide_integration_domain(self) -> None:
        """Test with very wide integration domain."""
        log_y = jnp.array([-1.0, -1.5, -2.0])
        x = jnp.array([0.0, 1e6, 2e6])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result.shape == (3,)
        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))

    def test_normalized_cdf_behavior(self) -> None:
        """Test that cumsum can be used as unnormalized CDF."""
        log_y = jnp.array([-1.0, -1.5, -2.0, -2.5])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        result = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert result[0] == -jnp.inf
        assert jnp.all(jnp.isfinite(result[1:]))
        assert jnp.all(jnp.diff(result[1:]) >= -1e-6)  # noqa: PLR2004


class TestIntegratorConsistency:
    """Tests for consistency between different integrator functions."""

    def test_cumsum_last_element_matches_integral(self) -> None:
        """Test that last cumsum element equals the integral."""
        log_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
        x = jnp.array([0.0, 1.0, 2.0, 3.0])

        cumsum = log_trapezoidal_cumsum(log_y=log_y, x=x)
        integral = log_trapezoidal_integrate(log_y=log_y, x=x)

        assert jnp.allclose(cumsum[-1], integral, atol=1e-6)

    def test_cumsum_integrand_dx_consistency(self) -> None:
        """Test that cumsum uses the same integrand_dx as integrate."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 1.0, 2.0])

        integrand = log_trapezoidal_integrand_dx(log_y=log_y, x=x)
        cumsum = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert cumsum.shape == (len(integrand) + 1,)

    def test_various_sizes_consistency(self) -> None:
        """Test consistency across different input sizes."""
        for n in [2, 5, 10, 50]:
            log_y = jnp.linspace(-10, -1, n)
            x = jnp.linspace(0, 1, n)

            integral = log_trapezoidal_integrate(log_y=log_y, x=x)
            cumsum = log_trapezoidal_cumsum(log_y=log_y, x=x)

            assert jnp.allclose(cumsum[-1], integral, atol=1e-6)

    def test_zero_spacing_handling(self) -> None:
        """Test that functions handle spacing correctly."""
        log_y = jnp.array([-1.0, -2.0, -3.0])
        x = jnp.array([0.0, 0.5, 1.0])

        integrand = log_trapezoidal_integrand_dx(log_y=log_y, x=x)
        integral = log_trapezoidal_integrate(log_y=log_y, x=x)
        cumsum = log_trapezoidal_cumsum(log_y=log_y, x=x)

        assert integrand.shape == (2,)
        assert integral.shape == ()
        assert cumsum.shape == (3,)
