"""Tests for flat lambda CDM cosmological functions."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gwmock_pop.constants import SPEED_OF_LIGHT
from gwmock_pop.cosmology.flat_lambda_cdm import (
    compute_comoving_distance,
    compute_differential_comoving_volume,
    compute_hubble_parameter,
    compute_normalized_hubble_parameter,
)


class TestFlatLambdaCDM:
    """Test suite for flat lambda CDM cosmological functions."""

    def test_compute_normalized_hubble_parameter(self) -> None:
        """Test the normalized Hubble parameter computation."""
        # Test with a simple case
        redshift = jnp.array(1.0)
        omega_m = jnp.array(0.3)

        result = compute_normalized_hubble_parameter(redshift=redshift, omega_m=omega_m)

        # Expected: sqrt(0.3 * (1 + 1)^3 + (1 - 0.3)) = sqrt(0.3 * 8 + 0.7) = sqrt(2.4 + 0.7) = sqrt(3.1)
        expected = jnp.sqrt(0.3 * 8 + 0.7)

        assert isinstance(result, Array)
        assert jnp.isclose(result, expected, atol=1e-10)

    def test_compute_normalized_hubble_parameter_edge_cases(self) -> None:
        """Test normalized Hubble parameter with edge cases."""
        # Test with redshift = 0
        result = compute_normalized_hubble_parameter(redshift=jnp.array(0.0), omega_m=jnp.array(0.3))
        expected = jnp.sqrt(0.3 + (1 - 0.3))  # sqrt(0.3 + 0.7) = 1.0
        assert jnp.isclose(result, expected, atol=1e-10)

        # Test with omega_m = 1.0 (pure radiation)
        result = compute_normalized_hubble_parameter(redshift=jnp.array(1.0), omega_m=jnp.array(1.0))
        expected = jnp.sqrt(1.0 * (1.0 + 1.0) ** 3)
        assert jnp.isclose(result, expected, atol=1e-10)

    def test_compute_hubble_parameter(self) -> None:
        """Test the Hubble parameter computation."""
        redshift = jnp.array(1.0)
        hubble_constant = jnp.array(70.0)  # km/s/Mpc
        omega_m = jnp.array(0.3)

        result = compute_hubble_parameter(redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m)

        # Should equal H0 * normalized_hubble_parameter
        expected_normalized = compute_normalized_hubble_parameter(redshift=redshift, omega_m=omega_m)
        expected = hubble_constant * expected_normalized

        assert isinstance(result, Array)
        assert jnp.isclose(result, expected, atol=1e-10)

    def test_compute_comoving_distance(self) -> None:
        """Test the comoving distance computation."""
        redshift = jnp.array(1.0)
        hubble_constant = jnp.array(70.0)  # km/s/Mpc
        omega_m = jnp.array(0.3)

        result = compute_comoving_distance(
            redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=1000
        )

        assert isinstance(result, Array)
        # Should be positive
        assert result > 0.0

    def test_compute_comoving_distance_edge_cases(self) -> None:
        """Test comoving distance with edge cases."""
        # Test with redshift = 0
        result = compute_comoving_distance(
            redshift=jnp.array(0.0), hubble_constant=jnp.array(70.0), omega_m=jnp.array(0.3), n_grid=1000
        )
        assert isinstance(result, Array)
        assert result == 0.0

    def test_compute_differential_comoving_volume(self) -> None:
        """Test the differential comoving volume computation."""
        redshift = jnp.array(1.0)
        hubble_constant = jnp.array(70.0)  # km/s/Mpc
        omega_m = jnp.array(0.3)

        result = compute_differential_comoving_volume(
            redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=1000
        )

        assert isinstance(result, Array)
        # Should be positive for physical cases
        assert result > 0.0

    def test_compute_differential_comoving_volume_edge_cases(self) -> None:
        """Test differential comoving volume with edge cases."""
        # Test with redshift = 0
        result = compute_differential_comoving_volume(
            redshift=jnp.array(0.0), hubble_constant=jnp.array(70.0), omega_m=jnp.array(0.3), n_grid=1000
        )
        # Should be zero at z=0 (no volume)
        assert isinstance(result, Array)
        assert result >= 0.0

    def test_comprehensive_cosmological_functions(self) -> None:
        """Test comprehensive behavior of all cosmological functions."""
        redshift = jnp.array(2.0)
        hubble_constant = jnp.array(70.0)
        omega_m = jnp.array(0.3)

        # Test all functions work together
        normalized_hubble = compute_normalized_hubble_parameter(redshift=redshift, omega_m=omega_m)
        hubble = compute_hubble_parameter(redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m)
        comoving_distance = compute_comoving_distance(
            redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=1000
        )
        differential_volume = compute_differential_comoving_volume(
            redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=1000
        )

        # Verify types
        assert isinstance(normalized_hubble, Array)
        assert isinstance(hubble, Array)
        assert isinstance(comoving_distance, Array)
        assert isinstance(differential_volume, Array)

        # Verify reasonable ranges
        assert normalized_hubble > 0.0
        assert hubble > 0.0
        assert comoving_distance >= 0.0
        assert differential_volume >= 0.0

    def test_vectorized_inputs(self) -> None:
        """Test that functions work with vectorized inputs."""
        redshifts = jnp.array([0.0, 1.0, 2.0])
        omega_m = jnp.array(0.3)

        # Test normalized Hubble parameter with array input
        result = compute_normalized_hubble_parameter(redshift=redshifts, omega_m=omega_m)

        assert isinstance(result, Array)
        assert result.shape == (3,)

    def test_numerical_integration_accuracy(self) -> None:
        """Test that the numerical integration produces reasonable results."""
        # Test with known simple case: omega_m = 1.0 (Einstein-de Sitter)
        redshift = jnp.array(1.0)
        hubble_constant = jnp.array(70.0)
        omega_m = jnp.array(1.0)

        # Exact comoving distance for Einstein-de Sitter:
        # d_C = 2*c/H_0 * (1 - 1/sqrt(1+z))
        distance = compute_comoving_distance(
            redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=1000
        )
        expected = SPEED_OF_LIGHT / 1000 / hubble_constant * 2.0 * (1.0 - 1.0 / jnp.sqrt(1.0 + redshift))

        assert jnp.isclose(distance, expected, rtol=1e-3)

    def test_parameter_ranges(self) -> None:
        """Test that functions handle parameter ranges appropriately."""
        redshift = jnp.array(0.5)
        hubble_constant = jnp.array(50.0)  # Lower H0
        omega_m = jnp.array(0.5)  # Mid-range matter density

        result = compute_comoving_distance(
            redshift=redshift, hubble_constant=hubble_constant, omega_m=omega_m, n_grid=1000
        )

        assert isinstance(result, Array)
