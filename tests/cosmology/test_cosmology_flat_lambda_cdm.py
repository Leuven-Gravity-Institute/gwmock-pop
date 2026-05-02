"""Tests for flat lambda CDM cosmological functions."""

from __future__ import annotations

import jax.numpy as jnp
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from jax import Array

from gwmock_pop.constants import SPEED_OF_LIGHT
from gwmock_pop.cosmology.flat_lambda_cdm import (
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
    build_distance_lookup,
    compute_comoving_distance,
    compute_differential_comoving_volume,
    compute_hubble_parameter,
    compute_luminosity_distance,
    compute_normalized_hubble_parameter,
    compute_redshift_from_luminosity_distance,
)

_REFERENCE_COSMOLOGY = FlatLambdaCDM(H0=PLANCK18_H0_KM_S_MPC, Om0=PLANCK18_OMEGA_M)


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

    def test_compute_luminosity_distance(self) -> None:
        """Luminosity distance is ``(1 + z)`` times the comoving distance."""
        redshift = jnp.array(1.0)
        hubble_constant = jnp.array(70.0)
        omega_m = jnp.array(0.3)

        luminosity_distance = compute_luminosity_distance(
            redshift=redshift,
            hubble_constant=hubble_constant,
            omega_m=omega_m,
            n_grid=1000,
        )
        comoving_distance = compute_comoving_distance(
            redshift=redshift,
            hubble_constant=hubble_constant,
            omega_m=omega_m,
            n_grid=1000,
        )

        assert jnp.isclose(luminosity_distance, (1.0 + redshift) * comoving_distance)

    def test_build_distance_lookup_starts_at_zero_and_increases_monotonically(self) -> None:
        """The lookup table should span physical, monotonically increasing distances."""
        redshift_grid, comoving_distance_grid, luminosity_distance_grid = build_distance_lookup()

        assert redshift_grid.shape == comoving_distance_grid.shape == luminosity_distance_grid.shape
        assert redshift_grid[0] == 0.0
        assert comoving_distance_grid[0] == 0.0
        assert luminosity_distance_grid[0] == 0.0
        assert jnp.all(jnp.diff(redshift_grid) > 0.0)
        assert jnp.all(jnp.diff(comoving_distance_grid) >= 0.0)
        assert jnp.all(jnp.diff(luminosity_distance_grid) >= 0.0)

    @staticmethod
    def _luminosity_distance_at_redshift(redshift: float) -> float:
        """Return the Astropy luminosity distance for the reference cosmology."""
        return float(_REFERENCE_COSMOLOGY.luminosity_distance(redshift).to_value(u.Mpc))

    @staticmethod
    def _astropy_redshift_from_distance(distance_mpc: float) -> float:
        """Return the Astropy redshift for a luminosity distance."""
        return float(z_at_value(_REFERENCE_COSMOLOGY.luminosity_distance, distance_mpc * u.Mpc))

    @staticmethod
    def _relative_error(measured: float, expected: float) -> float:
        """Return the relative error against a non-zero reference."""
        return abs(measured - expected) / expected

    def test_compute_redshift_from_luminosity_distance_matches_astropy_reference(self) -> None:
        """The lookup-based inversion matches the Astropy cosmology reference."""
        for expected_redshift in (0.01, 0.1, 1.0, 5.0, 10.0):
            luminosity_distance = self._luminosity_distance_at_redshift(expected_redshift)
            measured_redshift = float(compute_redshift_from_luminosity_distance(jnp.array(luminosity_distance)))
            reference_redshift = self._astropy_redshift_from_distance(luminosity_distance)
            assert self._relative_error(measured_redshift, reference_redshift) < 1e-3
