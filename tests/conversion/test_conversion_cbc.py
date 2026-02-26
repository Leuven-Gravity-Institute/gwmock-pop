"""Unit tests for CBC conversion functions."""

import jax.numpy as jnp
import pytest

from gwsim_pop.conversion.cbc import (
    compute_chirp_mass_from_mass_1_mass_2,
    compute_mass_ratio_from_mass_1_mass_2,
    compute_source_frame_mass_from_detector_frame_mass,
    compute_symmetric_mass_ratio_from_mass_1_mass_2,
    compute_total_mass_from_mass_1_mass_2,
)


class TestCBCConversionFunctions:
    """Test class for CBC conversion functions."""

    def test_compute_chirp_mass_from_mass_1_mass_2(self):
        """Test compute_chirp_mass_from_mass_1_mass_2 function."""
        # Test with scalar values
        mass_1 = jnp.array(1.4)
        mass_2 = jnp.array(1.6)
        result = compute_chirp_mass_from_mass_1_mass_2(mass_1, mass_2)
        expected = (1.4 * 1.6) ** 0.6 / (1.4 + 1.6) ** 0.2
        assert jnp.isclose(result, expected)

        # Test with arrays
        mass_1 = jnp.array([1.0, 2.0, 3.0])
        mass_2 = jnp.array([1.0, 1.0, 1.0])
        result = compute_chirp_mass_from_mass_1_mass_2(mass_1, mass_2)
        # Compute expected values manually for clarity
        expected_vals = []
        for i in range(len(mass_1)):
            m1 = mass_1[i].item()
            m2 = mass_2[i].item()
            val = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
            expected_vals.append(val)
        expected = jnp.array(expected_vals)
        # Use more liberal tolerance
        assert jnp.allclose(result, expected, atol=1e-5, rtol=1e-5)

    def test_compute_symmetric_mass_ratio_from_mass_1_mass_2(self):
        """Test compute_symmetric_mass_ratio_from_mass_1_mass_2 function."""
        # Test with scalar values
        mass_1 = jnp.array(2.0)
        mass_2 = jnp.array(1.0)
        result = compute_symmetric_mass_ratio_from_mass_1_mass_2(mass_1, mass_2)
        expected = (2.0 * 1.0) / (2.0 + 1.0) ** 2
        assert jnp.isclose(result, expected)

        # Test with arrays
        mass_1 = jnp.array([2.0, 3.0, 4.0])
        mass_2 = jnp.array([1.0, 1.0, 2.0])
        result = compute_symmetric_mass_ratio_from_mass_1_mass_2(mass_1, mass_2)
        expected = jnp.array([2.0 / 9.0, 3.0 / 16.0, 8.0 / 36.0])
        assert jnp.allclose(result, expected)

        # Test with equal masses (should be 0.25, the maximum)
        mass_1 = jnp.array(2.0)
        mass_2 = jnp.array(2.0)
        result = compute_symmetric_mass_ratio_from_mass_1_mass_2(mass_1, mass_2)
        assert jnp.isclose(result, 0.25)

    def test_compute_total_mass_from_mass_1_mass_2(self):
        """Test compute_total_mass_from_mass_1_mass_2 function."""
        # Test with scalar values
        mass_1 = jnp.array(2.0)
        mass_2 = jnp.array(3.0)
        result = compute_total_mass_from_mass_1_mass_2(mass_1, mass_2)
        expected = 5.0
        assert jnp.isclose(result, expected)

        # Test with arrays
        mass_1 = jnp.array([1.0, 2.0, 3.0])
        mass_2 = jnp.array([4.0, 5.0, 6.0])
        result = compute_total_mass_from_mass_1_mass_2(mass_1, mass_2)
        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_compute_mass_ratio_from_mass_1_mass_2(self):
        """Test compute_mass_ratio_from_mass_1_mass_2 function."""
        # Test with scalar values
        mass_1 = jnp.array(2.0)
        mass_2 = jnp.array(1.0)
        result = compute_mass_ratio_from_mass_1_mass_2(mass_1, mass_2)
        expected = 0.5
        assert jnp.isclose(result, expected)

        # Test with arrays
        mass_1 = jnp.array([2.0, 4.0, 6.0])
        mass_2 = jnp.array([1.0, 2.0, 3.0])
        result = compute_mass_ratio_from_mass_1_mass_2(mass_1, mass_2)
        expected = jnp.array([0.5, 0.5, 0.5])
        assert jnp.allclose(result, expected)

        # Test with mass_1 < mass_2 (should raise ValueError)
        mass_1 = jnp.array([1.0])
        mass_2 = jnp.array([2.0])
        with pytest.raises(
            ValueError, match="Input 'mass_1' must be >= 'mass_2' element-wise"
        ):  # checkify will raise an exception
            compute_mass_ratio_from_mass_1_mass_2(mass_1, mass_2)

    def test_compute_source_frame_mass_from_detector_frame_mass(self):
        """Test compute_source_frame_mass_from_detector_frame_mass function."""
        # Test with scalar values
        detector_frame_mass = jnp.array(2.0)
        redshift = jnp.array(1.0)
        result = compute_source_frame_mass_from_detector_frame_mass(detector_frame_mass, redshift)
        expected = 2.0 / (1.0 + 1.0)
        assert jnp.isclose(result, expected)

        # Test with arrays
        detector_frame_mass = jnp.array([2.0, 4.0, 6.0])
        redshift = jnp.array([1.0, 2.0, 3.0])
        result = compute_source_frame_mass_from_detector_frame_mass(detector_frame_mass, redshift)
        expected = jnp.array([2.0 / 2.0, 4.0 / 3.0, 6.0 / 4.0])
        assert jnp.allclose(result, expected)

        # Test with zero redshift
        detector_frame_mass = jnp.array(5.0)
        redshift = jnp.array(0.0)
        result = compute_source_frame_mass_from_detector_frame_mass(detector_frame_mass, redshift)
        assert jnp.isclose(result, detector_frame_mass)
