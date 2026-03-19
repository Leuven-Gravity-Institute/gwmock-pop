"""Tests for smoothing distribution functions."""

import jax.numpy as jnp

from gwmock_pop.distributions.smoothing import log_planck_tapering_function, planck_tapering_function


class TestPlanckTaperingFunction:
    """Tests for planck_tapering_function."""

    def test_basic_functionality(self):
        """Test basic functionality of the tapering function."""
        # Test with simple values
        x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        x_min = 1.0
        delta = 1.0

        result = planck_tapering_function(x, x_min, delta)
        assert result.shape == x.shape

        # Check that values are between 0 and 1
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_below_transition_region(self):
        """Test values below the transition region."""
        x = jnp.array([0.5])
        x_min = 1.0
        delta = 1.0

        result = planck_tapering_function(x, x_min, delta)
        assert result[0] == 0.0

    def test_above_transition_region(self):
        """Test values above the transition region."""
        x = jnp.array([2.5])
        x_min = 1.0
        delta = 1.0

        result = planck_tapering_function(x, x_min, delta)
        assert result[0] == 1.0

    def test_at_transition_region_edges(self):
        """Test values at the edges of the transition region."""
        x = jnp.array([1.0, 2.0])  # edges of [x_min, x_min + delta]
        x_min = 1.0
        delta = 1.0

        result = planck_tapering_function(x, x_min, delta)
        assert result[0] == 0.0  # At x_min, should be 0
        assert result[1] == 1.0  # At x_min + delta, should be 1

    def test_array_input(self):
        """Test with multiple values in array."""
        x = jnp.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
        x_min = 1.0
        delta = 1.0

        result = planck_tapering_function(x, x_min, delta)
        assert len(result) == len(x)

        # Verify boundaries
        assert result[0] == 0.0  # Below transition
        assert result[-1] == 1.0  # Above transition


class TestLogPlanckTaperingFunction:
    """Tests for log_planck_tapering_function."""

    def test_basic_functionality(self):
        """Test basic functionality of the log tapering function."""
        x = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        x_min = 1.0
        delta = 1.0

        result = log_planck_tapering_function(x, x_min, delta)
        assert result.shape == x.shape

    def test_below_transition_region(self):
        """Test values below the transition region."""
        x = jnp.array([0.5])
        x_min = 1.0
        delta = 1.0

        result = log_planck_tapering_function(x, x_min, delta)
        assert jnp.isinf(result[0])
        assert result[0] < 0

    def test_above_transition_region(self):
        """Test values above the transition region."""
        x = jnp.array([2.5])
        x_min = 1.0
        delta = 1.0

        result = log_planck_tapering_function(x, x_min, delta)
        assert result[0] == 0.0

    def test_at_transition_region_edges(self):
        """Test values at the edges of the transition region."""
        x = jnp.array([1.0, 2.0])  # edges of [x_min, x_min + delta]
        x_min = 1.0
        delta = 1.0

        result = log_planck_tapering_function(x, x_min, delta)
        # At x_min, should be -inf
        assert jnp.isinf(result[0])
        assert result[0] < 0
        assert result[1] == 0.0  # At x_min + delta, should be 0

    def test_array_input(self):
        """Test with multiple values in array."""
        x = jnp.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
        x_min = 1.0
        delta = 1.0

        result = log_planck_tapering_function(x, x_min, delta)
        assert len(result) == len(x)

        # Verify boundaries
        # Below transition
        assert jnp.isinf(result[0])
        assert result[0] < 0
        assert result[-1] == 0.0  # Above transition


class TestEdgeCases:
    """Tests for edge cases in both tapering functions."""

    def test_negative_delta(self):
        """Test with negative delta."""
        x = jnp.array([0.5, 1.0, 1.5])
        x_min = 1.0
        delta = -1.0

        # These should work but produce strange results
        result1 = planck_tapering_function(x, x_min, delta)
        result2 = log_planck_tapering_function(x, x_min, delta)
        assert result1.shape == x.shape
        assert result2.shape == x.shape

    def test_extreme_values(self):
        """Test with extreme input values."""
        x = jnp.array([0.0, 1e-10, 1.0, 1000.0, 1e10])
        x_min = 1.0
        delta = 1.0

        result1 = planck_tapering_function(x, x_min, delta)
        result2 = log_planck_tapering_function(x, x_min, delta)

        assert result1.shape == x.shape
        assert result2.shape == x.shape

    def test_small_values(self):
        """Test with very small input values."""
        x = jnp.array([1e-10, 1e-5, 1.0])
        x_min = 1.0
        delta = 1.0

        result1 = planck_tapering_function(x, x_min, delta)
        result2 = log_planck_tapering_function(x, x_min, delta)

        assert result1.shape == x.shape
        assert result2.shape == x.shape

    def test_large_values(self):
        """Test with large input values."""
        x = jnp.array([1000.0, 2000.0, 3000.0])
        x_min = 1.0
        delta = 1.0

        result1 = planck_tapering_function(x, x_min, delta)
        result2 = log_planck_tapering_function(x, x_min, delta)

        assert result1.shape == x.shape
        assert result2.shape == x.shape
