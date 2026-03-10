"""Unit tests for gwsim_pop.constants."""

from gwsim_pop.constants import SPEED_OF_LIGHT

SPEED_OF_LIGHT_REFERENCE = 299792458.0


class TestConstants:
    """Test the constants."""

    def test_speed_of_light(self):
        """Compare the speed of light with the expected value."""
        assert SPEED_OF_LIGHT == SPEED_OF_LIGHT_REFERENCE
