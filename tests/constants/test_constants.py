"""Unit tests for gwmock_pop.constants."""

from gwmock_pop import CBC_PARAMETER_NAMES
from gwmock_pop.constants import SPEED_OF_LIGHT

EXPECTED_CBC_PARAMETER_NAMES = frozenset(
    {
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "spin_1x",
        "spin_1y",
        "spin_1z",
        "spin_2x",
        "spin_2y",
        "spin_2z",
        "lambda_1",
        "lambda_2",
        "eccentricity",
        "distance",
        "coa_phase",
        "inclination",
        "theta_jn",
        "long_asc_node",
        "mean_per_ano",
        "coa_time",
        "right_ascension",
        "declination",
        "polarization_angle",
        "redshift",
        "f_ref",
    }
)
SPEED_OF_LIGHT_REFERENCE = 299792458.0


class TestConstants:
    """Test the constants."""

    def test_speed_of_light(self):
        """Compare the speed of light with the expected value."""
        assert SPEED_OF_LIGHT == SPEED_OF_LIGHT_REFERENCE

    def test_cbc_parameter_names_is_public_frozenset(self) -> None:
        """CBC parameter names are exported as a 23-name frozenset."""
        assert isinstance(CBC_PARAMETER_NAMES, frozenset)
        assert len(CBC_PARAMETER_NAMES) == 23

    def test_cbc_parameter_names_match_notes_table(self) -> None:
        """CBC parameter names match the canonical NOTES table."""
        assert CBC_PARAMETER_NAMES == EXPECTED_CBC_PARAMETER_NAMES
