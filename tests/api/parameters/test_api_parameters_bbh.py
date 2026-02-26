"""Unit tests for BBHParameters dataclass."""

import jax.numpy as jnp

from gwsim_pop.api.parameters.bbh import BBHParameters


class TestBBHParameters:
    """Test class for BBHParameters dataclass."""

    def test_bbh_parameters_initialization(self):
        """Test BBHParameters dataclass initialization."""
        # Test with scalar values
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
        )

        assert params.mass_1 == jnp.array(2.0)
        assert params.mass_2 == jnp.array(1.0)
        assert params.redshift == jnp.array(0.0)
        assert params.extra == {}

        # Test with array values
        params = BBHParameters(
            mass_1=jnp.array([2.0, 3.0]),
            mass_2=jnp.array([1.0, 2.0]),
            spin_1x=jnp.array([0.0, 0.0]),
            spin_1y=jnp.array([0.0, 0.0]),
            spin_1z=jnp.array([0.0, 0.0]),
            spin_2x=jnp.array([0.0, 0.0]),
            spin_2y=jnp.array([0.0, 0.0]),
            spin_2z=jnp.array([0.0, 0.0]),
            eccentricity=jnp.array([0.0, 0.0]),
            distance=jnp.array([100.0, 150.0]),
            coa_phase=jnp.array([0.0, 0.0]),
            inclination=jnp.array([0.0, 0.0]),
            theta_jn=jnp.array([0.0, 0.0]),
            long_asc_nodes=jnp.array([0.0, 0.0]),
            mean_per_ano=jnp.array([0.0, 0.0]),
            coa_time=jnp.array([0.0, 0.0]),
            right_ascension=jnp.array([0.0, 0.0]),
            declination=jnp.array([0.0, 0.0]),
            polarization_angle=jnp.array([0.0, 0.0]),
            redshift=jnp.array([0.0, 0.1]),
            f_ref=jnp.array([20.0, 20.0]),
        )

        assert jnp.array_equal(params.mass_1, jnp.array([2.0, 3.0]))
        assert jnp.array_equal(params.mass_2, jnp.array([1.0, 2.0]))
        assert jnp.array_equal(params.redshift, jnp.array([0.0, 0.1]))

    def test_bbh_parameters_chirp_mass(self):
        """Test chirp_mass property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
        )

        result = params.chirp_mass
        expected = (2.0 * 1.0) ** 0.6 / (2.0 + 1.0) ** 0.2
        assert jnp.isclose(result, expected)

        # Test with arrays
        params = BBHParameters(
            mass_1=jnp.array([2.0, 3.0]),
            mass_2=jnp.array([1.0, 2.0]),
            spin_1x=jnp.array([0.0, 0.0]),
            spin_1y=jnp.array([0.0, 0.0]),
            spin_1z=jnp.array([0.0, 0.0]),
            spin_2x=jnp.array([0.0, 0.0]),
            spin_2y=jnp.array([0.0, 0.0]),
            spin_2z=jnp.array([0.0, 0.0]),
            eccentricity=jnp.array([0.0, 0.0]),
            distance=jnp.array([100.0, 100.0]),
            coa_phase=jnp.array([0.0, 0.0]),
            inclination=jnp.array([0.0, 0.0]),
            theta_jn=jnp.array([0.0, 0.0]),
            long_asc_nodes=jnp.array([0.0, 0.0]),
            mean_per_ano=jnp.array([0.0, 0.0]),
            coa_time=jnp.array([0.0, 0.0]),
            right_ascension=jnp.array([0.0, 0.0]),
            declination=jnp.array([0.0, 0.0]),
            polarization_angle=jnp.array([0.0, 0.0]),
            redshift=jnp.array([0.0, 0.0]),
            f_ref=jnp.array([20.0, 20.0]),
        )

        result = params.chirp_mass
        expected = jnp.array([(2.0 * 1.0) ** 0.6 / (2.0 + 1.0) ** 0.2, (3.0 * 2.0) ** 0.6 / (3.0 + 2.0) ** 0.2])
        assert jnp.allclose(result, expected)

    def test_bbh_parameters_symmetric_mass_ratio(self):
        """Test symmetric_mass_ratio property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
        )

        result = params.symmetric_mass_ratio
        expected = (2.0 * 1.0) / (2.0 + 1.0) ** 2
        assert jnp.isclose(result, expected)

        # Test with equal masses (should be 0.25, the maximum)
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(2.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
        )

        result = params.symmetric_mass_ratio
        assert jnp.isclose(result, 0.25)

    def test_bbh_parameters_total_mass(self):
        """Test total_mass property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
        )

        result = params.total_mass
        expected = 3.0
        assert jnp.isclose(result, expected)

    def test_bbh_parameters_mass_ratio(self):
        """Test mass_ratio property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
        )

        result = params.mass_ratio
        expected = 0.5
        assert jnp.isclose(result, expected)

    def test_bbh_parameters_source_frame_mass_1(self):
        """Test source_frame_mass_1 property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.5),
            f_ref=jnp.array(20.0),
        )

        result = params.source_frame_mass_1
        expected = 2.0 / (1.0 + 0.5)
        assert jnp.isclose(result, expected)

    def test_bbh_parameters_source_frame_mass_2(self):
        """Test source_frame_mass_2 property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.5),
            f_ref=jnp.array(20.0),
        )

        result = params.source_frame_mass_2
        expected = 1.0 / (1.0 + 0.5)
        assert jnp.isclose(result, expected)

    def test_bbh_parameters_source_frame_chirp_mass(self):
        """Test source_frame_chirp_mass property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.5),
            f_ref=jnp.array(20.0),
        )

        # First compute the chirp mass in detector frame
        detector_chirp_mass = (2.0 * 1.0) ** 0.6 / (2.0 + 1.0) ** 0.2
        result = params.source_frame_chirp_mass
        expected = detector_chirp_mass / (1.0 + 0.5)
        assert jnp.isclose(result, expected)

    def test_bbh_parameters_source_frame_total_mass(self):
        """Test source_frame_total_mass property."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.5),
            f_ref=jnp.array(20.0),
        )

        # First compute the total mass in detector frame
        detector_total_mass = 3.0
        result = params.source_frame_total_mass
        expected = detector_total_mass / (1.0 + 0.5)
        assert jnp.isclose(result, expected)

    def test_bbh_parameters_extra_field(self):
        """Test extra field functionality."""
        params = BBHParameters(
            mass_1=jnp.array(2.0),
            mass_2=jnp.array(1.0),
            spin_1x=jnp.array(0.0),
            spin_1y=jnp.array(0.0),
            spin_1z=jnp.array(0.0),
            spin_2x=jnp.array(0.0),
            spin_2y=jnp.array(0.0),
            spin_2z=jnp.array(0.0),
            eccentricity=jnp.array(0.0),
            distance=jnp.array(100.0),
            coa_phase=jnp.array(0.0),
            inclination=jnp.array(0.0),
            theta_jn=jnp.array(0.0),
            long_asc_nodes=jnp.array(0.0),
            mean_per_ano=jnp.array(0.0),
            coa_time=jnp.array(0.0),
            right_ascension=jnp.array(0.0),
            declination=jnp.array(0.0),
            polarization_angle=jnp.array(0.0),
            redshift=jnp.array(0.0),
            f_ref=jnp.array(20.0),
            extra={"custom_param": "value"},
        )

        assert params.extra["custom_param"] == "value"
