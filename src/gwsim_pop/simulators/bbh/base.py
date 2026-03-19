"""Base class for binary black hole population simulators."""

from __future__ import annotations

from gwsim_pop.mixins.random import RandomMixin
from gwsim_pop.simulators.simulator import Simulator


class BBHSimulator(RandomMixin, Simulator):
    """Base class for binary black hole population simulator."""

    def __init__(self, *args: object, seed: int | None = None, **kwargs: object) -> None:
        """Initialize the instance.

        Args:
            *args: Positional arguments.
            seed: Random seed for reproducibility.
            **kwargs: Keyword arguments.
        """
        super().__init__(seed=seed, **kwargs)

    @property
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters.

        Definition of the parameters:

        - detector_frame_mass_1: The mass of the first component object in the binary (in solar masses) in detector frame.
        - detector_frame_mass_2: The mass of the second component object in the binary (in solar masses) in the detector frame.
        - spin_1x: The x component of the first binary component's dimensionless spin.
        - spin_1y: The y component of the first binary component's dimensionless spin.
        - spin_1z: The z component of the first binary component's dimensionless spin.
        - spin_2x: The x component of the second binary component's dimensionless spin.
        - spin_2y: The y component of the second binary component's dimensionless spin.
        - spin_2z: The z component of the second binary component's dimensionless spin.
        - eccentricity: Eccentricity.
        - distance: Luminosity distance to the binary (in Mpc).
        - coa_phase: Coalescence phase of the binary (in rad).
        - inclination: Inclination (rad), defined as the angle between the orbital angular momentum L and the line-of-sight at the reference frequency.
        - theta_jn: The angle between the total angular momentum J and the line-of-sight.
        - long_asc_node: Longitude of ascending nodes axis (rad).
        - mean_per_ano: Mean anomaly of the periastron (rad).
        - coa_time: Coalescence time (s) is the time when a GW reaches the origin of a certain coordinate system.
        - right_ascension: Right ascension (rad).
        - declination: Declination (rad).
        - polarization_angle: Polarization angle (rad) in a certain coordinate system.
        - redshift: Redshift.
        - f_ref: The reference frequency that defines the spin components.

        Returns:
            List of parameter names.

        """
        # Allow subclasses to override with dynamic parameter names
        if hasattr(self, "_parameter_names"):
            return self._parameter_names
        return [
            "detector_frame_mass_1",
            "detector_frame_mass_2",
            "spin_1x",
            "spin_1y",
            "spin_1z",
            "spin_2x",
            "spin_2y",
            "spin_2z",
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
        ]

    @property
    def source_type(self) -> str:
        """Get the source type.

        Returns:
            Source type string.

        """
        return "bbh"
