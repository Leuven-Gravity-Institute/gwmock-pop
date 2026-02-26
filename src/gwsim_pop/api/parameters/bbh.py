"""Dataclass for binary black holes parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jax import Array
from jax.tree_util import register_dataclass

from gwsim_pop.conversion.cbc import (
    compute_chirp_mass_from_mass_1_mass_2,
    compute_mass_ratio_from_mass_1_mass_2,
    compute_source_frame_mass_from_detector_frame_mass,
    compute_symmetric_mass_ratio_from_mass_1_mass_2,
    compute_total_mass_from_mass_1_mass_2,
)


@register_dataclass
@dataclass(frozen=True)
class BBHParameters:
    """Parameters for binary black holes."""

    mass_1: Array
    """The mass of the first component object in the binary (in solar masses)."""
    mass_2: Array
    """The mass of the second component object in the binary (in solar masses)."""
    spin_1x: Array
    """The x component of the first binary component's dimensionless spin."""
    spin_1y: Array
    """The y component of the first binary component's dimensionless spin."""
    spin_1z: Array
    """The z component of the first binary component's dimensionless spin."""
    spin_2x: Array
    """The x component of the second binary component's dimensionless spin."""
    spin_2y: Array
    """The y component of the second binary component's dimensionless spin."""
    spin_2z: Array
    """The z component of the second binary component's dimensionless spin."""
    eccentricity: Array
    """Eccentricity."""
    distance: Array
    """Luminosity distance to the binary (in Mpc)."""
    coa_phase: Array
    """Coalescence phase of the binary (in rad)."""
    inclination: Array
    """Inclination (rad), defined as the angle between the orbital angular momentum L and the line-of-sight at the reference frequency."""
    theta_jn: Array
    """The angle between the total angular momentum J and the line-of-sight."""
    long_asc_nodes: Array
    """Longitude of ascending nodes axis (rad)."""
    mean_per_ano: Array
    """Mean anomaly of the periastron (rad)."""
    coa_time: Array
    """Coalescence time (s) is the time when a GW reaches the origin of a certain coordinate system."""
    right_ascension: Array
    """Right ascension (rad)."""
    declination: Array
    """Declination (rad)."""
    polarization_angle: Array
    """Polarization angle (rad) in a certain coordinate system."""
    redshift: Array
    """Redshift."""
    f_ref: Array
    """The reference frequency that defines the spin components."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Extra parameters."""

    @property
    def chirp_mass(self) -> Array:
        """Get the chirp mass from the component masses.

        Returns:
            The chirp mass of the binary (in solar masses).
        """
        return compute_chirp_mass_from_mass_1_mass_2(mass_1=self.mass_1, mass_2=self.mass_2)

    @property
    def symmetric_mass_ratio(self) -> Array:
        """Get the symmetric mass ratio from the component masses.

        Returns:
            The symmetric mass ratio of the binary.
        """
        return compute_symmetric_mass_ratio_from_mass_1_mass_2(mass_1=self.mass_1, mass_2=self.mass_2)

    @property
    def total_mass(self) -> Array:
        """Get the total mass from the component masses.

        Returns:
            The total mass of the binary (in solar masses).
        """
        return compute_total_mass_from_mass_1_mass_2(mass_1=self.mass_1, mass_2=self.mass_2)

    @property
    def mass_ratio(self) -> Array:
        """Get the mass ratio from the component masses.

        Returns:
            The mass ratio, mass_2 / mass_1, where mass_2 <= mass_1.
        """
        return compute_mass_ratio_from_mass_1_mass_2(mass_1=self.mass_1, mass_2=self.mass_2)

    @property
    def source_frame_mass_1(self) -> Array:
        """Get the source frame mass of the first component.

        Returns:
            The mass of the first component object in the source frame (in solar masses).
        """
        return compute_source_frame_mass_from_detector_frame_mass(
            detector_frame_mass=self.mass_1, redshift=self.redshift
        )

    @property
    def source_frame_mass_2(self) -> Array:
        """Get the source frame mass of the second component.

        Returns:
            The mass of the second component object in the source frame (in solar masses).
        """
        return compute_source_frame_mass_from_detector_frame_mass(
            detector_frame_mass=self.mass_2, redshift=self.redshift
        )

    @property
    def source_frame_chirp_mass(self) -> Array:
        """Get the source frame chirp mass.

        Returns:
            The chirp mass of the binary in the source frame (in solar masses).
        """
        return compute_source_frame_mass_from_detector_frame_mass(
            detector_frame_mass=self.chirp_mass, redshift=self.redshift
        )

    @property
    def source_frame_total_mass(self) -> Array:
        """Get the source frame total mass.

        Returns:
            The total mass of the binary in the source frame (in solar masses).
        """
        return compute_source_frame_mass_from_detector_frame_mass(
            detector_frame_mass=self.total_mass, redshift=self.redshift
        )
