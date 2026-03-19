"""Conversion functions for compact binary coalescence."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify


def compute_chirp_mass_from_mass_1_mass_2(mass_1: Array, mass_2: Array) -> Array:
    """Compute the chirp mass from the component masses of the binary.

    Args:
        mass_1: The mass of the first component object in the binary (in solar masses).
        mass_2: The mass of the second component object in the binary (in solar masses).

    Returns:
        The chirp mass of the binary (in solar masses).
    """
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def compute_symmetric_mass_ratio_from_mass_1_mass_2(mass_1: Array, mass_2: Array) -> Array:
    """Compute the symmetric mass ratio from the component masses of the binary.

    Args:
        mass_1: The mass of the first component object in the binary (in solar masses).
        mass_2: The mass of the second component object in the binary (in solar masses).

    Returns:
        The symmetric mass ratio of the binary.
    """
    return jnp.minimum((mass_1 * mass_2) / (mass_1 + mass_2) ** 2, 0.25)


def compute_total_mass_from_mass_1_mass_2(mass_1: Array, mass_2: Array) -> Array:
    """Compute the total mass from the component masses of the binary.

    Args:
        mass_1: The mass of the first component object in the binary (in solar masses).
        mass_2: The mass of the second component object in the binary (in solar masses).

    Returns:
        The total mass of the binary (in solar masses).
    """
    return mass_1 + mass_2


def compute_mass_ratio_from_mass_1_mass_2(mass_1: Array, mass_2: Array) -> Array:
    """Compute the mass ratio from the component masses of the binary.

    Args:
        mass_1: The mass of the first component object in the binary (in solar masses).
        mass_2: The mass of the second component object in the binary (in solar masses).

    Returns:
        The mass ratio, m2/m1, where m2 <= m1.
    """
    checkify.check(jnp.all(mass_1 >= mass_2), "Input 'mass_1' must be >= 'mass_2' element-wise.")
    return mass_2 / mass_1


def compute_source_frame_mass_from_detector_frame_mass(detector_frame_mass: Array, redshift: Array) -> Array:
    """Compute the source frame mass from the detector frame mass.

    Args:
        detector_frame_mass: Detector frame mass (in solar mass).
        redshift: Redshift.

    Returns:
        Source frame mass (in solar mass).
    """
    return detector_frame_mass / (1 + redshift)
