"""Conversion functions for compact binary coalescence."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


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
