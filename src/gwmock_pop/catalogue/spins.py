"""Spin angles to Cartesian spin components, as LALSimulation converts them.

Catalogues in the bilby/gwfast convention describe a precessing binary by the angle between the
total angular momentum and the line of sight (``theta_jn``), the spin magnitudes and tilts, and two
azimuths (``phi_12``, ``phi_jl``), all defined at a reference frequency. A waveform takes the
inclination of the orbital angular momentum and the Cartesian spin components instead.
:func:`spin_angles_to_components` is that conversion, ported to arrays from LALSimulation's
``XLALSimInspiralTransformPrecessingNewInitialConditions`` step for step -- same rotations, same
order, same post-Newtonian orbital angular momentum -- so a catalogue is read in the frame LAL
would give it, without calling LAL once per row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: Solar mass in seconds, G Msun / c^3, LAL's ``LAL_MTSUN_SI``.
_LAL_MTSUN_SI = 4.925490947641266978197229498498379006e-6


def _rotate_z(angle: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    """LAL's ``ROTATEZ``: rotate (x, y, z) about the z axis by *angle*."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return x * cosine - y * sine, x * sine + y * cosine, z


def _rotate_y(angle: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    """LAL's ``ROTATEY``: rotate (x, y, z) about the y axis by *angle*."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return x * cosine + z * sine, y, -x * sine + z * cosine


def spin_angles_to_components(  # noqa: PLR0913, PLR0917 - LAL's own argument list, in its order
    theta_jn: ArrayLike,
    phi_jl: ArrayLike,
    tilt_1: ArrayLike,
    tilt_2: ArrayLike,
    phi_12: ArrayLike,
    spin_magnitude_1: ArrayLike,
    spin_magnitude_2: ArrayLike,
    mass_1: ArrayLike,
    mass_2: ArrayLike,
    reference_frequency: ArrayLike,
    reference_phase: ArrayLike,
) -> tuple[np.ndarray, ...]:
    """Convert spin angles to the inclination and Cartesian spin components.

    Args:
        theta_jn: Angle between the total angular momentum and the line of sight, rad.
        phi_jl: Azimuth of the orbital angular momentum about the total one, rad.
        tilt_1: Angle between the primary's spin and the orbital angular momentum, rad.
        tilt_2: The same for the secondary, rad.
        phi_12: Azimuth between the two spins' projections on the orbital plane, rad.
        spin_magnitude_1: Dimensionless spin magnitude of the primary.
        spin_magnitude_2: Dimensionless spin magnitude of the secondary.
        mass_1: Primary mass in solar masses, in the frame the waveform uses (LAL and the
            catalogues this reads use detector-frame masses).
        mass_2: Secondary mass in solar masses, same frame.
        reference_frequency: Frequency the angles are defined at, Hz.
        reference_phase: Orbital phase at the reference frequency, rad.

    Returns:
        ``(inclination, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)``, each an array of
        the broadcast input shape.
    """
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2, mass_1, mass_2, f_ref, phi_ref = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=float)
            for value in (
                theta_jn,
                phi_jl,
                tilt_1,
                tilt_2,
                phi_12,
                spin_magnitude_1,
                spin_magnitude_2,
                mass_1,
                mass_2,
                reference_frequency,
                reference_phase,
            )
        )
    )
    # Starting frame: L along z, spin 1 at azimuth phi_ref, spin 2 at phi_12 + phi_ref.
    s1x, s1y, s1z = np.sin(tilt_1) * np.cos(phi_ref), np.sin(tilt_1) * np.sin(phi_ref), np.cos(tilt_1)
    s2x, s2y, s2z = (
        np.sin(tilt_2) * np.cos(phi_12 + phi_ref),
        np.sin(tilt_2) * np.sin(phi_12 + phi_ref),
        np.cos(tilt_2),
    )

    total = mass_1 + mass_2
    eta = mass_1 * mass_2 / total / total
    v0 = np.cbrt(np.pi * f_ref * total * _LAL_MTSUN_SI)
    l_magnitude = total * total * eta / v0 * (1.0 + v0 * v0 * (1.5 + eta / 6.0))
    jx = mass_1 * mass_1 * chi_1 * s1x + mass_2 * mass_2 * chi_2 * s2x
    jy = mass_1 * mass_1 * chi_1 * s1y + mass_2 * mass_2 * chi_2 * s2y
    jz = l_magnitude + mass_1 * mass_1 * chi_1 * s1z + mass_2 * mass_2 * chi_2 * s2z
    j_norm = np.sqrt(jx * jx + jy * jy + jz * jz)
    theta_0 = np.arccos(jz / j_norm)
    phi_0 = np.arctan2(jy / j_norm, jx / j_norm)

    # Rotations 1-2: bring J onto the z axis.
    s1x, s1y, s1z = _rotate_z(-phi_0, s1x, s1y, s1z)
    s2x, s2y, s2z = _rotate_z(-phi_0, s2x, s2y, s2z)
    lx, ly, lz = _rotate_y(-theta_0, np.zeros_like(s1z), np.zeros_like(s1z), np.ones_like(s1z))
    s1x, s1y, s1z = _rotate_y(-theta_0, s1x, s1y, s1z)
    s2x, s2y, s2z = _rotate_y(-theta_0, s2x, s2y, s2z)

    # Rotation 3: put L at azimuth phi_jl about J.
    lx, ly, lz = _rotate_z(phi_jl - np.pi, lx, ly, lz)
    s1x, s1y, s1z = _rotate_z(phi_jl - np.pi, s1x, s1y, s1z)
    s2x, s2y, s2z = _rotate_z(phi_jl - np.pi, s2x, s2y, s2z)

    # The line of sight, in the y-z plane at theta_jn from J.
    nx, ny, nz = np.zeros_like(theta_jn), np.sin(theta_jn), np.cos(theta_jn)
    inclination = np.arccos(nx * lx + ny * ly + nz * lz)

    # Rotations 4-5: bring L onto the z axis.
    theta_lj = np.arccos(lz)
    phi_l = np.arctan2(ly, lx)
    s1x, s1y, s1z = _rotate_z(-phi_l, s1x, s1y, s1z)
    s2x, s2y, s2z = _rotate_z(-phi_l, s2x, s2y, s2z)
    nx, ny, nz = _rotate_z(-phi_l, nx, ny, nz)
    s1x, s1y, s1z = _rotate_y(-theta_lj, s1x, s1y, s1z)
    s2x, s2y, s2z = _rotate_y(-theta_lj, s2x, s2y, s2z)
    nx, ny, nz = _rotate_y(-theta_lj, nx, ny, nz)

    # Rotation 6: bring the line of sight into the y-z plane, less the reference phase.
    phi_n = np.arctan2(ny, nx)
    s1x, s1y, s1z = _rotate_z(np.pi / 2.0 - phi_n - phi_ref, s1x, s1y, s1z)
    s2x, s2y, s2z = _rotate_z(np.pi / 2.0 - phi_n - phi_ref, s2x, s2y, s2z)

    return inclination, s1x * chi_1, s1y * chi_1, s1z * chi_1, s2x * chi_2, s2y * chi_2, s2z * chi_2


__all__ = ["spin_angles_to_components"]
