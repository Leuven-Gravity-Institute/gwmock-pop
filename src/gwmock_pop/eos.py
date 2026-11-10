"""Neutron-star equations of state: the tidal deformability of a neutron star of a given mass.

A binary with neutron stars needs each star's dimensionless tidal deformability
``Lambda = (2/3) k2 (R c^2 / (G m))^5``, which an equation of state (EOS) fixes for each mass. Drawing it
from one named EOS, instead of from a distribution, gives every neutron star in a population the same
microphysics, as an EOS study needs.

The EOS and the stellar structure come from LALSuite's ``lalsimulation`` (its ``SimNeutronStar``
routines), which names some 150 equations of state -- among them ``SLY``, ``APR4_EPP``,
``PCP_BSK24_BSK24`` and ``XMLSLZ_DDLZ1_BSK24`` -- and solves the structure and tidal equations for each.
Every value returned is lalsimulation's own, at the given mass; nothing is interpolated here. Install
it with the ``eos`` extra: ``pip install "gwmock-pop[eos]"``.

The mass is the *source-frame* mass: the EOS describes the star itself, whose mass a detector sees
scaled by ``1 + z``.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

_LAL_IMPORT_HINT = 'Tidal deformabilities from an equation of state need lalsimulation: pip install "gwmock-pop[eos]".'


def _lalsimulation() -> tuple[Any, Any]:
    """Import ``lal`` and ``lalsimulation``, or say how to install them."""
    try:
        import lal  # noqa: PLC0415
        import lalsimulation  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(_LAL_IMPORT_HINT) from error
    return lal, lalsimulation


@functools.lru_cache(maxsize=32)
def _family(eos: str) -> tuple[Any, float, float]:
    """The sequence of neutron stars of ``eos``, with its smallest and largest mass, solar masses."""
    lal, lalsimulation = _lalsimulation()
    try:
        equation_of_state = lalsimulation.SimNeutronStarEOSByName(eos)
    except RuntimeError as error:
        raise ValueError(
            f"lalsimulation has no neutron-star equation of state called {eos!r}; "
            "its names are listed in lalsimulation's LALSimNeutronStarEOS."
        ) from error
    family = lalsimulation.CreateSimNeutronStarFamily(equation_of_state, 0)
    minimum = lalsimulation.SimNeutronStarFamMinimumMass(family) / lal.MSUN_SI
    maximum = lalsimulation.SimNeutronStarMaximumMass(family) / lal.MSUN_SI
    return family, float(minimum), float(maximum)


def neutron_star_mass_range(eos: str) -> tuple[float, float]:
    """The source-frame masses ``eos`` has neutron stars for, ``[minimum, maximum)``, solar masses.

    The maximum mass itself is excluded: it is the top of the stable branch, where lalsimulation's
    structure solution does not return a radius.

    Args:
        eos: lalsimulation's name for the equation of state, e.g. ``"PCP_BSK24_BSK24"``.

    Returns:
        The smallest and the largest mass, solar masses.

    Raises:
        ImportError: If lalsimulation is not installed.
        ValueError: If lalsimulation has no equation of state called ``eos``.
    """
    _, minimum, maximum = _family(eos)
    return minimum, maximum


def tidal_deformability(source_frame_mass: Any, eos: str) -> np.ndarray:
    """The dimensionless tidal deformability of a neutron star of ``eos``, at each source-frame mass.

    Args:
        source_frame_mass: Neutron-star masses in the source frame, solar masses.
        eos: lalsimulation's name for the equation of state, e.g. ``"PCP_BSK24_BSK24"``.

    Returns:
        ``Lambda`` at each mass, the shape of ``source_frame_mass``.

    Raises:
        ImportError: If lalsimulation is not installed.
        ValueError: If lalsimulation has no equation of state called ``eos``, or a mass is not finite or
            lies outside the masses ``eos`` has neutron stars for (see :func:`neutron_star_mass_range`):
            above them there is no neutron star, so no deformability to give.
    """
    lal, lalsimulation = _lalsimulation()
    family, minimum, maximum = _family(eos)
    masses = np.asarray(source_frame_mass, dtype=float)
    outside = ~np.isfinite(masses) | (masses < minimum) | (masses >= maximum)
    if np.any(outside):
        bad = masses[outside]
        raise ValueError(
            f"{eos} has neutron stars of source-frame mass in [{minimum}, {maximum}) solar masses; "
            f"{bad.size} mass(es) are outside it, e.g. {bad.flat[0]!r}."
        )
    geometric_mass = lal.G_SI / lal.C_SI**2
    values = np.empty(masses.shape)
    for index, mass in np.ndenumerate(masses):
        mass_si = mass * lal.MSUN_SI
        radius = lalsimulation.SimNeutronStarRadius(mass_si, family)
        k2 = lalsimulation.SimNeutronStarLoveNumberK2(mass_si, family)
        values[index] = 2.0 / 3.0 * k2 * (radius / (mass_si * geometric_mass)) ** 5
    return values
