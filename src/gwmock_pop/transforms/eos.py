"""Transform from a neutron star's mass to its tidal deformability, through a named equation of state."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from gwmock_pop.conversion.cbc import compute_source_frame_mass_from_detector_frame_mass
from gwmock_pop.eos import neutron_star_mass_range, tidal_deformability


def eos_tidal_deformability(detector_frame_mass: Array, redshift: Array, eos: str) -> Array:
    """The tidal deformability of each neutron star, from one equation of state.

    The equation of state describes the star itself, so it is evaluated at the source-frame mass,
    ``detector_frame_mass / (1 + redshift)``. Taking the detector-frame mass and the redshift, rather
    than a mass the caller has already converted, keeps a detector-frame mass from being passed where
    the source-frame one is meant. Each value is lalsimulation's own; see :mod:`gwmock_pop.eos`.

    Args:
        detector_frame_mass: Neutron-star masses in the detector frame, solar masses.
        redshift: The binaries' redshifts.
        eos: lalsimulation's name for the equation of state, e.g. ``"PCP_BSK24_BSK24"``.

    Returns:
        The dimensionless tidal deformability of each star. Where the input is traced (under
        ``jax.jit``, ``jax.vmap``, ...), a mass outside the equation of state's neutron stars cannot
        raise, and is returned as NaN instead.

    Raises:
        ImportError: If lalsimulation is not installed (the ``eos`` extra).
        ValueError: If lalsimulation has no equation of state called ``eos``, or, for concrete input, a
            source-frame mass lies outside the masses ``eos`` has neutron stars for.
    """
    source_frame_mass = compute_source_frame_mass_from_detector_frame_mass(
        jnp.asarray(detector_frame_mass), jnp.asarray(redshift)
    )
    minimum, maximum = neutron_star_mass_range(eos)  # refuses an unknown name even when traced
    if not isinstance(source_frame_mass, jax.core.Tracer):
        return jnp.asarray(tidal_deformability(np.asarray(source_frame_mass), eos), dtype=source_frame_mass.dtype)

    dtype = source_frame_mass.dtype

    def _on_host(masses: np.ndarray) -> np.ndarray:
        masses = np.asarray(masses, dtype=float)
        values = np.full(masses.shape, np.nan)
        inside = np.isfinite(masses) & (masses >= minimum) & (masses < maximum)
        values[inside] = tidal_deformability(masses[inside], eos)
        return values.astype(dtype)

    # _on_host works element by element on any shape, so a vmapped call can pass the whole batch at once.
    return jax.pure_callback(
        _on_host, jax.ShapeDtypeStruct(source_frame_mass.shape, dtype), source_frame_mass, vmap_method="expand_dims"
    )
