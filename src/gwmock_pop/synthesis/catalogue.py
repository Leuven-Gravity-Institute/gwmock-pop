"""Write drawn mergers as a catalogue a catalogue draw reads.

The file holds the draw's columns, as the Einstein Telescope design-comparison catalogues name
them (layout ``"coba"`` of :mod:`gwmock_pop.catalogue.layouts`). That is source-frame masses
``m1_source`` and ``m2_source``, aligned spins ``chi1z`` and ``chi2z``, luminosity distance ``dL``
in Gpc, ``ra``, ``dec``, ``psi``, inclination ``iota``, coalescence phase ``Phicoal`` and redshift
``z``, with ``Lambda1`` and ``Lambda2`` when given.

The orientation, sky position and phase are isotropic: ``ra`` and ``Phicoal`` uniform in
[0, 2 pi), ``sin(dec)`` and ``cos(iota)`` uniform in [-1, 1], and ``psi`` uniform in [0, pi).
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import numpy as np
from gwmock_data import hdf5

from gwmock_pop.cosmology.flat_lambda_cdm import build_distance_lookup, interpolate_within_lookup
from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.provenance.io import write_hdf5_provenance
from gwmock_pop.synthesis.convolution import CosmicHistory

#: Megaparsecs per gigaparsec: the catalogue's ``dL`` is in Gpc.
_MPC_PER_GPC = 1000.0


def catalogue_columns(  # noqa: PLR0913 - the catalogue's per-merger quantities, by name
    redshift: np.ndarray,
    mass_1: np.ndarray,
    mass_2: np.ndarray,
    spin_1z: np.ndarray,
    spin_2z: np.ndarray,
    *,
    rng: np.random.Generator,
    lambda_1: np.ndarray | None = None,
    lambda_2: np.ndarray | None = None,
    history: CosmicHistory | None = None,
) -> dict[str, np.ndarray]:
    """Columns of a catalogue of the given mergers, with isotropic orientations and sky positions.

    Args:
        redshift: Merger redshifts.
        mass_1: Source-frame mass of the first component, solar masses.
        mass_2: Source-frame mass of the second component, solar masses.
        spin_1z: Dimensionless spin of the first component along the orbital angular momentum.
        spin_2z: Same for the second component.
        rng: Random-number generator for the orientations, sky positions and phases.
        lambda_1: Tidal deformability of the first component, or ``None``.
        lambda_2: Tidal deformability of the second component, or ``None``.
        history: Cosmology for the luminosity distance; Planck 2018 when omitted.

    Returns:
        The catalogue's columns, by name.

    Raises:
        PopulationValidationError: If the inputs differ in length, or only one deformability is given.
    """
    history = history or CosmicHistory()
    given = {"mass_1": mass_1, "mass_2": mass_2, "spin_1z": spin_1z, "spin_2z": spin_2z}
    if (lambda_1 is None) != (lambda_2 is None):
        raise PopulationValidationError("Give both tidal deformabilities or neither.")
    if lambda_1 is not None and lambda_2 is not None:
        given |= {"lambda_1": lambda_1, "lambda_2": lambda_2}
    n_rows = redshift.shape[0]
    misaligned = {name: values.shape[0] for name, values in given.items() if values.shape[0] != n_rows}
    if misaligned:
        raise PopulationValidationError(f"Columns {misaligned} do not match the {n_rows} redshifts.")

    grid, _, luminosity_distance = build_distance_lookup(
        hubble_constant=history.hubble_constant, omega_m=history.omega_m, max_redshift=history.max_redshift
    )
    distance_mpc = interpolate_within_lookup(jnp.asarray(redshift), grid, luminosity_distance, "redshift")
    columns = {
        "m1_source": np.asarray(mass_1, dtype=float),
        "m2_source": np.asarray(mass_2, dtype=float),
        "chi1z": np.asarray(spin_1z, dtype=float),
        "chi2z": np.asarray(spin_2z, dtype=float),
        "dL": np.asarray(distance_mpc) / _MPC_PER_GPC,
        "ra": rng.uniform(0.0, 2.0 * np.pi, n_rows),
        "dec": np.arcsin(rng.uniform(-1.0, 1.0, n_rows)),
        "psi": rng.uniform(0.0, np.pi, n_rows),
        "iota": np.arccos(rng.uniform(-1.0, 1.0, n_rows)),
        "Phicoal": rng.uniform(0.0, 2.0 * np.pi, n_rows),
        "z": np.asarray(redshift, dtype=float),
    }
    if lambda_1 is not None and lambda_2 is not None:
        columns["Lambda1"] = np.asarray(lambda_1, dtype=float)
        columns["Lambda2"] = np.asarray(lambda_2, dtype=float)
    return columns


def write_catalogue(path: str | os.PathLike[str], columns: dict[str, np.ndarray], provenance: dict[str, Any]) -> None:
    """Write catalogue columns to an HDF5 file, one dataset each, with a provenance record.

    Args:
        path: Output file.
        columns: :func:`catalogue_columns`, or further columns beside them.
        provenance: Record of how the catalogue was made, embedded as gwmock-pop's provenance.
    """
    root = hdf5.Group(members={name: hdf5.Dataset(values) for name, values in columns.items()})
    write_hdf5_provenance(root, provenance)
    hdf5.write(root, path)
