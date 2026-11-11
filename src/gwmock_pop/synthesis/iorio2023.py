"""The merging compact binaries Iorio et al. (2023) published, read as a population-synthesis run.

Iorio et al. 2023 (MNRAS 524, 426; "Compact object mergers: exploring uncertainties from stellar
and binary evolution with SEVN") deposited their SEVN runs on Zenodo, record 7794546 (CC BY 4.0).
The record is one 3.8 GB archive, ``iorio+22_Zenodo_repository_V2.zip``. Each model's folder,
``data_from_simulations/simulation_<model>/``, holds a table of each merging class:

- ``BHBHm.csv``, black hole + black hole;
- ``BHNSm.csv``, black hole + neutron star;
- ``NSNSm.csv``, neutron star + neutron star.

Each table holds every metallicity (15, from 1e-4 to 0.03) and every common-envelope efficiency
``alpha`` (0.5, 1, 3 and 5) of the model. ``total_evolved_mass.txt`` gives the initial mass evolved
in each run: 22,154,859.026 solar masses for the fiducial model ``F``, the same at every
metallicity and ``alpha``.

:data:`IORIO2023_TABLES` pins the fiducial model's tables this package reads by SHA-256. Extract
them from the archive yourself and pass the path: the archive is too large to fetch for one table.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.synthesis.convolution import MergingBinaries

#: SHA-256 of the fiducial model's tables, by file name.
IORIO2023_TABLES: dict[str, str] = {
    "BHNSm.csv": "3ee9edaa86fced8c72833e408c2651665fc53f2d24667553c2f4f70bd6d2c4fa",
    "NSNSm.csv": "16da4c8858a9361ce88ad573da3314db0f5aa40825ce41b4a2822aaecdb65aa2",
}

#: Initial mass, in solar masses, evolved in each run of the fiducial model (``total_evolved_mass.txt``).
IORIO2023_EVOLVED_MASS = 22154859.025999993

#: Years per megayear: the tables give times in Myr.
_YR_PER_MYR = 1e6

#: Table columns read, and the names they are carried under.
_PROPERTIES = {
    "Mass_0": "mass_0",
    "Mass_1": "mass_1",
    "RemnantType_0": "remnant_type_0",
    "RemnantType_1": "remnant_type_1",
}


def read_iorio2023(path: str | os.PathLike[str], alpha: float = 1.0) -> MergingBinaries:
    """Read one of Iorio et al.'s fiducial merging-binary tables at one common-envelope efficiency.

    The file must match its pin in :data:`IORIO2023_TABLES`. A binary's delay is its
    formation-to-compact-binary time ``BWorldtime`` plus its gravitational-wave inspiral time
    ``GWtime``.

    Args:
        path: ``BHNSm.csv`` or ``NSNSm.csv`` from ``simulation_F``.
        alpha: Common-envelope efficiency: 0.5, 1, 3 or 5.

    Returns:
        The merging binaries, carrying the compact objects' masses (solar masses) and SEVN
        remnant types (``mass_0``, ``mass_1``, ``remnant_type_0``, ``remnant_type_1``; 4 and 5 are
        neutron stars, 6 a black hole).

    Raises:
        PopulationValidationError: If the file is not a pinned table, or holds no binary at ``alpha``.
    """
    table = Path(path)
    expected = IORIO2023_TABLES.get(table.name)
    if expected is None:
        raise PopulationValidationError(f"{table.name} is not one of {sorted(IORIO2023_TABLES)}.")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    if digest != expected:
        raise PopulationValidationError(f"{table} has SHA-256 {digest}, not the pinned {expected}.")
    columns = ("alpha", "Z", "BWorldtime", "GWtime", *_PROPERTIES)
    data = np.genfromtxt(table, delimiter=",", names=True, usecols=columns, dtype=None, encoding="utf-8")
    selected = data[data["alpha"] == alpha]
    if selected.size == 0:
        raise PopulationValidationError(f"{table} holds no binary at alpha = {alpha}.")
    return MergingBinaries(
        delay_time=(selected["BWorldtime"] + selected["GWtime"]) * _YR_PER_MYR,
        metallicity=np.asarray(selected["Z"], dtype=float),
        evolved_mass=IORIO2023_EVOLVED_MASS,
        properties={name: np.asarray(selected[column]) for column, name in _PROPERTIES.items()},
    )


#: SEVN remnant types of a neutron star (electron-capture and core-collapse) and a black hole.
_NEUTRON_STAR_TYPES = (4, 5)
_BLACK_HOLE_TYPE = 6


def black_hole_neutron_star_masses(mergers: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Split each black hole + neutron star merger's masses into the black hole's and the neutron star's.

    Args:
        mergers: Mergers drawn from :func:`read_iorio2023` of ``BHNSm.csv``.

    Returns:
        The black holes' masses and the neutron stars' masses, solar masses.

    Raises:
        PopulationValidationError: If a merger is not one black hole and one neutron star.
    """
    type_0, type_1 = mergers["remnant_type_0"], mergers["remnant_type_1"]
    first_is_black_hole = (type_0 == _BLACK_HOLE_TYPE) & np.isin(type_1, _NEUTRON_STAR_TYPES)
    second_is_black_hole = (type_1 == _BLACK_HOLE_TYPE) & np.isin(type_0, _NEUTRON_STAR_TYPES)
    if not np.all(first_is_black_hole | second_is_black_hole):
        raise PopulationValidationError("Every merger must be one black hole and one neutron star.")
    black_hole = np.where(first_is_black_hole, mergers["mass_0"], mergers["mass_1"])
    neutron_star = np.where(first_is_black_hole, mergers["mass_1"], mergers["mass_0"])
    return black_hole, neutron_star
