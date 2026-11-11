r"""The merging compact binaries Iorio et al. (2023) published, read as a population-synthesis run.

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

Normalisation
-------------

:func:`read_iorio2023` normalises the run as Iorio et al.'s own published merger rates do (their
Figs. 21 and 22):

- **IMF fraction 0.25643.** This is their correction for the mass cuts (their footnote 13 and
  ``InitialConditions/estimate_mass_correction.py``): the share of primary mass, in a Kroupa
  population of 0.08 to 150 solar masses with Sana et al. (2012) mass ratios, held by binaries with a
  primary of at least 5 solar masses and a secondary of at least 2.2. :func:`imf_fraction` computes
  it exactly. They quote 0.255 +- 0.001, from a Monte Carlo of the same definition; the exact value,
  not 0.255, reproduces their published rates.
- **Binary fraction 0.4**, the default of :class:`~gwmock_pop.synthesis.convolution.CosmicHistory`.

Neither value is stated for the rates; both are recovered from the cosmoRate output behind their
figures (``gitlab.com/iogiul/iorio22_plot``, tag ``v3``, ``Merger_rate/cosmo_rate_bavera``):

- cosmoRate names its output after the binary fraction, and theirs is ``MRD_spread_15Z_40_...``.
- With both values, this module reproduces their binary neutron star and black hole + neutron star
  curves below redshift 8 to a median of 0.3 %, and within 1.1 % in every bin.
- The 0.3 % is this package's cosmology, which departs from cosmoRate's on purpose; see
  :mod:`gwmock_pop.synthesis.convolution`. With cosmoRate's, the median is 0.02 %.

Their text uses a binary fraction of 0.5 elsewhere, for the merger efficiencies of their Sec. 4.3, and
explains neither value. A different normalisation is a different population: pass a different
``binary_fraction``, or build :class:`~gwmock_pop.synthesis.convolution.MergingBinaries` with a
different ``imf_fraction``.

The run's sampling does not hold exactly the binaries the IMF fraction counts. It gives every primary
of at least 5 solar masses a secondary of at least 2.2, redrawing the mass ratio, where the
correction counts only the binaries of the full population that pass both cuts. Weighting each binary
by the full population's chance of so heavy a secondary, and normalising the run's mass accordingly,
raises the binary neutron stars' rate by 5.7 % and the black hole + neutron star mergers' by 11.3 %.
This module follows the authors' normalisation rather than correcting it: the correction rests on an
assumption, that binaries with lighter secondaries contribute no merger, that nothing external
checks.
"""

from __future__ import annotations

import functools
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

#: The sampling of the deposit's ``InitialConditions``: Kroupa slopes and break, the full population's
#: primary range, the q slope and floor, its lightest secondary, and the run's mass cuts (solar masses).
_KROUPA_SLOPES = (-1.3, -2.3)
_KROUPA_BREAK = 0.5
_FULL_PRIMARY_RANGE = (0.08, 150.0)
_Q_SLOPE = -0.1
_Q_MIN = 0.1
_FULL_MIN_SECONDARY = 0.08
_RUN_MIN_PRIMARY = 5.0
_RUN_MIN_SECONDARY = 2.2
#: Points on the primary-mass grid the IMF fraction is integrated over.
_IMF_GRID = 400_001

#: Years per megayear: the tables give times in Myr.
_YR_PER_MYR = 1e6

#: Table columns read, and the names they are carried under.
_PROPERTIES = {
    "Mass_0": "mass_0",
    "Mass_1": "mass_1",
    "RemnantType_0": "remnant_type_0",
    "RemnantType_1": "remnant_type_1",
}


@functools.cache
def imf_fraction() -> float:
    """Iorio et al.'s correction for their mass cuts, computed exactly from its definition.

    Returns:
        The share of primary mass, in the full population, held by binaries passing both mass cuts.
    """
    primary = np.geomspace(*_FULL_PRIMARY_RANGE, _IMF_GRID)
    low, high = _KROUPA_SLOPES
    density = np.where(primary < _KROUPA_BREAK, (primary / _KROUPA_BREAK) ** low, (primary / _KROUPA_BREAK) ** high)
    # P(q >= the run's floor) for q ~ q^-0.1 on the full population's [floor, 1]
    exponent = _Q_SLOPE + 1.0
    full_floor = np.minimum(np.maximum(_FULL_MIN_SECONDARY / primary, _Q_MIN), 1.0)
    run_floor = np.maximum(np.maximum(_RUN_MIN_SECONDARY / primary, _Q_MIN), full_floor)
    span = 1.0 - full_floor**exponent
    passes = np.where(span > 0.0, (1.0 - run_floor**exponent) / np.where(span > 0.0, span, 1.0), 0.0)
    passes = np.where(primary >= _RUN_MIN_PRIMARY, passes, 0.0)
    return float(np.trapezoid(density * primary * passes, primary) / np.trapezoid(density * primary, primary))


def read_iorio2023(path: str | os.PathLike[str], alpha: float = 1.0) -> MergingBinaries:
    """Read one of Iorio et al.'s fiducial merging-binary tables at one common-envelope efficiency.

    The file must match its pin in :data:`IORIO2023_TABLES`. A binary's delay is its
    formation-to-compact-binary time ``BWorldtime`` plus its gravitational-wave inspiral time
    ``GWtime``. The run's IMF fraction is :func:`imf_fraction`.

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
        imf_fraction=imf_fraction(),
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
