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

The archive's ``InitialConditions`` folder defines the binaries' sampling:

- primaries are drawn from a Kroupa IMF (slopes -1.3 below 0.5 solar masses and -2.3 above) from 5
  to 150 solar masses;
- mass ratios are drawn as :math:`q^{-0.1}` (Sana et al. 2012) on
  :math:`[\max(2.2\,M_\odot/m_1, 0.1), 1]`.

That folder also defines the full population those binaries are drawn from: primaries from 0.08 to
150 solar masses and :math:`q` on :math:`[\max(0.08\,M_\odot/m_1, 0.1), 1]`.

A simulated binary with primary :math:`m_1` stands for the binaries of that primary in the full
population whose secondary is at least 2.2 solar masses. Lighter secondaries end as white dwarfs or
less, and form no compact merger. So the binary carries weight :math:`w(m_1)`, the full population's
probability of such a secondary: below 1 for primaries under 22 solar masses, and 1 above. The run
then stands for :math:`N/F_5` full-population binaries, where :math:`N` is the number of simulated
binaries and :math:`F_5` the full population's share of primaries of at least 5 solar masses.
:func:`imf_fraction` gives the evolved mass's share of their mass,
:math:`F_5 \langle m_1 + m_2 \rangle_\mathrm{run} / \langle m_1 + m_2 \rangle_\mathrm{full}`, which
is 0.284.

Iorio et al. state 0.255 instead: the mass fraction of binaries that pass both mass cuts in the full
population. That treats the run as if it held only those binaries. It does not: the run gives every
primary above 5 solar masses a secondary above 2.2, so its binaries with primaries under 22 solar
masses are over-represented, and the rest are counted against too much mass.

- Against 0.255 and unit weights, the merging binary neutron stars' rate is 5.7 % higher, and the
  black hole + neutron star mergers' 11.3 % higher. Their progenitors' primaries have medians of 14
  and 27 solar masses.
- cosmoRate's 0.285 is the same construction for runs whose secondaries are not truncated, and the
  value agrees.

With these weights and a binary fraction of 0.5 (Iorio et al.'s), ``NSNSm.csv`` at
:math:`\alpha_\mathrm{CE} = 1` gives 460,500 mergers per year, with 140 Gpc^-3 yr^-1 at
:math:`z = 0.0075`. Zenodo record 21682194, built from the same binaries, holds 351,205, 0.76 times as
many. The record's redshift distribution matches; its normalisation is not stated.
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

#: The initial-condition sampling of the deposit's ``InitialConditions``: Kroupa slopes and break,
#: primary range of the full population and of the run, the q slope and its floor, and the full
#: population's and the run's lightest secondary (solar masses).
_KROUPA_SLOPES = (-1.3, -2.3)
_KROUPA_BREAK = 0.5
_FULL_PRIMARY_RANGE = (0.08, 150.0)
_RUN_MIN_PRIMARY = 5.0
_Q_SLOPE = -0.1
_Q_MIN = 0.1
_FULL_MIN_SECONDARY = 0.08
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


def _q_floor(primary: np.ndarray, lightest_secondary: float) -> np.ndarray:
    return np.maximum(lightest_secondary / primary, _Q_MIN)


def _q_tail(floor: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """P(q >= threshold) for q ~ q^-0.1 on [floor, 1]; 1 where the floor is already at or above it."""
    exponent = _Q_SLOPE + 1.0
    floor = np.minimum(floor, 1.0)
    span = 1.0 - floor**exponent
    tail = 1.0 - np.maximum(threshold, floor) ** exponent
    return np.where(span > 0.0, tail / np.where(span > 0.0, span, 1.0), 1.0)


def _q_mean(floor: np.ndarray) -> np.ndarray:
    """Mean of q ~ q^-0.1 on [floor, 1]; 1 where the floor is 1."""
    floor = np.minimum(floor, 1.0)
    span = (1.0 - floor ** (_Q_SLOPE + 1.0)) / (_Q_SLOPE + 1.0)
    first = (1.0 - floor ** (_Q_SLOPE + 2.0)) / (_Q_SLOPE + 2.0)
    return np.where(span > 0.0, first / np.where(span > 0.0, span, 1.0), 1.0)


def secondary_weight(primary: np.ndarray) -> np.ndarray:
    """The full population's probability that a binary of this primary mass has a secondary of 2.2 or more.

    Args:
        primary: Primary ZAMS mass, solar masses.

    Returns:
        The weight of a simulated binary with that primary.
    """
    primary = np.asarray(primary, dtype=float)
    return _q_tail(_q_floor(primary, _FULL_MIN_SECONDARY), _q_floor(primary, _RUN_MIN_SECONDARY))


@functools.cache
def imf_fraction() -> float:
    """The evolved mass's share of the mass of the full population the run stands for.

    Returns:
        F_5 <m1 + m2>_run / <m1 + m2>_full, integrated over the Kroupa IMF.
    """
    primary = np.geomspace(*_FULL_PRIMARY_RANGE, _IMF_GRID)
    low, high = _KROUPA_SLOPES
    density = np.where(primary < _KROUPA_BREAK, (primary / _KROUPA_BREAK) ** low, (primary / _KROUPA_BREAK) ** high)
    in_run = primary >= _RUN_MIN_PRIMARY
    full_system = primary * (1.0 + _q_mean(_q_floor(primary, _FULL_MIN_SECONDARY)))
    run_system = primary * (1.0 + _q_mean(_q_floor(primary, _RUN_MIN_SECONDARY)))
    full_count = np.trapezoid(density, primary)
    run_count = np.trapezoid(density[in_run], primary[in_run])
    mean_full = np.trapezoid(density * full_system, primary) / full_count
    mean_run = np.trapezoid(density[in_run] * run_system[in_run], primary[in_run]) / run_count
    return float(run_count / full_count * mean_run / mean_full)


def read_iorio2023(path: str | os.PathLike[str], alpha: float = 1.0) -> MergingBinaries:
    """Read one of Iorio et al.'s fiducial merging-binary tables at one common-envelope efficiency.

    The file must match its pin in :data:`IORIO2023_TABLES`. A binary's delay is its
    formation-to-compact-binary time ``BWorldtime`` plus its gravitational-wave inspiral time
    ``GWtime``. Its weight is :func:`secondary_weight` of its primary, the heavier of its two stars'
    ZAMS masses (``Mzams_0``, ``Mzams_1``), and the run's IMF fraction is :func:`imf_fraction`.

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
    columns = ("alpha", "Z", "BWorldtime", "GWtime", "Mzams_0", "Mzams_1", *_PROPERTIES)
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
        weight=secondary_weight(np.maximum(selected["Mzams_0"], selected["Mzams_1"])),
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
