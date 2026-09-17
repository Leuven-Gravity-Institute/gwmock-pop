"""Shared builders for the catalogue draw tests.

The tests exercise the producer against small local catalogues rather than the
published ones, so the suite stays offline and the expected counts are exact.
The files carry the same columns the published product carries.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from gwmock_pop.catalogue.manifest import CatalogueFile
from gwmock_pop.catalogue.source import ResolvedCatalogue

_DEFAULT_COLUMNS = ("m1_source", "m2_source", "chi1z", "chi2z", "dL", "ra", "dec", "psi", "iota", "Phicoal", "z")


def write_catalogue(  # noqa: PLR0913  # the columns the published product carries
    path: Path,
    *,
    mass_1: np.ndarray,
    mass_2: np.ndarray,
    redshift: np.ndarray,
    spin_1z: np.ndarray | None = None,
    spin_2z: np.ndarray | None = None,
) -> None:
    """Write a minimal HDF5 catalogue with the published product's columns.

    Args:
        path: Destination file.
        mass_1: Source-frame primary masses.
        mass_2: Source-frame secondary masses.
        redshift: Source redshifts.
        spin_1z: Aligned primary spins. Defaults to zeros.
        spin_2z: Aligned secondary spins. Defaults to zeros.
    """
    n_rows = mass_1.shape[0]
    rng = np.random.default_rng(0)
    zeros = np.zeros(n_rows, dtype=float)
    with h5py.File(path, "w") as handle:
        handle["m1_source"] = np.asarray(mass_1, dtype=float)
        handle["m2_source"] = np.asarray(mass_2, dtype=float)
        handle["chi1z"] = np.asarray(spin_1z if spin_1z is not None else zeros, dtype=float)
        handle["chi2z"] = np.asarray(spin_2z if spin_2z is not None else zeros, dtype=float)
        handle["dL"] = np.linspace(0.05, 150.0, n_rows)
        handle["ra"] = rng.uniform(0.0, 2.0 * np.pi, n_rows)
        handle["dec"] = rng.uniform(-np.pi / 2.0, np.pi / 2.0, n_rows)
        handle["psi"] = rng.uniform(0.0, np.pi, n_rows)
        handle["iota"] = rng.uniform(0.0, np.pi, n_rows)
        handle["Phicoal"] = rng.uniform(0.0, 2.0 * np.pi, n_rows)
        handle["z"] = np.asarray(redshift, dtype=float)


def resolved_catalogue(
    path: Path,
    *,
    name: str,
    source_class: str,
    n_rows: int | None = None,
) -> ResolvedCatalogue:
    """Describe a written catalogue as an already-resolved fetch result.

    Args:
        path: Path of the catalogue file.
        name: Short catalogue name.
        source_class: Source class the file holds.
        n_rows: Row count for the pinned manifest. Defaults to the file's size.

    Returns:
        The resolved catalogue, with the metadata a verified cache hit carries.
    """
    with h5py.File(path, "r") as handle:
        file_rows = int(np.asarray(handle["m1_source"]).shape[0])
    catalogue = CatalogueFile(
        name=name,
        source_class=source_class,
        filename=path.name,
        sha256="0" * 64,
        size_bytes=path.stat().st_size,
        n_rows=file_rows if n_rows is None else n_rows,
    )
    return ResolvedCatalogue(catalogue=catalogue, path=path, metadata={"verified": True, "cache_hit": True})


def make_pair(  # noqa: PLR0913  # one row per mass/redshift knob
    tmp_path: Path,
    *,
    bns_mass_1: tuple[float, ...] = (1.2, 1.4, 1.6),
    bns_mass_2: tuple[float, ...] = (1.1, 1.3, 1.5),
    bns_redshift: tuple[float, ...] = (0.05, 0.1, 0.2),
    bbh_mass_1: tuple[float, ...] = (30.0, 40.0, 50.0),
    bbh_mass_2: tuple[float, ...] = (20.0, 30.0, 40.0),
    bbh_redshift: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> tuple[ResolvedCatalogue, ResolvedCatalogue]:
    """Write and resolve a matched BNS/BBH catalogue pair.

    Args:
        tmp_path: Directory to write the files in.
        bns_mass_1: Source-frame primary masses of the neutron-star rows.
        bns_mass_2: Source-frame secondary masses of the neutron-star rows.
        bns_redshift: Redshifts of the neutron-star rows.
        bbh_mass_1: Source-frame primary masses of the black-hole rows.
        bbh_mass_2: Source-frame secondary masses of the black-hole rows.
        bbh_redshift: Redshifts of the black-hole rows.

    Returns:
        The resolved BNS and BBH catalogues, in that order.
    """
    bns_path = tmp_path / "bns.h5"
    bbh_path = tmp_path / "bbh.h5"
    write_catalogue(
        bns_path,
        mass_1=np.asarray(bns_mass_1),
        mass_2=np.asarray(bns_mass_2),
        redshift=np.asarray(bns_redshift),
    )
    write_catalogue(
        bbh_path,
        mass_1=np.asarray(bbh_mass_1),
        mass_2=np.asarray(bbh_mass_2),
        redshift=np.asarray(bbh_redshift),
    )
    return (
        resolved_catalogue(bns_path, name="bns", source_class="BNS"),
        resolved_catalogue(bbh_path, name="bbh", source_class="BBH"),
    )


def make_random_pair(
    tmp_path: Path,
    *,
    n_rows: int = 4000,
    seed: int = 7,
) -> tuple[ResolvedCatalogue, ResolvedCatalogue]:
    """Write and resolve a random BNS/BBH pair shaped like the published product.

    The black-hole rows reach the intermediate-mass threshold so all three
    classes are present, and the neutron-star redshifts are wide enough that a
    meaningful fraction is pushed into the band by the ``(1 + z)`` boost.

    Args:
        tmp_path: Directory to write the files in.
        n_rows: Rows per catalogue.
        seed: Generator seed for the synthetic values.

    Returns:
        The resolved BNS and BBH catalogues, in that order.
    """
    rng = np.random.default_rng(seed)
    bns_mass_1 = rng.uniform(1.1, 2.5, n_rows)
    bns_mass_2 = bns_mass_1 * rng.uniform(0.8, 1.0, n_rows)
    bns_redshift = rng.uniform(0.01, 13.7, n_rows)
    bbh_mass_1 = rng.uniform(5.7, 300.0, n_rows)
    bbh_mass_2 = bbh_mass_1 * rng.uniform(0.3, 1.0, n_rows)
    bbh_redshift = rng.uniform(0.005, 14.0, n_rows)
    bns_path = tmp_path / "bns.h5"
    bbh_path = tmp_path / "bbh.h5"
    write_catalogue(bns_path, mass_1=bns_mass_1, mass_2=bns_mass_2, redshift=bns_redshift)
    write_catalogue(bbh_path, mass_1=bbh_mass_1, mass_2=bbh_mass_2, redshift=bbh_redshift)
    return (
        resolved_catalogue(bns_path, name="bns", source_class="BNS"),
        resolved_catalogue(bbh_path, name="bbh", source_class="BBH"),
    )


__all__ = [
    "make_pair",
    "make_random_pair",
    "resolved_catalogue",
    "write_catalogue",
]
