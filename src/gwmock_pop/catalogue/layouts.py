"""Read each published catalogue layout into the columns a catalogue draw uses.

A draw (:mod:`gwmock_pop.catalogue.population`) reads one set of columns, named as the Einstein
Telescope design-comparison catalogues name them: source-frame masses ``m1_source`` and
``m2_source``, aligned spins ``chi1z`` and ``chi2z``, luminosity distance ``dL`` in Gpc, sky position
``ra`` and ``dec``, polarisation ``psi``, inclination ``iota``, phase ``Phicoal`` and redshift ``z``,
plus the optional columns of :data:`gwmock_pop.catalogue.manifest.OPTIONAL_COLUMNS`. Other
published catalogues name, scale and parametrise the same quantities differently; a layout reader
turns one of them into these columns, so the draw itself is the same for every catalogue.

Layouts:

``"coba"``
    Already in the draw's columns (the design-comparison HDF5 files).

``"gwfish_pandas"``
    A pandas "fixed"-format HDF5 frame in gwfish's names, as in Zenodo record 21682194
    (Dupletsa et al. 2026): detector-frame ``chirp_mass`` and ``mass_ratio`` (m2/m1), ``redshift``,
    ``luminosity_distance`` in Mpc, ``ra``, ``dec``, ``psi``, ``phase``, ``theta_jn``, and either
    aligned signed spins ``a_1``, ``a_2`` with ``lambda_1``, ``lambda_2`` (binary neutron stars) or
    spin magnitudes ``a_1``, ``a_2`` with ``tilt_1``, ``tilt_2``, ``phi_12``, ``phi_jl`` (precessing).
    Read through :mod:`gwmock_data.hdf5`; neither pandas nor h5py is needed.

``"gwfast"``
    Flat HDF5 datasets in gwfast's names, as in the same record: ``m1_src``, ``m2_src``, ``z``,
    ``dL`` in Gpc, ``psi``, ``Phicoal``, a sky position as ``ra``/``dec`` or as ``theta``/``phi``
    (polar angle and azimuth), and either Cartesian spins ``chi1x`` ... ``chi2z`` with ``iota`` or
    spin angles ``chi1``, ``chi2``, ``tilt1``, ``tilt2``, ``phi12``, ``phiJL`` with ``thetaJN``.

Spin angles are converted to Cartesian components and the inclination by
:func:`gwmock_pop.catalogue.spins.spin_angles_to_components` -- LALSimulation's conversion --
at the catalogue's ``spin_reference_frequency`` with detector-frame masses and the coalescence
phase as reference phase. That is the convention recovered from Zenodo 21682194's ``BBHs.h5``, which
carries both forms: it reproduces the stored components to a median of 5e-11 over all 129,491
rows (the same with lalsuite 7.7 and 8.0). The rows that differ by more, up to 6.5e-3 rad, are the
near-equal-mass ones (m2/m1 > 0.987). A file carrying the components is read as stored, since
those are the values its authors injected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gwmock_data import hdf5

from gwmock_pop.catalogue.spins import spin_angles_to_components
from gwmock_pop.exceptions import PopulationValidationError

#: Layout names a catalogue entry may declare.
LAYOUTS = ("coba", "gwfish_pandas", "gwfast")

#: Megaparsecs per gigaparsec: the draw's ``dL`` is in Gpc.
_MPC_PER_GPC = 1000.0


def _require(columns: dict[str, np.ndarray], names: tuple[str, ...], path: Path) -> None:
    missing = [name for name in names if name not in columns]
    if missing:
        raise PopulationValidationError(f"Catalogue {path} is missing required columns: {', '.join(missing)}.")


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_pandas_fixed_frame(path: Path, key: str = "df") -> dict[str, np.ndarray]:
    """Read a pandas "fixed"-format HDF5 frame into columns, without pandas.

    pandas stores such a frame as a group holding, per dtype block, the column names
    (``block<k>_items``) and a rows-by-columns array (``block<k>_values``).

    Args:
        path: The HDF5 file.
        key: The frame's group, ``"df"`` unless the writer chose another.

    Returns:
        One 1-D array per frame column.

    Raises:
        PopulationValidationError: If the file holds no such frame.
    """
    group = hdf5.read(path, members=[key]).members.get(key)
    if not isinstance(group, hdf5.Group) or _text(group.attrs.get("pandas_type", "")) != "frame":
        raise PopulationValidationError(f"Catalogue {path} holds no pandas frame at {key!r}.")
    columns: dict[str, np.ndarray] = {}
    block = 0
    while f"block{block}_items" in group.members:
        names = [_text(name) for name in np.asarray(group.members[f"block{block}_items"].data)]
        values = np.asarray(group.members[f"block{block}_values"].data)
        for index, name in enumerate(names):
            columns[name] = values[:, index]
        block += 1
    if not columns:
        raise PopulationValidationError(f"Catalogue {path} holds a pandas frame with no column blocks.")
    return columns


def _read_flat(path: Path) -> dict[str, np.ndarray]:
    root = hdf5.read(path)
    return {name: np.asarray(item.data) for name, item in root.members.items() if isinstance(item, hdf5.Dataset)}


def _precessing(  # noqa: PLR0913 - the conversion's inputs, by name
    out: dict[str, np.ndarray],
    *,
    theta_jn: np.ndarray,
    phi_jl: np.ndarray,
    tilts: tuple[np.ndarray, np.ndarray],
    phi_12: np.ndarray,
    magnitudes: tuple[np.ndarray, np.ndarray],
    detector_masses: tuple[np.ndarray, np.ndarray],
    reference_frequency: float,
) -> None:
    inclination, s1x, s1y, s1z, s2x, s2y, s2z = spin_angles_to_components(
        theta_jn,
        phi_jl,
        tilts[0],
        tilts[1],
        phi_12,
        magnitudes[0],
        magnitudes[1],
        detector_masses[0],
        detector_masses[1],
        reference_frequency,
        out["Phicoal"],
    )
    out.update(iota=inclination, chi1x=s1x, chi1y=s1y, chi1z=s1z, chi2x=s2x, chi2y=s2y, chi2z=s2z)


def _gwfish_pandas(path: Path, reference_frequency: float | None) -> dict[str, np.ndarray]:
    raw = read_pandas_fixed_frame(path)
    _require(raw, ("chirp_mass", "mass_ratio", "redshift", "luminosity_distance", "ra", "dec", "psi", "phase"), path)
    redshift = raw["redshift"]
    chirp, ratio = raw["chirp_mass"], raw["mass_ratio"]
    detector_mass_1 = chirp * (1.0 + ratio) ** 0.2 / ratio**0.6
    detector_mass_2 = ratio * detector_mass_1
    out = {
        "m1_source": detector_mass_1 / (1.0 + redshift),
        "m2_source": detector_mass_2 / (1.0 + redshift),
        "dL": raw["luminosity_distance"] / _MPC_PER_GPC,
        "ra": raw["ra"],
        "dec": raw["dec"],
        "psi": raw["psi"],
        "Phicoal": raw["phase"],
        "z": redshift,
    }
    if "tilt_1" in raw:
        _require(raw, ("a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn"), path)
        if reference_frequency is None:
            raise PopulationValidationError(f"Catalogue {path} gives spin angles but no reference frequency.")
        _precessing(
            out,
            theta_jn=raw["theta_jn"],
            phi_jl=raw["phi_jl"],
            tilts=(raw["tilt_1"], raw["tilt_2"]),
            phi_12=raw["phi_12"],
            magnitudes=(raw["a_1"], raw["a_2"]),
            detector_masses=(detector_mass_1, detector_mass_2),
            reference_frequency=reference_frequency,
        )
    else:
        # Aligned spins, supplied signed as a_1 and a_2 (the record's README); with no in-plane
        # spin, J is along L and theta_jn is the inclination.
        _require(raw, ("a_1", "a_2", "theta_jn"), path)
        out.update(chi1z=raw["a_1"], chi2z=raw["a_2"], iota=raw["theta_jn"])
    if "lambda_1" in raw:
        out.update(Lambda1=raw["lambda_1"], Lambda2=raw["lambda_2"])
    return out


def _gwfast(path: Path, reference_frequency: float | None) -> dict[str, np.ndarray]:
    raw = _read_flat(path)
    _require(raw, ("m1_src", "m2_src", "z", "dL", "psi", "Phicoal"), path)
    out = {
        "m1_source": raw["m1_src"],
        "m2_source": raw["m2_src"],
        "dL": raw["dL"],
        "psi": raw["psi"],
        "Phicoal": raw["Phicoal"],
        "z": raw["z"],
    }
    if "ra" in raw:
        _require(raw, ("dec",), path)
        out.update(ra=raw["ra"], dec=raw["dec"])
    else:
        _require(raw, ("theta", "phi"), path)
        out.update(ra=raw["phi"], dec=np.pi / 2.0 - raw["theta"])
    if "chi1x" in raw:
        _require(raw, ("chi1y", "chi1z", "chi2x", "chi2y", "chi2z", "iota"), path)
        out.update({name: raw[name] for name in ("chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z", "iota")})
    else:
        _require(raw, ("chi1", "chi2", "tilt1", "tilt2", "phi12", "phiJL", "thetaJN"), path)
        if reference_frequency is None:
            raise PopulationValidationError(f"Catalogue {path} gives spin angles but no reference frequency.")
        one_plus_z = 1.0 + raw["z"]
        _precessing(
            out,
            theta_jn=raw["thetaJN"],
            phi_jl=raw["phiJL"],
            tilts=(raw["tilt1"], raw["tilt2"]),
            phi_12=raw["phi12"],
            magnitudes=(raw["chi1"], raw["chi2"]),
            detector_masses=(raw["m1_src"] * one_plus_z, raw["m2_src"] * one_plus_z),
            reference_frequency=reference_frequency,
        )
    return out


def read_layout(path: Path, layout: str, reference_frequency: float | None = None) -> dict[str, np.ndarray]:
    """Read a catalogue file in another layout into the draw's columns.

    Args:
        path: The verified local file.
        layout: ``"gwfish_pandas"`` or ``"gwfast"`` (``"coba"`` files are read by the draw directly).
        reference_frequency: Frequency in Hz at which spin angles are defined, for a file that
            gives spins as angles.

    Returns:
        The draw's columns, plus whichever optional columns the file provides.

    Raises:
        PopulationValidationError: If the layout is unknown or the file lacks a column it needs.
    """
    if layout == "gwfish_pandas":
        return _gwfish_pandas(Path(path), reference_frequency)
    if layout == "gwfast":
        return _gwfast(Path(path), reference_frequency)
    raise PopulationValidationError(f"Unknown catalogue layout {layout!r}; known: {', '.join(LAYOUTS)}.")


__all__ = ["LAYOUTS", "read_layout", "read_pandas_fixed_frame"]
