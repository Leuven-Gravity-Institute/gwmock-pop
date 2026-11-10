"""Catalogues published in other layouts, read into the draw's columns, and fetched from an archive.

The files here are small stand-ins written in the layouts of Zenodo record 21682194 (Dupletsa et al.
2026): a pandas "fixed" frame with gwfish names, and flat datasets with gwfast names. They are written
through gwmock_data.hdf5, so the tests need neither pandas nor h5py.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gwmock_data import hdf5

from gwmock_pop.catalogue import source
from gwmock_pop.catalogue.layouts import read_layout, read_pandas_fixed_frame
from gwmock_pop.catalogue.manifest import (
    DUPLETSA2026_BBH,
    DUPLETSA2026_BNS,
    DUPLETSA2026_CATALOGUES,
    DUPLETSA2026_PBH,
    CatalogueArchive,
    CatalogueFile,
)
from gwmock_pop.catalogue.population import draw_catalogue_population
from gwmock_pop.catalogue.spins import spin_angles_to_components
from gwmock_pop.exceptions import PopulationFetchError, PopulationValidationError

pytestmark = pytest.mark.unit

N = 40


def _write_pandas_frame(path: Path, columns: dict[str, np.ndarray]) -> None:
    """A pandas 'fixed'-format frame, as pandas writes one: one float block with named columns."""
    names = list(columns)
    frame = hdf5.Group(
        members={
            "axis0": hdf5.Dataset(np.asarray(names, dtype="S19")),
            "axis1": hdf5.Dataset(np.arange(N, dtype=np.int64)),
            "block0_items": hdf5.Dataset(np.asarray(names, dtype="S19")),
            "block0_values": hdf5.Dataset(np.column_stack([columns[name] for name in names])),
        },
        attrs={"pandas_type": "frame", "pandas_version": "0.15.2", "encoding": "UTF-8"},
    )
    hdf5.write(hdf5.Group(members={"df": frame}), path)


def _write_flat(path: Path, columns: dict[str, np.ndarray]) -> None:
    hdf5.write(hdf5.Group(members={name: hdf5.Dataset(values) for name, values in columns.items()}), path)


def _common(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "redshift": rng.uniform(0.1, 3.0, N),
        "ra": rng.uniform(0, 2 * np.pi, N),
        "dec": rng.uniform(-1.5, 1.5, N),
        "psi": rng.uniform(0, np.pi, N),
        "phase": rng.uniform(0, 2 * np.pi, N),
        "theta_jn": rng.uniform(0, np.pi, N),
    }


def _bns_frame(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        **_common(rng),
        "chirp_mass": rng.uniform(1.5, 4.0, N),
        "mass_ratio": rng.uniform(0.7, 1.0, N),
        "luminosity_distance": rng.uniform(100.0, 1e5, N),
        "a_1": rng.uniform(-0.05, 0.05, N),
        "a_2": rng.uniform(-0.05, 0.05, N),
        "lambda_1": rng.uniform(10, 2000, N),
        "lambda_2": rng.uniform(10, 2000, N),
    }


def test_a_pandas_frame_is_read_without_pandas(tmp_path):
    """A pandas frame is read without pandas."""
    columns = _bns_frame(np.random.default_rng(1))
    _write_pandas_frame(tmp_path / "f.hdf5", columns)

    read = read_pandas_fixed_frame(tmp_path / "f.hdf5")

    assert set(read) == set(columns)
    for name, values in columns.items():
        np.testing.assert_array_equal(read[name], values)


def test_aligned_gwfish_rows_become_the_draw_columns(tmp_path):
    """Masses from the detector-frame chirp mass and m2/m1, Mpc to Gpc, signed spins, theta_jn as iota."""
    columns = _bns_frame(np.random.default_rng(2))
    _write_pandas_frame(tmp_path / "bns.hdf5", columns)

    out = read_layout(tmp_path / "bns.hdf5", "gwfish_pandas")

    detector_mass_1 = out["m1_source"] * (1 + columns["redshift"])
    detector_mass_2 = out["m2_source"] * (1 + columns["redshift"])
    chirp = (detector_mass_1 * detector_mass_2) ** 0.6 / (detector_mass_1 + detector_mass_2) ** 0.2
    np.testing.assert_allclose(chirp, columns["chirp_mass"], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(detector_mass_2 / detector_mass_1, columns["mass_ratio"], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(out["dL"], columns["luminosity_distance"] / 1000.0, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(out["chi1z"], columns["a_1"])
    np.testing.assert_array_equal(out["iota"], columns["theta_jn"])
    np.testing.assert_array_equal(out["Lambda2"], columns["lambda_2"])
    assert "chi1x" not in out


def test_precessing_gwfish_rows_are_converted_at_the_reference_frequency(tmp_path):
    """Precessing gwfish rows are converted at the reference frequency."""
    rng = np.random.default_rng(3)
    columns = {
        **_bns_frame(rng),
        "a_1": rng.uniform(0, 0.9, N),
        "a_2": rng.uniform(0, 0.9, N),
        "tilt_1": rng.uniform(0, np.pi, N),
        "tilt_2": rng.uniform(0, np.pi, N),
        "phi_12": rng.uniform(0, 2 * np.pi, N),
        "phi_jl": rng.uniform(0, 2 * np.pi, N),
    }
    for name in ("lambda_1", "lambda_2"):
        columns.pop(name)
    _write_pandas_frame(tmp_path / "pop3.hdf5", columns)

    out = read_layout(tmp_path / "pop3.hdf5", "gwfish_pandas", reference_frequency=2.0)

    detector = (out["m1_source"] * (1 + columns["redshift"]), out["m2_source"] * (1 + columns["redshift"]))
    expected = spin_angles_to_components(
        columns["theta_jn"], columns["phi_jl"], columns["tilt_1"], columns["tilt_2"], columns["phi_12"],
        columns["a_1"], columns["a_2"], detector[0], detector[1], 2.0, columns["phase"],
    )  # fmt: skip
    # The test rebuilds the detector-frame masses through (1 + z), so they can differ in the last bit.
    for name, values in zip(("iota", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z"), expected, strict=True):
        np.testing.assert_allclose(out[name], values, rtol=1e-12, atol=1e-15)
    with pytest.raises(PopulationValidationError, match="no reference frequency"):
        read_layout(tmp_path / "pop3.hdf5", "gwfish_pandas")


def test_gwfast_rows_keep_stored_components_and_convert_angles_and_polar_sky(tmp_path):
    """Components are read as stored; spin angles are converted; theta/phi become dec/ra."""
    rng = np.random.default_rng(4)
    base = {
        "m1_src": rng.uniform(5, 80, N),
        "m2_src": rng.uniform(5, 80, N),
        "z": rng.uniform(0.1, 5, N),
        "dL": rng.uniform(0.1, 50, N),
        "psi": rng.uniform(0, np.pi, N),
        "Phicoal": rng.uniform(0, 2 * np.pi, N),
    }
    stored = {
        **base,
        "ra": rng.uniform(0, 2 * np.pi, N),
        "dec": rng.uniform(-1.5, 1.5, N),
        "iota": rng.uniform(0, np.pi, N),
        **{name: rng.uniform(-0.5, 0.5, N) for name in ("chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z")},
    }
    _write_flat(tmp_path / "bbh.h5", stored)
    out = read_layout(tmp_path / "bbh.h5", "gwfast")
    for name in ("iota", "chi1x", "chi2z", "ra", "dec", "dL"):
        np.testing.assert_array_equal(out[name], stored[name])

    angles = {
        **base,
        "theta": rng.uniform(0, np.pi, N),
        "phi": rng.uniform(0, 2 * np.pi, N),
        "thetaJN": rng.uniform(0, np.pi, N),
        "chi1": rng.uniform(0, 0.9, N),
        "chi2": rng.uniform(0, 0.9, N),
        "tilt1": rng.uniform(0, np.pi, N),
        "tilt2": rng.uniform(0, np.pi, N),
        "phi12": rng.uniform(0, 2 * np.pi, N),
        "phiJL": rng.uniform(0, 2 * np.pi, N),
    }
    _write_flat(tmp_path / "pbh.h5", angles)
    out = read_layout(tmp_path / "pbh.h5", "gwfast", reference_frequency=2.0)
    np.testing.assert_array_equal(out["ra"], angles["phi"])
    np.testing.assert_allclose(out["dec"], np.pi / 2 - angles["theta"], rtol=0.0, atol=1e-15)
    one_plus_z = 1 + angles["z"]
    expected = spin_angles_to_components(
        angles["thetaJN"], angles["phiJL"], angles["tilt1"], angles["tilt2"], angles["phi12"], angles["chi1"],
        angles["chi2"], angles["m1_src"] * one_plus_z, angles["m2_src"] * one_plus_z, 2.0, angles["Phicoal"],
    )  # fmt: skip
    np.testing.assert_allclose(out["iota"], expected[0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(out["chi2y"], expected[5], rtol=1e-12, atol=1e-15)


def test_a_missing_column_is_named(tmp_path):
    """A missing column is named."""
    columns = _bns_frame(np.random.default_rng(5))
    columns.pop("luminosity_distance")
    _write_pandas_frame(tmp_path / "bns.hdf5", columns)
    with pytest.raises(PopulationValidationError, match="luminosity_distance"):
        read_layout(tmp_path / "bns.hdf5", "gwfish_pandas")


def test_an_unknown_layout_is_refused():
    """An unknown layout is refused."""
    with pytest.raises(ValueError, match="unknown layout"):
        replace(DUPLETSA2026_BNS, layout="other")


def test_the_pinned_entries_point_at_the_record_s_archive():
    """The pinned entries point at the record's archive."""
    assert {catalogue.archive for catalogue in DUPLETSA2026_CATALOGUES} == {DUPLETSA2026_BNS.archive}
    assert all(catalogue.url == DUPLETSA2026_BNS.archive.url for catalogue in DUPLETSA2026_CATALOGUES)
    assert all(catalogue.record_url == "https://zenodo.org/records/21682194" for catalogue in DUPLETSA2026_CATALOGUES)
    assert DUPLETSA2026_PBH.source_class == "PBH"
    assert DUPLETSA2026_BBH.spin_reference_frequency == 2.0


# --- the archive fetch and a draw end to end ---------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def archived(tmp_path, monkeypatch):
    """A local zip standing in for the record's archive, with a BNS and a BBH member pinned."""
    rng = np.random.default_rng(6)
    _write_pandas_frame(tmp_path / "BNSs.hdf5", _bns_frame(rng))
    bbh = {
        "m1_src": rng.uniform(5, 80, N),
        "m2_src": rng.uniform(5, 80, N),
        "z": rng.uniform(0.1, 5, N),
        "dL": rng.uniform(0.1, 50, N),
        "psi": rng.uniform(0, np.pi, N),
        "Phicoal": rng.uniform(0, 2 * np.pi, N),
        "ra": rng.uniform(0, 2 * np.pi, N),
        "dec": rng.uniform(-1.5, 1.5, N),
        "iota": rng.uniform(0, np.pi, N),
        **{name: rng.uniform(-0.5, 0.5, N) for name in ("chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z")},
    }
    _write_flat(tmp_path / "BBHs.h5", bbh)
    archive_path = tmp_path / "injections.zip"
    with zipfile.ZipFile(archive_path, "w") as bundle:
        for name in ("BNSs.hdf5", "BBHs.h5"):
            bundle.write(tmp_path / name, f"injections/{name}")
    archive = CatalogueArchive("injections.zip", "https://example.org/injections.zip", "0" * 64, 1)
    entries = []
    for entry, name in ((DUPLETSA2026_BNS, "BNSs.hdf5"), (DUPLETSA2026_BBH, "BBHs.h5")):
        data = (tmp_path / name).read_bytes()
        entries.append(replace(entry, sha256=_sha256(data), size_bytes=len(data), n_rows=N, archive=archive))
    cache = tmp_path / "cache"
    cache.mkdir()
    cached_archive = cache / "archive.zip"
    cached_archive.write_bytes(archive_path.read_bytes())

    def _pinned(url, **kwargs):
        assert url == archive.url
        # As the real fetch behaves: a refresh while offline cannot be met.
        if kwargs.get("refresh") and kwargs.get("offline"):
            raise PopulationFetchError("not cached and downloads are disabled")
        return source.FetchResult(path=cached_archive, metadata={"cache_hit": True, "verified": True})

    monkeypatch.setattr(source, "resolve_digest_pinned_path", _pinned)
    return entries, cache


def test_a_member_is_extracted_verified_and_reused(archived):
    """A member is extracted verified and reused."""
    entries, cache = archived
    first = source.fetch_catalogue(entries[0])
    again = source.fetch_catalogue(entries[0])

    assert first.path == again.path
    assert first.path.parent == cache
    assert first.metadata["member"] == "injections/BNSs.hdf5"
    assert again.metadata["cache_hit"] is True
    assert _sha256(first.path.read_bytes()) == entries[0].sha256


def test_a_member_that_differs_from_its_pin_is_refused(archived):
    """A member that differs from its pin is refused."""
    entries, _ = archived
    tampered = replace(entries[0], sha256="f" * 64)
    with pytest.raises(PopulationFetchError, match="refusing it"):
        source.fetch_catalogue(tampered)
    with pytest.raises(PopulationFetchError, match="no member"):
        source.fetch_catalogue(replace(entries[0], filename="injections/missing.hdf5"))


def test_a_draw_reads_the_archived_catalogues(archived):
    """Counts follow the one-year rate; each class's rows come from its own file, in its units."""
    entries, _ = archived
    draw = draw_catalogue_population(seed=7, span_seconds=0.25 * 365.25 * 86400, catalogues=entries)

    assert set(draw.drawn_per_class) == {"BNS", "BBH"}
    assert draw.expected_per_class == pytest.approx({"BNS": N / 4, "BBH": N / 4}, rel=1e-12, abs=0.0)
    bns = draw.source_class == "BNS"
    assert np.all(draw.parameters["lambda_1"][bns] > 0)
    assert np.all(draw.parameters["spin_1x"][bns] == 0)
    assert np.all(draw.parameters["lambda_1"][~bns] == 0)
    assert np.all(draw.parameters["distance"][bns] >= 100.0)  # Mpc, from luminosity_distance in Mpc


def test_an_undeclared_optional_column_is_refused(archived):
    """A BBH file with Cartesian spins whose entry does not declare them would emit zero spins."""
    entries, _ = archived
    undeclared = replace(entries[1], extra_columns=())
    with pytest.raises(PopulationValidationError, match="does not declare"):
        draw_catalogue_population(seed=7, span_seconds=0.25 * 365.25 * 86400, catalogues=[undeclared])


def test_catalogue_file_still_builds_without_an_archive():
    """The design-comparison entries, with no archive, keep their document-server URLs."""
    plain = CatalogueFile(name="x", source_class="BNS", filename="a.h5", sha256="0" * 64, size_bytes=1, n_rows=1)
    assert plain.url.startswith("https://apps.et-gw.eu/tds/?call_file=")
    assert plain.layout == "coba"


def test_an_offline_refresh_re_extracts_from_the_cached_archive(archived):
    """Refreshing a member offline needs no download: the verified archive is already cached."""
    entries, _ = archived
    first = source.fetch_catalogue(entries[0])
    first.path.write_bytes(b"damaged")

    again = source.fetch_catalogue(entries[0], refresh=True, offline=True)

    assert again.path == first.path
    assert again.metadata["cache_hit"] is False
    assert _sha256(again.path.read_bytes()) == entries[0].sha256


@pytest.mark.parametrize("rows", [0, "ragged"])
def test_a_layout_file_with_ragged_or_no_rows_is_refused(tmp_path, rows):
    """The draw refuses such a file whatever its layout, as it does for the design-comparison files."""
    from gwmock_pop.catalogue.population import _catalogue_columns
    from gwmock_pop.catalogue.source import ResolvedCatalogue

    rng = np.random.default_rng(8)
    count = 0 if rows == 0 else N
    columns = {
        name: rng.uniform(1.0, 2.0, count)
        for name in ("m1_src", "m2_src", "z", "dL", "psi", "Phicoal", "ra", "dec", "iota")
    }
    columns.update(
        {name: rng.uniform(-0.5, 0.5, count) for name in ("chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z")}
    )
    if rows == "ragged":
        columns["psi"] = columns["psi"][:-1]
    _write_flat(tmp_path / "bbh.h5", columns)
    resolved = ResolvedCatalogue(catalogue=DUPLETSA2026_BBH, path=tmp_path / "bbh.h5", metadata={})

    with pytest.raises(PopulationValidationError, match="empty" if rows == 0 else "mismatched column lengths"):
        _catalogue_columns(resolved)
