"""Tests for the pinned catalogue manifest."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from gwmock_pop.catalogue import draw_catalogue_population, manifest
from gwmock_pop.catalogue.population import SECONDS_PER_YEAR, read_catalogue_columns
from gwmock_pop.catalogue.source import ResolvedCatalogue, fetch_catalogue
from gwmock_pop.synthesis.catalogue import catalogue_columns, write_catalogue


def test_catalogue_files_are_pinned_with_usable_identity() -> None:
    """Every pinned catalogue carries a SHA-256, a size and a row count."""
    for catalogue in manifest.COBA_CATALOGUES:
        assert len(catalogue.sha256) == 64
        assert all(character in "0123456789abcdef" for character in catalogue.sha256)
        assert catalogue.size_bytes > 0
        assert catalogue.n_rows > 0
        assert catalogue.source_class in {"BNS", "BBH"}


def test_catalogue_urls_name_the_record_and_the_file() -> None:
    """A catalogue's URL carries the document record and the file name."""
    for catalogue in manifest.COBA_CATALOGUES:
        url = catalogue.url
        assert url.startswith(manifest.TDS_BASE_URL)
        assert "call_file=" in url
        assert catalogue.filename in url
        assert catalogue.record_url.endswith(f"?r={manifest.TDS_CATALOGUE_RECORD}")


def test_draw_order_is_binary_neutron_stars_first() -> None:
    """The draw order is part of the seeded draw's identity and is pinned."""
    assert tuple(catalogue.name for catalogue in manifest.COBA_CATALOGUES) == ("bns", "bbh")


def test_catalogue_url_lookup_by_name() -> None:
    """A named lookup returns the pinned catalogue's URL."""
    assert manifest.catalogue_url("bbh") == manifest.COBA_BBH.url


def test_catalogue_url_lookup_rejects_unknown_name() -> None:
    """An unknown catalogue name is refused rather than silently defaulted."""
    with pytest.raises(KeyError):
        manifest.catalogue_url("does-not-exist")


def test_both_citations_are_recorded_with_distinct_roles() -> None:
    """The catalogue's source paper and the configuration reference are distinct."""
    assert manifest.CATALOGUE_SOURCE_CITATION != manifest.SCIENCE_REFERENCE_CITATION
    assert "2303.15923" in manifest.CATALOGUE_SOURCE_CITATION
    assert "2503.12263" in manifest.SCIENCE_REFERENCE_CITATION
    assert set(manifest.CITATION_ROLES) == {"catalogue_source_paper", "science_reference"}
    for role, description in manifest.CITATION_ROLES.items():
        assert role
        assert description


def test_the_published_catalogues_declare_their_extra_columns() -> None:
    """BNS carries tidal deformabilities; BBH carries in-plane spins defined at 2 Hz."""
    assert manifest.COBA_BNS.extra_columns == ("Lambda1", "Lambda2")
    assert manifest.COBA_BNS.spin_reference_frequency is None
    assert manifest.COBA_BBH.extra_columns == ("chi1x", "chi1y", "chi2x", "chi2y")
    assert manifest.COBA_BBH.spin_reference_frequency == 2.0


def test_the_nsbh_entry_names_its_record_and_file() -> None:
    """The NSBH file is fetched from its Zenodo record, read as is, with its tidal deformabilities."""
    entry = manifest.IORIO2023_NSBH
    assert entry.source_class == "NSBH"
    assert entry.layout == "coba"
    assert entry.archive is None
    assert entry.url == "https://zenodo.org/records/24127546/files/nsbh_iorio2023_F_alpha1.h5?download=1"
    assert entry.record_url == manifest.IORIO2023_NSBH_RECORD == "https://zenodo.org/records/24127546"
    assert entry.citation == "https://doi.org/10.5281/zenodo.24127546"
    assert entry.extra_columns == ("Lambda1", "Lambda2")
    assert (entry.sha256, entry.size_bytes, entry.n_rows) == (
        "91057e685e5298a9f3b5d5611d41f414e9e10fc25d50f777b285b0afd23c50ac",
        4064328,
        39027,
    )


def _resolved(path, entry):
    return ResolvedCatalogue(
        catalogue=dataclasses.replace(
            entry, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size
        ),
        path=path,
        metadata={},
    )


def _write(path, rng, mass_1, mass_2, lambda_2):
    """Write a catalogue as the deposit was written; the first component is a black hole (Lambda1 = 0)."""
    n = mass_1.size
    columns = catalogue_columns(
        rng.uniform(0.1, 3.0, n),
        mass_1,
        mass_2,
        rng.uniform(-0.05, 0.05, n),
        rng.uniform(-0.05, 0.05, n),
        rng=rng,
        lambda_1=np.zeros(n),
        lambda_2=lambda_2,
    )
    write_catalogue(path, columns, {"test": True})


def test_the_nsbh_entry_is_drawn_after_a_bbh_catalogue_and_keeps_its_class(tmp_path) -> None:
    """Drawn after a BBH catalogue, as with the Dupletsa populations, NSBH rows follow it and stay NSBH.

    Half of each catalogue's rows exceed the IMBH mass threshold: the BBH ones are relabelled IMBH,
    the NSBH ones, whose black holes are as heavy, are not.
    """
    rng = np.random.default_rng(3)
    n = 400
    heavy = np.where(np.arange(n) % 2 == 0, 150.0, 10.0)
    bbh_path, nsbh_path = tmp_path / "bbh.h5", tmp_path / manifest.IORIO2023_NSBH.filename
    _write(bbh_path, rng, heavy, np.full(n, 8.0), np.zeros(n))
    _write(nsbh_path, rng, heavy, rng.uniform(1.1, 1.6, n), rng.uniform(200.0, 1500.0, n))
    bbh_entry = manifest.CatalogueFile(
        name="bbh_fixture",
        source_class="BBH",
        filename="bbh.h5",
        sha256="0" * 64,
        size_bytes=1,
        n_rows=n,
        extra_columns=("Lambda1", "Lambda2"),
    )
    nsbh_entry = dataclasses.replace(manifest.IORIO2023_NSBH, n_rows=n)
    resolved = (_resolved(bbh_path, bbh_entry), _resolved(nsbh_path, nsbh_entry))
    draw = draw_catalogue_population(
        seed=1,
        span_seconds=0.5 * SECONDS_PER_YEAR,
        catalogues=tuple(item.catalogue for item in resolved),
        resolved=resolved,
    )
    source = draw.source_class.tolist()
    n_bbh, n_nsbh = source.count("BBH"), source.count("NSBH")
    assert source == ["BBH"] * n_bbh + ["NSBH"] * n_nsbh  # catalogue order, in blocks
    for count in (n_bbh, n_nsbh):
        assert abs(count - n / 2) < 5 * np.sqrt(n / 2)
    nsbh_rows = draw.source_class == "NSBH"
    assert set(draw.population_class[nsbh_rows].tolist()) == {"NSBH"}
    assert np.any(draw.parameters["detector_frame_mass_1"][nsbh_rows] > 100.0)  # heavy enough to be relabelled
    assert set(draw.population_class[~nsbh_rows].tolist()) == {"BBH", "IMBH"}
    assert np.all(draw.parameters["lambda_1"][nsbh_rows] == 0.0)
    assert np.all(
        (draw.parameters["lambda_2"][nsbh_rows] >= 200.0) & (draw.parameters["lambda_2"][nsbh_rows] <= 1500.0)
    )


def test_the_manifest_exports_the_nsbh_entry() -> None:
    """``from gwmock_pop.catalogue.manifest import *`` carries the entry and its record, as for the other products."""
    assert {"IORIO2023_NSBH", "IORIO2023_NSBH_RECORD"} <= set(manifest.__all__)


@pytest.mark.integration
def test_the_nsbh_record_serves_the_pinned_file(tmp_path) -> None:
    """The published file downloads, verifies against its pins, and holds the pinned rows (network)."""
    resolved = fetch_catalogue(manifest.IORIO2023_NSBH, cache_dir=tmp_path)
    columns = read_catalogue_columns(resolved.path, ("m1_source", "m2_source", "z", "Lambda1", "Lambda2"))
    assert columns["z"].size == manifest.IORIO2023_NSBH.n_rows
    assert columns["m1_source"].min() > 5.0
    assert columns["m2_source"].max() < 2.0
