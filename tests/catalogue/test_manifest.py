"""Tests for the pinned catalogue manifest."""

from __future__ import annotations

import pytest

from gwmock_pop.catalogue import manifest


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
