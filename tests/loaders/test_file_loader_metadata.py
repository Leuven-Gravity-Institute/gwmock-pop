"""Unit tests for the metadata ``FilePopulationLoader`` records about its input."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gwmock_pop import FilePopulationLoader, PopulationValidationError
from gwmock_pop.loaders import file_loader


def _write_csv_catalogue(path: Path, n_rows: int) -> None:
    """Write a small header-based CBC catalogue with ``n_rows`` rows."""
    rows = [f"{30.0 + row},{20.0 + row},0.1" for row in range(n_rows)]
    path.write_text("m1,m2,z\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_metadata_records_the_sha256_of_the_file_read(tmp_path: Path) -> None:
    """The loader names the file it read by content, not only by path."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)

    loader = FilePopulationLoader("bbh", path)

    assert loader.metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_metadata_sha256_follows_the_content_at_one_path(tmp_path: Path) -> None:
    """Two catalogues written in turn to the same path are told apart by their digests."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    first = FilePopulationLoader("bbh", path).metadata["sha256"]
    _write_csv_catalogue(path, 2)
    second = FilePopulationLoader("bbh", path).metadata["sha256"]

    assert first != second
    assert second == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_file_changed_while_it_is_read_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A digest taken before parsing must not be recorded for bytes that changed during it."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    read = file_loader.read_population_catalogue

    def read_then_replace(*args, **kwargs):
        catalogue = read(*args, **kwargs)
        _write_csv_catalogue(path, 2)
        return catalogue

    monkeypatch.setattr(file_loader, "read_population_catalogue", read_then_replace)

    with pytest.raises(PopulationValidationError, match="changed while it was read"):
        FilePopulationLoader("bbh", path)


@pytest.mark.parametrize("make", [lambda path: None, lambda path: path.mkdir()], ids=["missing", "directory"])
def test_an_unsupported_suffix_is_reported_before_the_file_is_opened(make, tmp_path: Path) -> None:
    """The unsupported-format error does not depend on whether the path can be opened."""
    path = tmp_path / "catalogue.fits"
    make(path)

    with pytest.raises(PopulationValidationError, match=r"Supported suffixes: \.csv, \.h5, \.hdf5"):
        FilePopulationLoader("bbh", path)


def test_provenance_is_refused_once_the_file_has_changed_since_loading(tmp_path: Path) -> None:
    """The record read later must come from the bytes the recorded digest describes."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    loader = FilePopulationLoader("bbh", path)
    assert loader.provenance is None
    _write_csv_catalogue(path, 2)

    with pytest.raises(PopulationValidationError, match="changed since it was loaded"):
        _ = loader.provenance
