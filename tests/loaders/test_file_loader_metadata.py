"""Unit tests for the metadata ``FilePopulationLoader`` records about its input."""

from __future__ import annotations

import hashlib
from pathlib import Path

from gwmock_pop import FilePopulationLoader


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
