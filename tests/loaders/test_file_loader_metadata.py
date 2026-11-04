"""Unit tests for the metadata ``FilePopulationLoader`` records about its input."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gwmock_pop import FilePopulationLoader, PopulationValidationError
from gwmock_pop.loaders import file_loader
from gwmock_pop.provenance import provenance_sidecar_path, write_provenance_sidecar


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


def test_provenance_is_the_record_present_at_load(tmp_path: Path) -> None:
    """A later change to the catalogue or its sidecar does not reach the loaded record."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    write_provenance_sidecar(path, {"origin": "first"})
    loader = FilePopulationLoader("bbh", path)
    _write_csv_catalogue(path, 2)
    write_provenance_sidecar(path, {"origin": "second"})

    assert loader.provenance == {"origin": "first"}
    path.unlink()
    assert loader.provenance == {"origin": "first"}


def test_provenance_returns_a_copy(tmp_path: Path) -> None:
    """A caller editing the record it was given does not edit the loader's."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    write_provenance_sidecar(path, {"origin": {"name": "first"}})
    loader = FilePopulationLoader("bbh", path)

    loader.provenance["origin"]["name"] = "edited"

    assert loader.provenance == {"origin": {"name": "first"}}


def test_an_undecodable_record_is_raised_only_when_asked_for(tmp_path: Path) -> None:
    """The catalogue still loads; the record's error surfaces on access, as when it was read lazily."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    provenance_sidecar_path(path).write_text("not json", encoding="utf-8")
    loader = FilePopulationLoader("bbh", path)

    with pytest.raises(ValueError, match=r"provenance") as caught:
        _ = loader.provenance
    assert caught.value.__cause__ is not None


def test_repeated_access_to_an_undecodable_record_does_not_grow_its_traceback(tmp_path: Path) -> None:
    """Each access raises its own exception, so earlier accesses' frames are not kept."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    provenance_sidecar_path(path).write_text("not json", encoding="utf-8")
    loader = FilePopulationLoader("bbh", path)

    def depth() -> int:
        with pytest.raises(ValueError, match=r"provenance") as caught:
            _ = loader.provenance
        frames, traceback = 0, caught.value.__traceback__
        while traceback is not None:
            frames, traceback = frames + 1, traceback.tb_next
        return frames

    assert depth() == depth() == depth()


def test_a_stored_provenance_error_keeps_no_frame_of_the_constructor(tmp_path: Path) -> None:
    """The constructor's locals hold the parsed catalogue, so no kept traceback may reach its frame."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    provenance_sidecar_path(path).write_text("not json", encoding="utf-8")
    loader = FilePopulationLoader("bbh", path)

    with pytest.raises(ValueError, match=r"provenance") as caught:
        _ = loader.provenance

    names, chained, seen = set(), [caught.value], set()
    while chained:
        error = chained.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        chained += [error.__cause__, error.__context__]
        traceback = error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            while frame is not None:
                names.add(frame.f_code.co_qualname)
                frame = frame.f_back
            traceback = traceback.tb_next
    assert "FilePopulationLoader.__init__" not in names
    assert any("Raised while the catalogue was loaded" in note for note in caught.value.__notes__)


def test_a_note_added_to_a_raised_provenance_error_does_not_reach_the_next(tmp_path: Path) -> None:
    """Each raised error has its own notes, so a caller's note is not kept by the loader."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(path, 3)
    provenance_sidecar_path(path).write_text("not json", encoding="utf-8")
    loader = FilePopulationLoader("bbh", path)

    with pytest.raises(ValueError, match=r"provenance") as first:
        _ = loader.provenance
    first.value.add_note("added by a caller")
    with pytest.raises(ValueError, match=r"provenance") as second:
        _ = loader.provenance

    assert "added by a caller" not in second.value.__notes__
