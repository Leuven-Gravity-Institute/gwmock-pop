"""Tests for reading and writing provenance records on disk."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gwmock_pop import write_population_catalogue
from gwmock_pop.provenance import (
    HDF5_METADATA_GROUP,
    HDF5_PROVENANCE_PATH,
    PROVENANCE_SIDECAR_SUFFIX,
    encode_provenance,
    provenance_sidecar_path,
    read_provenance,
    write_provenance_sidecar,
)

_RECORD = {"schema_version": "1.0", "catalogue": {"n_samples": 2}, "note": "a record"}


def _population() -> dict[str, np.ndarray]:
    """Return a tiny two-column population."""
    return {"mass_1": np.array([10.0, 20.0]), "mass_2": np.array([5.0, 6.0])}


class TestSidecarPath:
    """Tests for the sidecar naming rule."""

    def test_appends_to_the_full_file_name(self, tmp_path: Path) -> None:
        """The suffix is appended, so CSV and HDF5 siblings cannot collide."""
        assert provenance_sidecar_path(tmp_path / "c.csv").name == f"c.csv{PROVENANCE_SIDECAR_SUFFIX}"
        assert provenance_sidecar_path(tmp_path / "c.hdf5").name == f"c.hdf5{PROVENANCE_SIDECAR_SUFFIX}"

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        """A string path is accepted like a ``Path``."""
        assert provenance_sidecar_path(str(tmp_path / "c.csv")).parent == tmp_path


class TestEncodeProvenance:
    """Tests for the on-disk JSON encoding."""

    def test_preserves_key_order(self) -> None:
        """Graph node order is load-bearing, so the encoding must not sort keys."""
        encoded = encode_provenance({"b": 1, "a": 2})

        assert list(json.loads(encoded)) == ["b", "a"]
        assert encoded.index('"b"') < encoded.index('"a"')

    def test_round_trips_through_json(self) -> None:
        """The encoding is plain JSON."""
        assert json.loads(encode_provenance(_RECORD)) == _RECORD


class TestHdf5Provenance:
    """Tests for records embedded in HDF5 catalogues."""

    def test_writes_a_metadata_group_beside_the_data(self, tmp_path: Path) -> None:
        """The record lives under a ``metadata`` group next to ``data``."""
        output_path = tmp_path / "c.hdf5"

        write_population_catalogue(output_path=output_path, population=_population(), provenance=_RECORD)

        with h5py.File(output_path, "r") as handle:
            assert HDF5_METADATA_GROUP in handle
            assert HDF5_PROVENANCE_PATH in handle
            assert "data" in handle

    def test_is_read_back_unchanged(self, tmp_path: Path) -> None:
        """A written record reads back equal to what was given."""
        output_path = tmp_path / "c.hdf5"

        write_population_catalogue(output_path=output_path, population=_population(), provenance=_RECORD)

        assert read_provenance(output_path) == _RECORD

    def test_absent_when_no_record_is_supplied(self, tmp_path: Path) -> None:
        """Omitting the record leaves the file free of a metadata group."""
        output_path = tmp_path / "c.hdf5"

        write_population_catalogue(output_path=output_path, population=_population())

        with h5py.File(output_path, "r") as handle:
            assert HDF5_METADATA_GROUP not in handle
        assert read_provenance(output_path) is None

    def test_survives_a_compressed_dataset(self, tmp_path: Path) -> None:
        """Compression applies to the data, not to the record."""
        output_path = tmp_path / "c.hdf5"

        write_population_catalogue(
            output_path=output_path, population=_population(), provenance=_RECORD, compression="gzip"
        )

        with h5py.File(output_path, "r") as handle:
            assert handle["data"].compression == "gzip"
        assert read_provenance(output_path) == _RECORD

    def test_compression_is_skipped_for_an_empty_catalogue(self, tmp_path: Path) -> None:
        """A zero-row dataset cannot be chunked, so compression is dropped, not fatal."""
        output_path = tmp_path / "c.hdf5"

        write_population_catalogue(
            output_path=output_path,
            population={"mass_1": np.array([]), "mass_2": np.array([])},
            provenance=_RECORD,
            compression="gzip",
        )

        with h5py.File(output_path, "r") as handle:
            assert handle["data"].shape == (0,)
        assert read_provenance(output_path) == _RECORD


class TestCsvProvenance:
    """Tests for records beside CSV catalogues."""

    def test_written_as_a_sidecar(self, tmp_path: Path) -> None:
        """CSV cannot embed a record, so one is written alongside it."""
        output_path = tmp_path / "c.csv"

        write_population_catalogue(output_path=output_path, population=_population(), provenance=_RECORD)

        sidecar = provenance_sidecar_path(output_path)
        assert json.loads(sidecar.read_text(encoding="utf-8")) == _RECORD
        assert read_provenance(output_path) == _RECORD

    def test_absent_when_no_record_is_supplied(self, tmp_path: Path) -> None:
        """No record means no sidecar file."""
        output_path = tmp_path / "c.csv"

        write_population_catalogue(output_path=output_path, population=_population())

        assert not provenance_sidecar_path(output_path).exists()
        assert read_provenance(output_path) is None


class TestStaleRecordRemoval:
    """A catalogue written without a record must not inherit an old one.

    Overwriting the samples while leaving the previous sidecar in place is the
    exact failure this package exists to prevent: ``read_provenance`` would then
    hand back a record describing data that is no longer in the file.
    """

    def test_csv_rewritten_without_a_record_drops_the_old_sidecar(self, tmp_path: Path) -> None:
        """The sidecar of the replaced CSV catalogue does not survive."""
        output_path = tmp_path / "c.csv"
        write_population_catalogue(output_path=output_path, population=_population(), provenance=_RECORD)

        write_population_catalogue(output_path=output_path, population=_population())

        assert not provenance_sidecar_path(output_path).exists()
        assert read_provenance(output_path) is None

    def test_hdf5_rewritten_without_a_record_drops_a_sidecar_too(self, tmp_path: Path) -> None:
        """An HDF5 rewrite truncates the embedded record, and the sidecar goes with it.

        ``read_provenance`` falls back to a sidecar when a file carries no
        embedded record, so a lingering one would resurface after the rewrite.
        """
        output_path = tmp_path / "c.hdf5"
        write_population_catalogue(output_path=output_path, population=_population())
        write_provenance_sidecar(output_path, _RECORD)

        write_population_catalogue(output_path=output_path, population=_population())

        assert not provenance_sidecar_path(output_path).exists()
        assert read_provenance(output_path) is None

    def test_a_supplied_record_replaces_the_previous_one(self, tmp_path: Path) -> None:
        """Rewriting with a new record leaves the new record, not a merge of both."""
        output_path = tmp_path / "c.csv"
        write_population_catalogue(output_path=output_path, population=_population(), provenance=_RECORD)

        replacement = {"schema_version": "1.0", "note": "the second run"}
        write_population_catalogue(output_path=output_path, population=_population(), provenance=replacement)

        assert read_provenance(output_path) == replacement


class TestReadProvenance:
    """Tests for locating a record."""

    def test_returns_none_for_a_missing_file(self, tmp_path: Path) -> None:
        """A path with no catalogue and no sidecar has no record."""
        assert read_provenance(tmp_path / "absent.hdf5") is None

    def test_falls_back_to_a_sidecar_for_an_hdf5_catalogue(self, tmp_path: Path) -> None:
        """A record kept beside an HDF5 file is still found."""
        output_path = tmp_path / "c.hdf5"
        write_population_catalogue(output_path=output_path, population=_population())
        write_provenance_sidecar(output_path, _RECORD)

        assert read_provenance(output_path) == _RECORD

    def test_prefers_the_embedded_record(self, tmp_path: Path) -> None:
        """An embedded record wins over a sidecar that disagrees with it."""
        output_path = tmp_path / "c.hdf5"
        write_population_catalogue(output_path=output_path, population=_population(), provenance=_RECORD)
        write_provenance_sidecar(output_path, {"note": "stale sidecar"})

        assert read_provenance(output_path) == _RECORD

    def test_rejects_a_corrupt_sidecar(self, tmp_path: Path) -> None:
        """Unreadable JSON is an error, not a silently absent record."""
        output_path = tmp_path / "c.csv"
        provenance_sidecar_path(output_path).write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="provenance"):
            read_provenance(output_path)
