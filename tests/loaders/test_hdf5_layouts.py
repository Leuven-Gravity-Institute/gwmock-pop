"""HDF5 catalogue layouts read through gwmock-data's HDF5 support, with or without h5py."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from gwmock_data import hdf5

from gwmock_pop import FilePopulationLoader, PopulationValidationError
from gwmock_pop.catalogue.population import read_catalogue_columns
from gwmock_pop.loaders.file_loader import write_population_catalogue
from gwmock_pop.provenance import read_provenance, write_hdf5_provenance

pytestmark = pytest.mark.unit

_MASSES = {"mass_1": np.array([30.0, 20.0, 10.0]), "mass_2": np.array([25.0, 15.0, 5.0])}


def _structured() -> np.ndarray:
    """The two mass columns as one compound array."""
    structured = np.zeros(3, dtype=[(name, np.float64) for name in _MASSES])
    for name, values in _MASSES.items():
        structured[name] = values
    return structured


def test_a_catalogue_round_trips_with_its_provenance(tmp_path: Path):
    """What the writer produces, the loader and the provenance reader read back."""
    path = tmp_path / "population.h5"
    write_population_catalogue(path, _MASSES, provenance={"note": "embedded"}, compression="gzip")

    loader = FilePopulationLoader("bbh", path)

    np.testing.assert_array_equal(loader.simulate(None)["detector_frame_mass_1"], _MASSES["mass_1"])
    assert read_provenance(path) == {"note": "embedded"}


def test_the_dataset_may_sit_inside_groups(tmp_path: Path):
    """``hdf5_dataset`` may be a path, as it could be when h5py resolved it."""
    path = tmp_path / "nested.h5"
    hdf5.write(hdf5.Group({"run1": hdf5.Group({"data": hdf5.Dataset(_structured())})}), path)

    for name in ("run1/data", "/run1/data/", "run1//data", "./run1/./data"):
        loader = FilePopulationLoader("bbh", path, hdf5_dataset=name)

        np.testing.assert_array_equal(loader.simulate(None)["detector_frame_mass_2"], _MASSES["mass_2"])


def test_the_root_group_may_hold_the_columns(tmp_path: Path):
    """``"/"`` names the file itself, whose datasets are then the columns."""
    path = tmp_path / "flat.h5"
    hdf5.write(hdf5.Group({name: hdf5.Dataset(values) for name, values in _MASSES.items()}), path)

    loader = FilePopulationLoader("bbh", path, hdf5_dataset="/")

    np.testing.assert_array_equal(loader.simulate(None)["detector_frame_mass_1"], _MASSES["mass_1"])


@pytest.mark.parametrize("name", ["absent", "data/absent", "data/mass_1/deeper"])
def test_a_missing_dataset_is_a_key_error(tmp_path: Path, name: str):
    """A name the file lacks, at any depth, is reported with the path asked for."""
    path = tmp_path / "group.h5"
    hdf5.write(hdf5.Group({"data": hdf5.Group({"mass_1": hdf5.Dataset(np.ones(3))})}), path)

    with pytest.raises(KeyError, match=name):
        FilePopulationLoader("bbh", path, hdf5_dataset=name)


def test_a_group_holding_a_group_is_refused(tmp_path: Path):
    """A group-of-datasets catalogue holds datasets only."""
    path = tmp_path / "group.h5"
    hdf5.write(hdf5.Group({"data": hdf5.Group({"mass_1": hdf5.Dataset(np.ones(3)), "inner": hdf5.Group()})}), path)

    with pytest.raises(TypeError, match="non-dataset member 'inner'"):
        FilePopulationLoader("bbh", path)


def test_a_two_dimensional_column_is_refused(tmp_path: Path):
    """Each column of a group-of-datasets catalogue is 1-D."""
    path = tmp_path / "group.h5"
    hdf5.write(hdf5.Group({"data": hdf5.Group({"mass_1": hdf5.Dataset(np.ones((3, 2)))})}), path)

    with pytest.raises(ValueError, match="must be a 1-D array"):
        FilePopulationLoader("bbh", path)


def test_provenance_joins_an_existing_metadata_group():
    """Embedding a record keeps whatever else the metadata group already holds."""
    root = hdf5.Group({"metadata": hdf5.Group({"other": hdf5.Dataset(np.ones(1))})})

    write_hdf5_provenance(root, {"note": "embedded"})

    assert set(root.members["metadata"].members) == {"other", "provenance"}


def test_metadata_that_is_not_a_group_holds_no_provenance(tmp_path: Path):
    """A ``metadata`` dataset is not where a record lives; the reader falls back to the sidecar."""
    path = tmp_path / "population.h5"
    hdf5.write(hdf5.Group({"data": hdf5.Dataset(_structured()), "metadata": hdf5.Dataset(np.ones(1))}), path)

    assert read_provenance(path) is None


def test_a_column_that_is_a_group_is_missing(tmp_path: Path):
    """Catalogue columns are datasets; a group of that name does not count as one."""
    path = tmp_path / "columns.h5"
    hdf5.write(hdf5.Group({"mass_1": hdf5.Dataset(np.ones(3)), "mass_2": hdf5.Group()}), path)

    with pytest.raises(PopulationValidationError, match="missing required columns: mass_2"):
        read_catalogue_columns(path, ["mass_1", "mass_2"])


def test_a_group_of_columns_is_a_catalogue(tmp_path: Path):
    """Each dataset of the group is one column."""
    path = tmp_path / "group.h5"
    hdf5.write(hdf5.Group({"data": hdf5.Group({name: hdf5.Dataset(values) for name, values in _MASSES.items()})}), path)

    loader = FilePopulationLoader("bbh", path)

    np.testing.assert_array_equal(loader.simulate(None)["detector_frame_mass_2"], _MASSES["mass_2"])


def test_columns_of_different_lengths_are_refused(tmp_path: Path):
    """The columns of a group-of-datasets catalogue are one length."""
    path = tmp_path / "group.h5"
    hdf5.write(
        hdf5.Group({"data": hdf5.Group({"mass_1": hdf5.Dataset(np.ones(3)), "mass_2": hdf5.Dataset(np.ones(2))})}), path
    )

    with pytest.raises(ValueError, match="'mass_2' has 2, expected 3"):
        FilePopulationLoader("bbh", path)


@pytest.mark.requires_h5py
def test_provenance_is_embedded_through_an_open_h5py_file(tmp_path: Path):
    """An h5py file handle still takes a record, as it did before the built-in writer."""
    import h5py  # only where it is installed

    path = tmp_path / "population.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=_structured())
        write_hdf5_provenance(handle, {"note": "h5py"})

    assert read_provenance(path) == {"note": "h5py"}


def test_a_write_that_fails_keeps_the_sidecar_of_the_catalogue_it_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The stale-sidecar cleanup runs only once the new catalogue is written."""
    path = tmp_path / "population.h5"
    write_population_catalogue(path, _MASSES)
    sidecar = tmp_path / "population.h5.provenance.json"
    sidecar.write_text('{"note": "describes the file still there"}\n', encoding="utf-8")
    # The built-in writer, which refuses half floats, whether or not h5py is installed.
    monkeypatch.setattr(hdf5, "_h5py", lambda: None)

    with pytest.raises(TypeError, match="float16"):
        write_population_catalogue(path, {"mass_1": np.ones(3, dtype=np.float16)})

    assert sidecar.is_file()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["population.h5", "population.h5.provenance.json"]


@pytest.mark.requires_h5py
def test_a_write_that_fails_after_opening_its_file_leaves_the_old_catalogue(tmp_path: Path):
    """h5py opens, and so truncates, its file before it refuses an unknown filter; the old catalogue survives."""
    path = tmp_path / "population.h5"
    write_population_catalogue(path, _MASSES, provenance={"note": "old"})
    before = path.read_bytes()

    with pytest.raises(ValueError, match="not-a-filter"):
        write_population_catalogue(path, _MASSES, compression="not-a-filter")

    assert path.read_bytes() == before
    assert read_provenance(path) == {"note": "old"}
    assert [item.name for item in tmp_path.iterdir()] == ["population.h5"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("name", ["population.h5", "population.csv"])
def test_a_rewrite_keeps_the_permissions_of_the_catalogue_it_replaces(tmp_path: Path, name: str):
    """A deliberately restricted catalogue stays restricted when it is written again."""
    path = tmp_path / name
    write_population_catalogue(path, _MASSES)
    path.chmod(0o600)

    write_population_catalogue(path, _MASSES)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privileges on Windows")
@pytest.mark.parametrize("name", ["population.h5", "population.csv"])
def test_a_symlinked_destination_is_written_through(tmp_path: Path, name: str):
    """The file a symlink points to is replaced, and the link stays a link, as when the path was opened."""
    actual = tmp_path / "store" / name
    actual.parent.mkdir()
    write_population_catalogue(actual, _MASSES)
    alias = tmp_path / name
    alias.symlink_to(actual)

    write_population_catalogue(alias, {"mass_1": np.array([1.0, 2.0]), "mass_2": np.array([0.5, 1.5])})

    assert alias.is_symlink()
    np.testing.assert_array_equal(
        FilePopulationLoader("bbh", actual).simulate(None)["detector_frame_mass_1"], [1.0, 2.0]
    )
