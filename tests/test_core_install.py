"""What works with only the core dependencies: no h5py.

This is the installation Python 3.15 gets today, where h5py publishes no wheel. CI runs this module
there with only the package and pytest installed; it also runs with everything installed, where it
checks the same paths never import h5py.

Each test runs its scenario in a fresh interpreter, so what the package imports can be observed:
``sys.modules`` in the test process already holds whatever other tests imported. The scenarios about a
missing h5py hide it with an import hook, so they test that path whether or not h5py is installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

#: Makes ``import h5py`` fail as it does where h5py is not installed.
_HIDE_H5PY = """
import importlib.abc, sys

class _HideH5py(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "h5py" or name.startswith("h5py."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _HideH5py())
"""


def _run(code: str, *, hide_h5py: bool = False) -> dict:
    """Run ``code`` in a fresh interpreter; it sets ``result``, printed as JSON with whether h5py loaded."""
    script = (
        (_HIDE_H5PY if hide_h5py else "")
        + textwrap.dedent(code)
        + textwrap.dedent(
            """
            import json as _json, sys as _sys
            print(_json.dumps({"result": result, "h5py_loaded": "h5py" in _sys.modules}))
            """
        )
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_package_and_its_cli_import_without_h5py():
    """Importing the package, its CLI, the loaders, the catalogue draw and the provenance I/O loads no h5py."""
    outcome = _run(
        """
        import gwmock_pop, gwmock_pop.cli.main, gwmock_pop.catalogue.population
        import gwmock_pop.loaders.file_loader, gwmock_pop.provenance.io
        result = None
        """
    )

    assert outcome["h5py_loaded"] is False


def test_a_csv_catalogue_round_trips_with_its_provenance_without_h5py(tmp_path: Path):
    """CSV is written, read back and its sidecar provenance found, with h5py hidden."""
    path = tmp_path / "population.csv"
    outcome = _run(
        f"""
        import numpy as np
        from gwmock_pop.loaders.file_loader import FilePopulationLoader, write_population_catalogue
        from gwmock_pop.provenance import read_provenance

        population = {{"mass_1": np.array([30.0, 20.0, 10.0]), "mass_2": np.array([25.0, 15.0, 5.0])}}
        write_population_catalogue({str(path)!r}, population, provenance={{"note": "core"}})
        loader = FilePopulationLoader("bbh", {str(path)!r})
        result = {{
            "columns": loader.parameter_names,
            "provenance": read_provenance({str(path)!r}),
        }}
        """,
        hide_h5py=True,
    )

    assert outcome["h5py_loaded"] is False
    # The loader canonicalises the column names.
    assert outcome["result"] == {
        "columns": ["detector_frame_mass_1", "detector_frame_mass_2"],
        "provenance": {"note": "core"},
    }


def test_every_hdf5_entry_point_works_without_h5py(tmp_path: Path):
    """An HDF5 catalogue is written, loaded, its provenance and its columns read, with h5py hidden."""
    catalogue = tmp_path / "population.h5"
    columns = tmp_path / "columns.h5"
    outcome = _run(
        f"""
        import numpy as np
        from gwmock_data import hdf5
        from gwmock_pop.catalogue.population import read_catalogue_columns
        from gwmock_pop.loaders.file_loader import FilePopulationLoader, write_population_catalogue
        from gwmock_pop.provenance import read_provenance

        population = {{"mass_1": np.array([30.0, 20.0, 10.0]), "mass_2": np.array([25.0, 15.0, 5.0])}}
        write_population_catalogue({str(catalogue)!r}, population, provenance={{"note": "core"}})
        loader = FilePopulationLoader("bbh", {str(catalogue)!r})
        hdf5.write(hdf5.Group({{"a": hdf5.Dataset(np.arange(3.0)), "b": hdf5.Dataset(np.ones(3))}}), {str(columns)!r})
        result = {{
            "columns": loader.parameter_names,
            "provenance": read_provenance({str(catalogue)!r}),
            "read": {{name: values.tolist() for name, values in read_catalogue_columns({str(columns)!r}, ["b"]).items()}},
        }}
        """,
        hide_h5py=True,
    )

    assert outcome["h5py_loaded"] is False
    assert outcome["result"] == {
        "columns": ["detector_frame_mass_1", "detector_frame_mass_2"],
        "provenance": {"note": "core"},
        "read": {"b": [1.0, 1.0, 1.0]},
    }
