"""Importing gwmock_pop loads none of its submodules' heavy dependencies; every export still resolves."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import gwmock_pop

pytestmark = pytest.mark.unit


def test_importing_the_package_loads_no_heavy_dependency():
    """NetworkX, NumPy and YAML were most of the import; nothing that needs them is touched on import."""
    code = (
        "import json, sys, gwmock_pop; "
        "print(json.dumps(sorted(m for m in ('networkx', 'numpy', 'yaml', 'jax', 'h5py', 'pydantic') if m in sys.modules)))"
    )
    loaded = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)  # noqa: S603

    assert loaded == []


@pytest.mark.parametrize("name", sorted(gwmock_pop.__all__))
def test_every_export_resolves(name):
    """Each name in __all__ is reachable from the package, as when it was imported eagerly."""
    assert getattr(gwmock_pop, name) is not None


def test_an_unknown_name_is_an_attribute_error():
    """A name the package does not export is still an AttributeError, not an import error."""
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        _ = gwmock_pop.no_such_name


def test_the_exports_are_listed():
    """``dir()`` lists the lazily exported names, as it listed the eagerly imported ones."""
    assert set(gwmock_pop.__all__) <= set(dir(gwmock_pop))


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15")
def test_the_cli_defers_what_only_its_commands_need():
    """From Python 3.15, building the command-line app (and so ``--help``) imports no command's heavy dependency."""
    code = (
        "import json, sys, gwmock_pop.cli.main; "
        "print(json.dumps(sorted(m for m in ('networkx', 'numpy') if m in sys.modules)))"
    )
    loaded = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)  # noqa: S603

    assert loaded == []
