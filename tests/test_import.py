"""Test importing the gwmock_pop package and its modules."""

from __future__ import annotations

import pkgutil

import pytest

import gwmock_pop


def get_all_submodules(package):
    """Discover all submodules in the package.

    Args:
        package: The package to inspect.

    """
    submodules = []
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        submodules.append(mod_name)
    return submodules


def test_import_main_package():
    """Test that the main gwmock_pop package can be imported."""
    assert hasattr(gwmock_pop, "__version__")
    assert hasattr(gwmock_pop, "CBC_PARAMETER_NAMES")
    assert gwmock_pop.__version__ is not None


@pytest.mark.parametrize("module_name", get_all_submodules(gwmock_pop))
def test_import_submodule(module_name):
    """Test that all submodules can be imported."""
    __import__(module_name)
