"""Utility functions for importing."""

from __future__ import annotations

import importlib
from typing import Any


def import_from_string(object_path: str, default_module: str | None = None) -> Any:
    """Import an object from a string path like 'module.submodule.object'.

    Args:
        object_path: Path of the object.
        default_module: Default module path tp import if it is not included in object_path.

    Returns:
        Attribute.

    """
    try:
        splitted_import_path = object_path.rsplit(sep=".", maxsplit=1)
        if len(splitted_import_path) == 1:
            if default_module is not None:
                module_path = default_module
                object_name = object_path
            else:
                raise ValueError(
                    f"object_path={object_path} does not contain a module path, and default_module is None."
                )
        else:
            module_path, object_name = splitted_import_path
        module = importlib.import_module(module_path)

        return getattr(module, object_name)

    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {object_path}: {e}") from e
