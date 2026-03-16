"""Functions to extract dependencies of transform."""

from __future__ import annotations

from typing import Any

from gwsim_pop.graph.generic import extract_references


def extract_transform_dependencies(transform: str | dict[str, Any]) -> set[str]:
    """Extract the dependencies of transform.

    Args:
        transform: Expression or dictionary of transform.

    Returns:
        Dependencies of transform.
    """
    if isinstance(transform, str):
        dependencies = extract_references(transform)
    elif isinstance(transform, dict) and "function" in transform:
        args = transform.get("arguments", [])

        if isinstance(args, list):
            dependencies = {arg[1:] for arg in args if isinstance(arg, str) and arg.startswith("@") and len(arg) > 1}
        elif isinstance(args, dict):
            dependencies = {v[1:] for v in args.values() if isinstance(v, str) and v.startswith("@") and len(v) > 1}
        else:
            dependencies = set()

    else:
        dependencies = set()

    return dependencies
