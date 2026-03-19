"""Helper functions to extract dependencies from generic lists and dictionaries."""

from __future__ import annotations

import re
from typing import Any


def extract_references(expr: str) -> set[str]:
    """Very simple parser to find all @references in a string expression.

    Example: "@m1 * @q + 5" → ['m1', 'q']

    Args:
        expr: Expression.

    Returns:
        A set of variables names (without the @).
    """
    # Find all @ followed by word characters
    matches = re.findall(r"@([a-zA-Z_][a-zA-Z0-9_]*)", expr)
    return set(matches)


def extract_dependencies_from_spec(spec: dict[str, Any]) -> set[str]:
    """Recursively find all @references in a parameter spec (handles nested dicts/lists).

    Args:
        spec: A dictionary of specification.

    Returns:
        A set of @references.
    """
    deps: set[str] = set()

    def recurse(value: Any) -> None:
        """A recursive function to find the @references.

        Args:
            value: It can be a str, dict, or list.
        """
        if isinstance(value, str):
            deps.update(extract_references(value))
        elif isinstance(value, dict):
            for v in value.values():
                recurse(v)
        elif isinstance(value, list):
            for item in value:
                recurse(item)

    recurse(value=spec)
    return deps
