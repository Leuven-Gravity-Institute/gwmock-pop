"""Simulator classes."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cbc_prior import CBCPriorSimulator
    from .graph import GraphSimulator
    from .simulator import Simulator

__all__ = ["CBCPriorSimulator", "GraphSimulator", "Simulator"]


def __getattr__(name: str) -> Any:
    """Lazily import simulator classes for package-level access."""
    module_map = {
        "CBCPriorSimulator": ".cbc_prior",
        "GraphSimulator": ".graph",
        "Simulator": ".simulator",
    }
    if name not in module_map:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_map[name], package=__name__)
    return getattr(module, name)
