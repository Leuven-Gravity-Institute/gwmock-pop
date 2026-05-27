"""Simulator classes."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cbc import BBHSimulator, BNSSimulator, CBCSimulator, NSBHSimulator
    from .graph import GraphSimulator
    from .mixture import MixtureSimulator
    from .poisson_event import PoissonEventSampler
    from .simulator import Simulator

__all__ = [
    "BBHSimulator",
    "BNSSimulator",
    "CBCSimulator",
    "GraphSimulator",
    "MixtureSimulator",
    "NSBHSimulator",
    "PoissonEventSampler",
    "Simulator",
]


def __getattr__(name: str) -> Any:
    """Lazily import simulator classes for package-level access."""
    module_map = {
        "BBHSimulator": ".cbc",
        "BNSSimulator": ".cbc",
        "CBCSimulator": ".cbc",
        "GraphSimulator": ".graph",
        "MixtureSimulator": ".mixture",
        "NSBHSimulator": ".cbc",
        "PoissonEventSampler": ".poisson_event",
        "Simulator": ".simulator",
    }
    if name not in module_map:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_map[name], package=__name__)
    return getattr(module, name)
