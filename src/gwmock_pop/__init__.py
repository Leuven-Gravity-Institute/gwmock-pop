"""Top-level package for gwmock_pop."""

from __future__ import annotations

from gwmock_pop.configs import list_presets
from gwmock_pop.constants import CBC_PARAMETER_NAMES
from gwmock_pop.exceptions import PopulationError, PopulationFetchError, PopulationValidationError
from gwmock_pop.loaders import FilePopulationLoader
from gwmock_pop.protocols import ExternalPopulationLoader, GWPopSimulator
from gwmock_pop.simulators import (
    BBHSimulator,
    BNSPriorSimulator,
    CBCPriorSimulator,
    GraphSimulator,
    MixtureSimulator,
    NSBHPriorSimulator,
    PoissonEventSampler,
)
from gwmock_pop.validation import validate_sample
from gwmock_pop.version import __version__

__all__ = [
    "CBC_PARAMETER_NAMES",
    "BBHSimulator",
    "BNSPriorSimulator",
    "CBCPriorSimulator",
    "ExternalPopulationLoader",
    "FilePopulationLoader",
    "GWPopSimulator",
    "GraphSimulator",
    "MixtureSimulator",
    "NSBHPriorSimulator",
    "PoissonEventSampler",
    "PopulationError",
    "PopulationFetchError",
    "PopulationValidationError",
    "__version__",
    "list_presets",
    "validate_sample",
]
