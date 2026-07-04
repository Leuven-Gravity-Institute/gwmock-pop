"""Top-level package for gwmock_pop."""

from __future__ import annotations

from gwmock_pop.coercion import coerce_to_numpy
from gwmock_pop.configs import list_presets
from gwmock_pop.constants import CBC_PARAMETER_NAMES
from gwmock_pop.exceptions import PopulationError, PopulationFetchError, PopulationValidationError
from gwmock_pop.loaders import FilePopulationLoader, read_population_catalogue, write_population_catalogue
from gwmock_pop.protocols import ExternalPopulationLoader, GWPopSimulator
from gwmock_pop.simulators import (
    BBHSimulator,
    BNSSimulator,
    CBCSimulator,
    GraphSimulator,
    MixtureSimulator,
    NSBHSimulator,
    PoissonEventSampler,
)
from gwmock_pop.validation import validate_sample
from gwmock_pop.version import __version__

__all__ = [
    "CBC_PARAMETER_NAMES",
    "BBHSimulator",
    "BNSSimulator",
    "CBCSimulator",
    "ExternalPopulationLoader",
    "FilePopulationLoader",
    "GWPopSimulator",
    "GraphSimulator",
    "MixtureSimulator",
    "NSBHSimulator",
    "PoissonEventSampler",
    "PopulationError",
    "PopulationFetchError",
    "PopulationValidationError",
    "__version__",
    "coerce_to_numpy",
    "list_presets",
    "read_population_catalogue",
    "validate_sample",
    "write_population_catalogue",
]
