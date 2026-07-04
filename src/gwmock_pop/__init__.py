"""Top-level package for gwmock_pop."""

from __future__ import annotations

from gwmock_pop._precision import enable_x64_by_default
from gwmock_pop.coercion import coerce_to_numpy
from gwmock_pop.configs import list_presets
from gwmock_pop.constants import CBC_PARAMETER_NAMES
from gwmock_pop.exceptions import PopulationError, PopulationFetchError, PopulationValidationError
from gwmock_pop.loaders import FilePopulationLoader
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

# GPS-scale parameters (e.g. coa_time ~ 1.6e9 s) are unusable in float32 —
# enable 64-bit JAX floats at import. Arrays are only created at sampling time
# (no submodule builds float arrays at import), so configuring here is early
# enough. Opt out with GWMOCK_POP_DISABLE_X64=1.
enable_x64_by_default()

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
    "validate_sample",
]
