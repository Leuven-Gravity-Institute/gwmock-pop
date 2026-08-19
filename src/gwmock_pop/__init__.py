"""Top-level package for gwmock_pop."""

from __future__ import annotations

from gwmock_pop._precision import enable_x64_by_default
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

# GPS-scale parameters (e.g. coa_time ~ 1.6e9 s) are unusable in float32 —
# enable 64-bit JAX floats at import. Arrays are only created at sampling time
# (no submodule builds float arrays at import), so configuring here is early
# enough. Opt out with GWMOCK_POP_DISABLE_X64=1.
enable_x64_by_default()

# Invariant: importing gwmock_pop must not import jax.
#
# `import jax` costs about 240 ms of the ~490 ms it used to take to import
# gwmock_pop.cli.main, and it is what puts a process on the road to XLA. The
# thread pools themselves start later, on the first JAX computation — 16
# `tf_XLAEigen`, 16 `tf_foreach` and 10 `llvm-worker-*` on a 24-core host — and
# from that point the process can no longer be forked safely: the child gets one
# thread and every lock the others held, held by nobody, so it blocks forever on
# the first one it needs. A process that reads a population file, or runs
# `gwmock-pop --help`, does neither of those things, and should not be dragged
# down that road by an import.
#
# So modules in the import closure of this package import jax inside the
# functions that use it, and take `jax.Array` under `typing.TYPE_CHECKING`
# (annotations are strings here, courtesy of `from __future__ import
# annotations`). Modules outside that closure — the samplers, distributions and
# transforms, reached on demand through gwmock_pop.utils.import_utils — still
# import jax at module scope: they are JAX kernels and nothing else.
#
# tests/test_lazy_jax_import.py enforces the invariant.

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
