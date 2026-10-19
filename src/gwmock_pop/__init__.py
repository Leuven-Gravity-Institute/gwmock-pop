"""Top-level package for gwmock_pop."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from gwmock_pop._precision import enable_x64_by_default

# GPS-scale parameters (e.g. coa_time ~ 1.6e9 s) are unusable in float32 —
# enable 64-bit JAX floats at import. Arrays are only created at sampling time
# (no submodule builds float arrays at import), so configuring here is early
# enough. Opt out with GWMOCK_POP_DISABLE_X64=1.
enable_x64_by_default()

if TYPE_CHECKING:
    # Static-only imports so type checkers and code scanners can resolve the names listed in __all__.
    # At runtime every one of them is resolved by __getattr__ below on first use, so importing the
    # package loads none of its submodules (and none of NetworkX, NumPy or YAML they bring) until
    # something is asked for.
    from gwmock_pop.coercion import (
        coerce_to_numpy,
    )
    from gwmock_pop.configs import (
        list_presets,
    )
    from gwmock_pop.constants import (
        CBC_PARAMETER_NAMES,
    )
    from gwmock_pop.exceptions import (
        PopulationError,
        PopulationFetchError,
        PopulationValidationError,
    )
    from gwmock_pop.loaders import (
        FilePopulationLoader,
        read_population_catalogue,
        write_population_catalogue,
    )
    from gwmock_pop.protocols import (
        ExternalPopulationLoader,
        GWPopSimulator,
    )
    from gwmock_pop.simulators import (
        BBHSimulator,
        BNSSimulator,
        CBCSimulator,
        GraphSimulator,
        MixtureSimulator,
        NSBHSimulator,
        PoissonEventSampler,
    )
    from gwmock_pop.validation import (
        validate_sample,
    )
    from gwmock_pop.version import (
        __version__,
    )

#: Public name -> the module defining it, imported on first access.
_LAZY_EXPORTS = {
    "__version__": "gwmock_pop.version",
    "BBHSimulator": "gwmock_pop.simulators",
    "BNSSimulator": "gwmock_pop.simulators",
    "CBC_PARAMETER_NAMES": "gwmock_pop.constants",
    "CBCSimulator": "gwmock_pop.simulators",
    "coerce_to_numpy": "gwmock_pop.coercion",
    "ExternalPopulationLoader": "gwmock_pop.protocols",
    "FilePopulationLoader": "gwmock_pop.loaders",
    "GraphSimulator": "gwmock_pop.simulators",
    "GWPopSimulator": "gwmock_pop.protocols",
    "list_presets": "gwmock_pop.configs",
    "MixtureSimulator": "gwmock_pop.simulators",
    "NSBHSimulator": "gwmock_pop.simulators",
    "PoissonEventSampler": "gwmock_pop.simulators",
    "PopulationError": "gwmock_pop.exceptions",
    "PopulationFetchError": "gwmock_pop.exceptions",
    "PopulationValidationError": "gwmock_pop.exceptions",
    "read_population_catalogue": "gwmock_pop.loaders",
    "validate_sample": "gwmock_pop.validation",
    "write_population_catalogue": "gwmock_pop.loaders",
}

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


def __getattr__(name: str) -> Any:
    """Resolve a top-level export on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        export = getattr(import_module(module_name), name)
        globals()[name] = export
        return export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the lazily exported names alongside the module's own."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
