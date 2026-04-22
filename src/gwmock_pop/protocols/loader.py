"""Protocol definition for external population catalogue loaders."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gwmock_pop.protocols.simulator import GWPopSimulator


@runtime_checkable
class ExternalPopulationLoader(GWPopSimulator, Protocol):
    """Structural contract for file-backed population catalogues.

    ``ExternalPopulationLoader`` intentionally has the same runtime surface as
    :class:`~gwmock_pop.protocols.simulator.GWPopSimulator`: any object with
    ``parameter_names``, ``source_type``, and ``simulate()`` satisfies both
    protocols automatically.

    In this context, :meth:`simulate` means sample without replacement from a
    catalogue that was loaded from an external file such as HDF5 or CSV.
    Concrete loaders may perform that I/O eagerly during construction or via
    another package-specific setup step. Unsupported file formats should raise a
    clear ``ValueError`` at load time.

    ``source_type`` must still be a non-empty routing key that is either set
    explicitly by the caller or derived from file metadata.
    """
