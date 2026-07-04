"""File-backed population catalogue loaders."""

from __future__ import annotations

from gwmock_pop.loaders.file_loader import (
    FilePopulationLoader,
    read_population_catalogue,
    write_population_catalogue,
)

__all__ = ["FilePopulationLoader", "read_population_catalogue", "write_population_catalogue"]
