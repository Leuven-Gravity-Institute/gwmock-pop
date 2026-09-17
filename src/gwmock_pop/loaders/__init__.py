"""File-backed population catalogue loaders."""

from __future__ import annotations

from gwmock_pop.loaders._fetch import (
    FetchResult,
    is_population_url,
    resolve_digest_pinned_path,
    resolve_population_path,
)
from gwmock_pop.loaders.file_loader import (
    FilePopulationLoader,
    read_population_catalogue,
    write_population_catalogue,
)

__all__ = [
    "FetchResult",
    "FilePopulationLoader",
    "is_population_url",
    "read_population_catalogue",
    "resolve_digest_pinned_path",
    "resolve_population_path",
    "write_population_catalogue",
]
