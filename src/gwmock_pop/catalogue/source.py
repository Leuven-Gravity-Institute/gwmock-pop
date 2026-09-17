"""Fetch pinned catalogue files on demand and cache them locally.

The fetch itself lives in :mod:`gwmock_pop.loaders`; this module binds it to the
pinned catalogue manifest so a draw names a catalogue rather than a URL. A
resolved catalogue carries the digest the bytes were checked against, which is
what the draw's provenance records.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gwmock_pop.catalogue.manifest import COBA_CATALOGUES, CatalogueFile
from gwmock_pop.loaders import FetchResult, resolve_digest_pinned_path


@dataclass(frozen=True, slots=True)
class ResolvedCatalogue:
    """A catalogue file on local disk, with the identity it was verified against.

    Attributes:
        catalogue: The pinned catalogue the file belongs to.
        path: Local path of the verified file.
        metadata: Fetch metadata, including whether the cache was hit and the
            digest the bytes matched.
    """

    catalogue: CatalogueFile
    path: Path
    metadata: dict[str, Any]


def fetch_catalogue(
    catalogue: CatalogueFile,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    timeout: int = 300,
) -> ResolvedCatalogue:
    """Fetch one pinned catalogue into the local cache, verifying its digest.

    Args:
        catalogue: The pinned catalogue to fetch.
        cache_dir: Cache directory. Defaults to the package cache location.
        refresh: Whether to re-download even when a matching entry is cached.
        offline: Whether to refuse network access and use only a matching cache.
        timeout: Timeout in seconds for the download.

    Returns:
        The verified local file and its fetch metadata.

    Raises:
        PopulationFetchError: If the file cannot be fetched, or the bytes do not
            match the pinned digest or size.
    """
    fetch_result: FetchResult = resolve_digest_pinned_path(
        catalogue.url,
        sha256=catalogue.sha256,
        size=catalogue.size_bytes,
        filename=catalogue.filename,
        cache_dir=cache_dir,
        refresh=refresh,
        offline=offline,
        timeout=timeout,
    )
    return ResolvedCatalogue(catalogue=catalogue, path=fetch_result.path, metadata=fetch_result.metadata)


def fetch_catalogues(
    catalogues: Sequence[CatalogueFile] = COBA_CATALOGUES,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    timeout: int = 300,
) -> tuple[ResolvedCatalogue, ...]:
    """Fetch several pinned catalogues, preserving the given order.

    Args:
        catalogues: The pinned catalogues to fetch.
        cache_dir: Cache directory. Defaults to the package cache location.
        refresh: Whether to re-download even when a matching entry is cached.
        offline: Whether to refuse network access and use only a matching cache.
        timeout: Timeout in seconds for the download.

    Returns:
        One resolved catalogue per input, in the same order.

    Raises:
        PopulationFetchError: If any file cannot be fetched or does not match
            its pin.
    """
    return tuple(
        fetch_catalogue(catalogue, cache_dir=cache_dir, refresh=refresh, offline=offline, timeout=timeout)
        for catalogue in catalogues
    )


__all__ = ["ResolvedCatalogue", "fetch_catalogue", "fetch_catalogues"]
