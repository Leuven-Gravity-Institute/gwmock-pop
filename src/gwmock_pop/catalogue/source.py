"""Fetch pinned catalogue files on demand and cache them locally.

The fetch itself lives in :mod:`gwmock_pop.loaders`; this module binds it to the
pinned catalogue manifest so a draw names a catalogue rather than a URL. A
resolved catalogue carries the digest the bytes were checked against, which is
what the draw's provenance records.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gwmock_pop.catalogue.manifest import COBA_CATALOGUES, CatalogueFile
from gwmock_pop.exceptions import PopulationFetchError
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

    A file the record publishes inside an archive is fetched by fetching the
    archive -- verified against its own pin -- and extracting the member beside
    it, verified against the member's pin; a verified extracted copy is reused.
    For such a file, ``refresh`` re-extracts the member, and re-downloads the
    archive too unless ``offline`` is set, in which case the verified cached
    archive is used.

    Raises:
        PopulationFetchError: If the file cannot be fetched, or the bytes do not
            match the pinned digest or size.
    """
    archive = catalogue.archive
    fetch_result: FetchResult = resolve_digest_pinned_path(
        catalogue.url,
        sha256=catalogue.sha256 if archive is None else archive.sha256,
        size=catalogue.size_bytes if archive is None else archive.size_bytes,
        filename=catalogue.filename if archive is None else archive.filename,
        cache_dir=cache_dir,
        # Offline, a refresh of an archive member re-extracts it from the verified cached archive.
        refresh=refresh and not (offline and archive is not None),
        offline=offline,
        timeout=timeout,
    )
    if archive is None:
        return ResolvedCatalogue(catalogue=catalogue, path=fetch_result.path, metadata=fetch_result.metadata)
    member, cache_hit = _extract_member(fetch_result.path, catalogue, refresh=refresh)
    metadata = {
        **fetch_result.metadata,
        "archive_sha256": archive.sha256,
        "archive_cache_hit": fetch_result.metadata.get("cache_hit", False),
        "member": catalogue.filename,
        "content_sha256": catalogue.sha256,
        "cache_hit": cache_hit and bool(fetch_result.metadata.get("cache_hit", False)),
        "verified": True,
    }
    return ResolvedCatalogue(catalogue=catalogue, path=member, metadata=metadata)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_member(archive_path: Path, catalogue: CatalogueFile, *, refresh: bool) -> tuple[Path, bool]:
    """Extract *catalogue*'s member from the verified archive, verifying it against its pin.

    The copy is named by the member's digest, so it can only ever hold those bytes.

    Returns:
        The extracted file, and whether a verified copy was already there.

    Raises:
        PopulationFetchError: If the archive lacks the member, or the member does not match its pin.
    """
    target = archive_path.with_name(f"{catalogue.sha256}{Path(catalogue.filename).suffix}")
    if (
        not refresh
        and target.is_file()
        and target.stat().st_size == catalogue.size_bytes
        and _sha256_of(target) == catalogue.sha256
    ):
        return target, True
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            info = bundle.getinfo(catalogue.filename)
            if info.file_size != catalogue.size_bytes:
                raise PopulationFetchError(
                    f"{catalogue.filename} in {archive_path.name} is {info.file_size} bytes, not the pinned "
                    f"{catalogue.size_bytes}."
                )
            with bundle.open(info) as source, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as staged:
                staged_path = Path(staged.name)
                digest = hashlib.sha256()
                try:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        digest.update(block)
                        staged.write(block)
                except BaseException:
                    staged.close()
                    staged_path.unlink(missing_ok=True)
                    raise
    except KeyError as error:
        raise PopulationFetchError(f"{archive_path.name} holds no member {catalogue.filename!r}.") from error
    except zipfile.BadZipFile as error:
        raise PopulationFetchError(f"{archive_path} is not a readable zip archive.") from error
    if digest.hexdigest() != catalogue.sha256:
        staged_path.unlink(missing_ok=True)
        raise PopulationFetchError(
            f"{catalogue.filename} in {archive_path.name} has SHA-256 {digest.hexdigest()}, not the pinned "
            f"{catalogue.sha256}; refusing it."
        )
    staged_path.replace(target)
    return target, False


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
