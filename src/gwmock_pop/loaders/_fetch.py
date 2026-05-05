"""Remote population-file fetching and local cache resolution."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from gwmock_pop.exceptions import PopulationFetchError

_CACHE_ENV_VAR = "GWMOCK_POP_CACHE_DIR"
_TOKEN_ENV_VAR = "GWMOCK_POP_TOKEN"  # noqa: S105
_ZENODO_TOKEN_ENV_VAR = "ZENODO_TOKEN"  # noqa: S105
_SUPPORTED_URL_SCHEMES = frozenset({"http", "https", "s3", "zenodo"})
_HTTP_RESPONSE_HEADER_KEYS = ("Content-Length", "Content-Type", "ETag", "Last-Modified")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Resolved file location and fetch metadata."""

    path: Path
    metadata: dict[str, Any]


def is_population_url(path: str | os.PathLike[str]) -> bool:
    """Return whether ``path`` is a supported remote URL."""
    return urlparse(os.fspath(path)).scheme.lower() in _SUPPORTED_URL_SCHEMES


def resolve_population_path(  # noqa: PLR0913
    path: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    token: str | None = None,
    credentials: Mapping[str, str] | Mapping[str, Mapping[str, str]] | None = None,
    timeout: int = 300,
) -> FetchResult:
    """Resolve a local path or fetch a remote URL into the population cache."""
    original_path = os.fspath(path)
    if not is_population_url(original_path):
        resolved_path = Path(original_path).expanduser()
        return FetchResult(
            path=resolved_path,
            metadata={
                "cache_hit": False,
                "original_path": original_path,
                "remote": False,
                "resolved_path": str(resolved_path),
            },
        )

    parsed = urlparse(original_path)
    cache_root = _resolve_cache_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    resolved_url = _resolve_remote_url(parsed)
    cache_key = hashlib.sha256(original_path.encode("utf-8")).hexdigest()
    cache_path = cache_root / f"{cache_key}{_infer_cache_suffix(resolved_url)}"
    cache_metadata_path = cache_root / f"{cache_key}.json"

    if cache_path.exists() and not refresh:
        return FetchResult(
            path=cache_path,
            metadata=_load_cached_metadata(
                cache_metadata_path,
                fallback={
                    "cache_hit": True,
                    "cache_path": str(cache_path),
                    "original_url": original_path,
                    "remote": True,
                    "resolved_url": resolved_url,
                    "scheme": parsed.scheme.lower(),
                },
            ),
        )

    headers = _build_request_headers(
        scheme=parsed.scheme.lower(),
        token=token,
        credentials=credentials,
    )
    if parsed.scheme.lower() == "s3":
        download_metadata = _download_s3_url(
            parsed=parsed,
            destination=cache_path,
            credentials=credentials,
        )
    else:
        download_metadata = _download_http_url(
            resolved_url,
            destination=cache_path,
            headers=headers,
            timeout=timeout,
        )

    fetch_metadata = {
        **download_metadata,
        "cache_hit": False,
        "cache_path": str(cache_path),
        "original_url": original_path,
        "remote": True,
        "resolved_url": resolved_url,
        "scheme": parsed.scheme.lower(),
    }
    cache_metadata_path.write_text(json.dumps(fetch_metadata, sort_keys=True, indent=2), encoding="utf-8")
    return FetchResult(path=cache_path, metadata=fetch_metadata)


def _resolve_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve the configured population cache directory."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    env_cache_dir = os.environ.get(_CACHE_ENV_VAR)
    if env_cache_dir:
        return Path(env_cache_dir).expanduser()

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "gwmock-pop"

    return Path.home() / ".cache" / "gwmock-pop"


def _resolve_remote_url(parsed) -> str:
    """Translate custom URL schemes into fetchable URLs."""
    if parsed.scheme.lower() != "zenodo":
        return parsed.geturl()

    record_id = parsed.netloc.strip()
    file_name = parsed.path.lstrip("/")
    if not record_id or not file_name:
        raise PopulationFetchError(
            "Zenodo URLs must use the form 'zenodo://<record>/<file>' so the record and file are unambiguous."
        )
    return f"https://zenodo.org/records/{quote(record_id)}/files/{quote(file_name)}?download=1"


def _infer_cache_suffix(url: str) -> str:
    """Infer a stable file suffix for a cached remote object."""
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix or ".download"


def _build_request_headers(
    *,
    scheme: str,
    token: str | None,
    credentials: Mapping[str, str] | Mapping[str, Mapping[str, str]] | None,
) -> dict[str, str]:
    """Construct request headers from explicit credentials or environment."""
    headers: dict[str, str] = {}
    if credentials is not None:
        raw_headers: Mapping[str, str]
        if "headers" in credentials:
            nested_headers = credentials["headers"]
            if not isinstance(nested_headers, Mapping):
                raise PopulationFetchError("'credentials[\"headers\"]' must be a mapping of header names to values.")
            raw_headers = nested_headers
        else:
            raw_headers = credentials
        headers.update({str(name): str(value) for name, value in raw_headers.items()})

    resolved_token = token or _resolve_token_from_environment(scheme)
    if resolved_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {resolved_token}"
    return headers


def _resolve_token_from_environment(scheme: str) -> str | None:
    """Resolve a scheme-specific token from the environment."""
    if scheme == "zenodo" and os.environ.get(_ZENODO_TOKEN_ENV_VAR):
        return os.environ[_ZENODO_TOKEN_ENV_VAR]
    return os.environ.get(_TOKEN_ENV_VAR)


def _download_http_url(
    url: str,
    *,
    destination: Path,
    headers: Mapping[str, str],
    timeout: int,
) -> dict[str, Any]:
    """Download a remote HTTP(S)-accessible file to the cache."""
    parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("URL must use http or https")

    request = Request(url, headers=dict(headers))  # noqa: S310
    temporary_path = _temporary_download_path(destination)

    try:
        with urlopen(request, timeout=timeout) as response, temporary_path.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
            response_headers = {
                key: value for key in _HTTP_RESPONSE_HEADER_KEYS if (value := response.headers.get(key)) is not None
            }
    except HTTPError as exc:
        temporary_path.unlink(missing_ok=True)
        raise PopulationFetchError(f"Failed to fetch population URL {url!r}: HTTP {exc.code} {exc.reason}.") from exc
    except (TimeoutError, URLError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise PopulationFetchError(f"Failed to fetch population URL {url!r}: {exc}.") from exc

    temporary_path.replace(destination)
    return {
        "content_sha256": _sha256_file(destination),
        "headers": response_headers,
    }


def _download_s3_url(
    *,
    parsed,
    destination: Path,
    credentials: Mapping[str, str] | Mapping[str, Mapping[str, str]] | None,
) -> dict[str, Any]:
    """Download an object from S3 using boto3 when available."""
    try:
        boto3 = importlib.import_module("boto3")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
    except ImportError as exc:
        raise PopulationFetchError("s3:// population URLs require boto3 and botocore to be installed.") from exc
    boto_core_error = botocore_exceptions.BotoCoreError
    client_error = botocore_exceptions.ClientError

    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise PopulationFetchError("s3 URLs must use the form 's3://<bucket>/<key>' so the object is unambiguous.")

    client_kwargs: dict[str, str] = {}
    if credentials is not None:
        for field in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token", "endpoint_url", "region_name"):
            value = credentials.get(field)
            if value is not None and not isinstance(value, Mapping):
                client_kwargs[field] = str(value)

    temporary_path = _temporary_download_path(destination)
    try:
        boto3.client("s3", **client_kwargs).download_file(bucket, key, str(temporary_path))
    except (boto_core_error, client_error, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise PopulationFetchError(f"Failed to fetch population URL {parsed.geturl()!r}: {exc}.") from exc

    temporary_path.replace(destination)
    return {
        "content_sha256": _sha256_file(destination),
        "headers": {},
    }


def _load_cached_metadata(cache_metadata_path: Path, *, fallback: dict[str, Any]) -> dict[str, Any]:
    """Load cached fetch metadata, preserving the required runtime fields."""
    if not cache_metadata_path.exists():
        return fallback

    try:
        cached_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PopulationFetchError(
            f"Cache metadata file {cache_metadata_path} is not valid JSON. Remove it or refresh the cache."
        ) from exc

    return {
        **cached_metadata,
        "cache_hit": True,
        "cache_path": fallback["cache_path"],
        "original_url": fallback["original_url"],
        "remote": True,
        "resolved_url": fallback["resolved_url"],
        "scheme": fallback["scheme"],
    }


def _temporary_download_path(destination: Path) -> Path:
    """Create a temporary file path next to the target cache entry."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        return Path(handle.name)


def _sha256_file(path: Path) -> str:
    """Compute the SHA256 digest of a downloaded cache entry."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["FetchResult", "is_population_url", "resolve_population_path"]
