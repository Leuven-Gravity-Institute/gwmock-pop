"""Unit tests for digest-pinned remote fetches."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Self

import pytest

from gwmock_pop.exceptions import PopulationFetchError
from gwmock_pop.loaders import _fetch

_URL = "https://apps.example.org/tds/?call_file=18321_1yrCatalogBBH.h5"


class _MockResponse(BytesIO):
    """Minimal file-like HTTP response for urlopen patches."""

    def __init__(self, payload: bytes, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(payload)
        self.headers = headers or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.close()
        return False


def _payload(text: str = "catalogue-bytes") -> bytes:
    """Return an arbitrary payload."""
    return text.encode("utf-8")


def _digest(payload: bytes) -> str:
    """Return the hex SHA-256 of a payload."""
    return hashlib.sha256(payload).hexdigest()


def test_pinned_fetch_downloads_and_verifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cold cache downloads the file and records the verified digest."""
    payload = _payload()
    calls: list[str] = []

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        calls.append(request.full_url)
        return _MockResponse(payload, headers={"ETag": "abc"})

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)

    result = _fetch.resolve_digest_pinned_path(
        _URL,
        sha256=_digest(payload),
        size=len(payload),
        filename="18321_1yrCatalogBBH.h5",
        cache_dir=tmp_path,
    )

    assert calls == [_URL]
    assert result.path.read_bytes() == payload
    assert result.path.suffix == ".h5"
    assert result.metadata["cache_hit"] is False
    assert result.metadata["verified"] is True
    assert result.metadata["content_sha256"] == _digest(payload)

    # A second call with a matching cache does not touch the network.
    def fail_urlopen(request, *, timeout: int) -> _MockResponse:
        raise AssertionError("the network must not be reached for a matching cache")

    monkeypatch.setattr(_fetch, "urlopen", fail_urlopen)
    cached = _fetch.resolve_digest_pinned_path(
        _URL,
        sha256=_digest(payload),
        size=len(payload),
        filename="18321_1yrCatalogBBH.h5",
        cache_dir=tmp_path,
    )
    assert cached.metadata["cache_hit"] is True
    assert cached.path == result.path


def test_pinned_fetch_rejects_a_wrong_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bytes that do not match the pin are refused and never cached."""
    payload = _payload()

    monkeypatch.setattr(_fetch, "urlopen", lambda request, *, timeout: _MockResponse(payload))

    with pytest.raises(PopulationFetchError, match="SHA-256"):
        _fetch.resolve_digest_pinned_path(_URL, sha256="0" * 64, cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_pinned_fetch_rejects_a_wrong_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A size that does not match the pin is refused."""
    payload = _payload()

    monkeypatch.setattr(_fetch, "urlopen", lambda request, *, timeout: _MockResponse(payload))

    with pytest.raises(PopulationFetchError, match="bytes"):
        _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), size=1, cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_pinned_fetch_redownloads_a_corrupted_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cached file whose digest drifted is re-fetched rather than used."""
    payload = _payload()
    monkeypatch.setattr(_fetch, "urlopen", lambda request, *, timeout: _MockResponse(payload))
    first = _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path)

    first.path.write_bytes(_payload("corrupted"))

    recovered = _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path)

    assert recovered.path.read_bytes() == payload
    assert recovered.metadata["content_sha256"] == _digest(payload)


def test_pinned_fetch_refresh_forces_a_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Refreshing re-downloads even when the cache matches."""
    payload = _payload()
    calls: list[str] = []

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        calls.append(request.full_url)
        return _MockResponse(payload)

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)
    _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path)
    refreshed = _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path, refresh=True)

    assert len(calls) == 2
    assert refreshed.metadata["cache_hit"] is False


def test_pinned_fetch_offline_uses_only_a_matching_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Offline refuses to download, then serves a matching cache without the network."""
    payload = _payload()

    def offline_urlopen(request, *, timeout: int) -> _MockResponse:
        raise AssertionError("offline must not reach the network")

    monkeypatch.setattr(_fetch, "urlopen", offline_urlopen)
    with pytest.raises(PopulationFetchError, match="offline"):
        _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path, offline=True)

    monkeypatch.setattr(_fetch, "urlopen", lambda request, *, timeout: _MockResponse(payload))
    _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path)

    monkeypatch.setattr(_fetch, "urlopen", offline_urlopen)
    cached = _fetch.resolve_digest_pinned_path(_URL, sha256=_digest(payload), cache_dir=tmp_path, offline=True)
    assert cached.metadata["cache_hit"] is True


def test_pinned_fetch_rejects_a_local_path(tmp_path: Path) -> None:
    """A local path is not a pinned remote fetch."""
    with pytest.raises(PopulationFetchError, match="remote URL"):
        _fetch.resolve_digest_pinned_path(tmp_path / "local.h5", sha256="0" * 64)


def test_pinned_fetch_rejects_a_malformed_digest(tmp_path: Path) -> None:
    """A digest that is not a SHA-256 is refused before any download."""
    with pytest.raises(PopulationFetchError, match="hexadecimal"):
        _fetch.resolve_digest_pinned_path(_URL, sha256="not-a-digest", cache_dir=tmp_path)


def test_pinned_fetch_surfaces_http_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An HTTP failure is raised as a typed fetch error."""
    from urllib.error import HTTPError

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        raise HTTPError(request.full_url, 503, "Service Unavailable", hdrs=None, fp=None)

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)

    with pytest.raises(PopulationFetchError, match="HTTP 503"):
        _fetch.resolve_digest_pinned_path(_URL, sha256="0" * 64, cache_dir=tmp_path)
