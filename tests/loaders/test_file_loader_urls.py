"""Unit tests for remote ``FilePopulationLoader`` inputs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from gwmock_pop import FilePopulationLoader, PopulationFetchError
from gwmock_pop.loaders import _fetch


class _MockResponse(BytesIO):
    """Minimal file-like HTTP response for urlopen patches."""

    def __init__(self, payload: bytes, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(payload)
        self.headers = headers or {}

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.close()
        return False


def _catalogue_bytes(rows: tuple[tuple[float, float, float], ...] = ((30.0, 20.0, 0.1), (35.0, 25.0, 0.2))) -> bytes:
    """Encode a small CBC catalogue as CSV bytes."""
    csv_rows = ["m1,m2,z"]
    csv_rows.extend(f"{mass_1},{mass_2},{redshift}" for mass_1, mass_2, redshift in rows)
    return ("\n".join(csv_rows) + "\n").encode("utf-8")


def test_remote_loader_uses_cache_without_refetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cached HTTP catalogue is reused on subsequent loader construction."""
    calls: list[str] = []

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        calls.append(request.full_url)
        return _MockResponse(_catalogue_bytes(), headers={"ETag": "abc123"})

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)

    url = "https://example.com/population.csv"
    first_loader = FilePopulationLoader("bbh", url, cache_dir=tmp_path)
    second_loader = FilePopulationLoader("bbh", url, cache_dir=tmp_path)

    assert len(calls) == 1
    assert first_loader.metadata["fetch"]["cache_hit"] is False
    assert second_loader.metadata["fetch"]["cache_hit"] is True
    assert Path(first_loader.metadata["resolved_path"]).exists()


def test_remote_loader_refresh_forces_redownload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Refreshing a cached URL re-downloads and replaces the local cache entry."""
    payloads = iter(
        (
            _catalogue_bytes(((30.0, 20.0, 0.1),)),
            _catalogue_bytes(((40.0, 28.0, 0.3),)),
        )
    )
    calls = 0

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        nonlocal calls
        calls += 1
        return _MockResponse(next(payloads))

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)

    url = "https://example.com/population.csv"
    first_loader = FilePopulationLoader("bbh", url, cache_dir=tmp_path)
    refreshed_loader = FilePopulationLoader("bbh", url, cache_dir=tmp_path, refresh=True)

    assert calls == 2
    assert first_loader.metadata["fetch"]["cache_hit"] is False
    assert refreshed_loader.metadata["fetch"]["cache_hit"] is False
    assert Path(refreshed_loader.metadata["resolved_path"]).read_text(encoding="utf-8").startswith("m1,m2,z\n40.0")


def test_remote_loader_uses_environment_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remote fetches pick up bearer tokens from the documented environment variable."""
    seen_authorization_header: list[str | None] = []

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        seen_authorization_header.append(request.headers.get("Authorization"))
        return _MockResponse(_catalogue_bytes())

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)
    monkeypatch.setenv("GWMOCK_POP_TOKEN", "secret-token")

    FilePopulationLoader("bbh", "https://example.com/protected.csv", cache_dir=tmp_path)

    assert seen_authorization_header == ["Bearer secret-token"]


def test_remote_loader_surfaces_http_auth_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HTTP authentication failures are raised as typed fetch errors."""

    def fake_urlopen(request, *, timeout: int) -> _MockResponse:
        raise HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(_fetch, "urlopen", fake_urlopen)

    with pytest.raises(PopulationFetchError, match=r"HTTP 401 Unauthorized"):
        FilePopulationLoader("bbh", "https://example.com/protected.csv", cache_dir=tmp_path)
