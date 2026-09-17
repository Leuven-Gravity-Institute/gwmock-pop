"""Tests for the CLI catalogue draw command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gwmock_pop.catalogue import population as population_module
from gwmock_pop.catalogue.population import SECONDS_PER_YEAR
from gwmock_pop.cli import catalogue as catalogue_module
from gwmock_pop.cli.main import app
from gwmock_pop.provenance import read_provenance
from tests.catalogue._fixtures import make_pair

_RUNNER = CliRunner()


@pytest.fixture
def offline_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Patch the catalogue source to a local pair and return the pair paths.

    The draw and its anchor read the same resolved files, so patching the fetch
    is enough for the whole command to run offline.

    Args:
        monkeypatch: pytest patch fixture.
        tmp_path: Temporary directory.

    Returns:
        The resolved BNS and BBH catalogue paths.
    """
    bns, bbh = make_pair(tmp_path)
    monkeypatch.setattr(catalogue_module, "COBA_CATALOGUES", (bns.catalogue, bbh.catalogue))
    monkeypatch.setattr(population_module, "fetch_catalogues", lambda catalogues, **kwargs: (bns, bbh))
    return bns.path, bbh.path


def test_catalogue_command_writes_a_stamped_draw(offline_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """The command writes a provenance-carrying draw and a composition stamp."""
    output = tmp_path / "draw.h5"
    stamp = tmp_path / "stamp.json"

    result = _RUNNER.invoke(
        app,
        [
            "catalogue",
            "--seed",
            "1",
            "--draws",
            "2",
            "--span-seconds",
            str(SECONDS_PER_YEAR * 3.0),
            "--output",
            str(output),
            "--stamp",
            str(stamp),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert stamp.exists()
    assert "BNS" in result.output

    provenance = read_provenance(output)
    assert provenance is not None
    assert provenance["origin"]["kind"] == "catalogue_draw"
    assert provenance["origin"]["files"][0]["sha256"] == "0" * 64
    assert provenance["run"]["seed"] == 1

    payload = json.loads(stamp.read_text(encoding="utf-8"))
    assert payload["draw"]["n_rows"] > 0
    assert payload["draw"]["seed"] == 1
    assert payload["composition"]["n_draws"] == 2
    assert payload["catalogue_anchor"]["band_detector_frame_chirp_mass"] == 5.0


def test_catalogue_command_refuses_to_overwrite(offline_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """An existing output is not replaced without --overwrite."""
    output = tmp_path / "draw.h5"
    output.write_text("existing", encoding="utf-8")

    result = _RUNNER.invoke(
        app,
        ["catalogue", "--span-seconds", "1", "--output", str(output)],
    )

    assert result.exit_code == 1
    assert output.read_text(encoding="utf-8") == "existing"


def test_catalogue_command_stamp_requires_output(offline_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """A stamp without an output cannot bind a digest and is refused."""
    result = _RUNNER.invoke(
        app,
        ["catalogue", "--span-seconds", "1", "--stamp", str(tmp_path / "stamp.json")],
    )

    assert result.exit_code == 1


def test_catalogue_command_rejects_an_unsupported_output_suffix(
    offline_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    """An output the catalogue writer cannot produce is refused before any draw."""
    result = _RUNNER.invoke(
        app,
        ["catalogue", "--span-seconds", "1", "--output", str(tmp_path / "draw.txt")],
    )

    assert result.exit_code == 1
