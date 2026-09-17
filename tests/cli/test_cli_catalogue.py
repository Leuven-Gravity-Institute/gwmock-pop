"""Tests for the CLI catalogue draw command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner, Result

from gwmock_pop.catalogue import population as population_module
from gwmock_pop.catalogue.population import SECONDS_PER_YEAR
from gwmock_pop.cli import catalogue as catalogue_module
from gwmock_pop.cli.main import app
from gwmock_pop.provenance import read_provenance
from tests.catalogue._fixtures import make_pair

_RUNNER = CliRunner()


def _offline_pair_from(pair: tuple, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Patch the catalogue source to a local pair and return the pair paths.

    Args:
        pair: The resolved BNS/BBH pair to patch in.
        monkeypatch: pytest patch fixture.

    Returns:
        The resolved BNS and BBH catalogue paths.
    """
    bns, bbh = pair
    monkeypatch.setattr(catalogue_module, "COBA_CATALOGUES", (bns.catalogue, bbh.catalogue))
    monkeypatch.setattr(population_module, "fetch_catalogues", lambda catalogues, **kwargs: (bns, bbh))
    return bns.path, bbh.path


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
    return _offline_pair_from(make_pair(tmp_path), monkeypatch)


@pytest.fixture
def heavy_offline_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Patch the source to a pair whose black-hole catalogue reaches the IMBH cut.

    Args:
        monkeypatch: pytest patch fixture.
        tmp_path: Temporary directory.

    Returns:
        The resolved BNS and BBH catalogue paths.
    """
    pair = make_pair(
        tmp_path,
        bbh_mass_1=(10.0, 60.0, 60.0),
        bbh_mass_2=(10.0, 60.0, 60.0),
        bbh_redshift=(0.0, 0.0, 0.0),
    )
    return _offline_pair_from(pair, monkeypatch)


def _run_catalogue(output: Path, stamp: Path, seed: int = 1, draws: int = 2) -> Result:
    """Run the catalogue CLI into a draw and a stamp.

    Args:
        output: Destination draw file.
        stamp: Destination stamp file.
        seed: Seed of the archived draw.
        draws: Number of draws whose spread is reported.

    Returns:
        The CLI result.
    """
    return _RUNNER.invoke(
        app,
        [
            "catalogue",
            "--seed",
            str(seed),
            "--draws",
            str(draws),
            "--span-seconds",
            str(SECONDS_PER_YEAR * 3.0),
            "--output",
            str(output),
            "--stamp",
            str(stamp),
        ],
    )


def test_catalogue_command_writes_a_stamped_draw(offline_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """The command writes a provenance-carrying draw and a composition stamp."""
    output = tmp_path / "draw.h5"
    stamp = tmp_path / "stamp.json"

    result = _run_catalogue(output=output, stamp=stamp)

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert stamp.exists()
    assert "BNS" in result.output

    provenance = read_provenance(output)
    assert provenance is not None
    assert provenance["origin"]["kind"] == "catalogue_draw"
    assert provenance["origin"]["files"][0]["sha256"] == "0" * 64
    assert provenance["run"]["seed"] == 1
    citations = {entry["role"]: entry["reference"] for entry in provenance["origin"]["catalogue"]["citations"]}
    assert set(citations) == {"catalogue_source_paper", "science_reference"}
    assert "2303.15923" in citations["catalogue_source_paper"]
    assert "2503.12263" in citations["science_reference"]

    payload = json.loads(stamp.read_text(encoding="utf-8"))
    assert payload["draw"]["n_rows"] > 0
    assert payload["draw"]["seed"] == 1
    assert payload["composition"]["n_draws"] == 2
    assert payload["catalogue_anchor"]["band_detector_frame_chirp_mass"] == 5.0


def test_catalogue_command_regeneration_is_data_deterministic(offline_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """A rerun reproduces the draw data and rebinds the stamp to its own digest."""
    first_output = tmp_path / "first.h5"
    first_stamp = tmp_path / "first.json"
    second_output = tmp_path / "second.h5"
    second_stamp = tmp_path / "second.json"

    assert _run_catalogue(output=first_output, stamp=first_stamp).exit_code == 0
    assert _run_catalogue(output=second_output, stamp=second_stamp).exit_code == 0

    with h5py.File(first_output, "r") as first, h5py.File(second_output, "r") as second:
        assert first["data"].dtype.names == second["data"].dtype.names
        for name in first["data"].dtype.names:
            assert np.array_equal(first["data"][name], second["data"][name]), name

    payload = json.loads(second_stamp.read_text(encoding="utf-8"))
    digest = hashlib.sha256(second_output.read_bytes()).hexdigest()
    assert payload["draw"]["sha256"] == digest


def test_catalogue_command_black_hole_rates_partition(heavy_offline_pair: tuple[Path, Path], tmp_path: Path) -> None:
    """The printed BBH and IMBH rates partition the black-hole catalogue rate."""
    output = tmp_path / "draw.h5"
    stamp = tmp_path / "stamp.json"

    result = _run_catalogue(output=output, stamp=stamp)
    assert result.exit_code == 0, result.output

    payload = json.loads(stamp.read_text(encoding="utf-8"))
    rates = payload["composition"]["drawn_rate_per_year"]
    anchor = {entry["name"]: entry["drawn"] for entry in payload["catalogue_anchor"]["class_counts"]}
    black_hole_rows = anchor.get("BBH", 0) + anchor.get("IMBH", 0)
    assert black_hole_rows > 0
    assert rates["BBH"] + rates["IMBH"] == pytest.approx(black_hole_rows)
    assert rates["IMBH"] > 0.0


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


def _record_draw_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the draw with a recorder that also fails if it is reached.

    Args:
        monkeypatch: pytest patch fixture.

    Returns:
        The list the recorder appends to on every call.
    """
    calls: list[int] = []

    def record(*args: object, **kwargs: object) -> None:
        calls.append(1)
        raise AssertionError("the draw must not run when the persistence options are invalid")

    monkeypatch.setattr(catalogue_module, "draw_catalogue_population_many", record)
    return calls


def test_catalogue_command_validates_the_stamp_before_the_draw(
    offline_pair: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stamp without an output is refused without drawing the population."""
    calls = _record_draw_calls(monkeypatch)

    result = _RUNNER.invoke(
        app,
        ["catalogue", "--span-seconds", "1", "--stamp", str(tmp_path / "stamp.json")],
    )

    assert result.exit_code == 1
    assert calls == []


def test_catalogue_command_rejects_output_and_stamp_on_the_same_path(
    offline_pair: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An output and a stamp naming the same file are refused before the draw."""
    calls = _record_draw_calls(monkeypatch)
    target = tmp_path / "draw.h5"

    result = _RUNNER.invoke(
        app,
        ["catalogue", "--span-seconds", "1", "--output", str(target), "--stamp", str(target)],
    )

    assert result.exit_code == 1
    assert calls == []
    assert not target.exists()


def test_catalogue_command_rejects_an_unsupported_output_suffix(
    offline_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    """An output the catalogue writer cannot produce is refused before any draw."""
    result = _RUNNER.invoke(
        app,
        ["catalogue", "--span-seconds", "1", "--output", str(tmp_path / "draw.txt")],
    )

    assert result.exit_code == 1
