"""Unit tests for CBC population validation and canonicalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_pop import FilePopulationLoader, PopulationValidationError


def _write_csv_catalogue(path: Path, rows: list[dict[str, float]]) -> None:
    """Write a small header-based CSV catalogue."""
    column_names = list(rows[0])
    matrix = np.column_stack([[row[name] for row in rows] for name in column_names])
    np.savetxt(path, matrix, delimiter=",", header=",".join(column_names), comments="")


def test_loader_canonicalizes_cbc_aliases_and_derives_detector_masses(tmp_path: Path) -> None:
    """CBC aliases are mapped and detector-frame masses are derived before sampling."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(
        path,
        [
            {"m1_source": 24.0, "m2_source": 19.0, "z": 0.05, "tc": 50.0},
            {"m1_source": 28.0, "m2_source": 23.0, "z": 0.10, "tc": 100.0},
        ],
    )

    loader = FilePopulationLoader("bbh", path)
    result = loader.simulate(2, seed=0)

    assert "detector_frame_mass_1" in loader.parameter_names
    assert "detector_frame_mass_2" in loader.parameter_names
    assert "coa_time" in loader.parameter_names
    assert "source_frame_mass_1" not in loader.parameter_names
    np.testing.assert_allclose(
        np.sort(np.asarray(result["detector_frame_mass_1"])),
        np.array([24.0 * 1.05, 28.0 * 1.10]),
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(result["detector_frame_mass_2"])),
        np.array([19.0 * 1.05, 23.0 * 1.10]),
    )


def test_loader_rejects_unknown_cbc_columns_with_context(tmp_path: Path) -> None:
    """Unknown CBC column names fail with a message that names the offending column."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(
        path,
        [
            {"detector_frame_mass_1": 30.0, "detector_frame_mass_2": 20.0, "mystery_mass": 5.0},
        ],
    )

    with pytest.raises(PopulationValidationError, match="mystery_mass"):
        FilePopulationLoader("bbh", path)


def test_loader_rejects_missing_mass_inputs_with_context(tmp_path: Path) -> None:
    """Missing mass inputs fail with an actionable validation error."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(
        path,
        [
            {"z": 0.1, "tc": 100.0},
        ],
    )

    with pytest.raises(PopulationValidationError, match="detector_frame_mass_1"):
        FilePopulationLoader("bbh", path)


def test_loader_rejects_non_finite_values_with_context(tmp_path: Path) -> None:
    """NaN and inf values fail before the loader can sample from the catalogue."""
    path = tmp_path / "catalogue.csv"
    _write_csv_catalogue(
        path,
        [
            {"detector_frame_mass_1": np.nan, "detector_frame_mass_2": 20.0},
        ],
    )

    with pytest.raises(PopulationValidationError, match="detector_frame_mass_1"):
        FilePopulationLoader("bbh", path)
