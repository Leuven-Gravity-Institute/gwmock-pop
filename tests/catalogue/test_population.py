"""Tests for the seeded catalogue draw."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gwmock_pop.catalogue.manifest import CatalogueFile
from gwmock_pop.catalogue.population import (
    PARAMETER_NAMES,
    SECONDS_PER_YEAR,
    chirp_mass,
    draw_catalogue_population,
    inband_duration_seconds,
    read_catalogue_columns,
)
from gwmock_pop.exceptions import PopulationValidationError
from tests.catalogue._fixtures import make_pair, resolved_catalogue, write_catalogue


def test_draw_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    """The same seed, span and files reproduce the draw exactly."""
    bns, bbh = make_pair(tmp_path)
    settings = {"seed": 11, "span_seconds": SECONDS_PER_YEAR * 3.0, "resolved": (bns, bbh)}

    first = draw_catalogue_population(**settings)
    second = draw_catalogue_population(**settings)

    assert first.n_rows > 0
    for name in PARAMETER_NAMES:
        assert np.array_equal(first.parameters[name], second.parameters[name]), name
    assert np.array_equal(first.population_class, second.population_class)
    assert np.array_equal(first.in_band, second.in_band)


def test_draw_changes_with_the_seed(tmp_path: Path) -> None:
    """A different seed draws a different realisation."""
    bns, bbh = make_pair(tmp_path)
    common = {"span_seconds": SECONDS_PER_YEAR * 3.0, "resolved": (bns, bbh)}

    first = draw_catalogue_population(seed=1, **common)
    second = draw_catalogue_population(seed=2, **common)

    assert first.n_rows > 0
    differs = any(not np.array_equal(first.parameters[name], second.parameters[name]) for name in PARAMETER_NAMES)
    assert differs


def test_draw_converts_units_and_frames(tmp_path: Path) -> None:
    """Distances are converted to Mpc and masses are boosted to the detector frame."""
    bns, bbh = make_pair(tmp_path)
    draw = draw_catalogue_population(seed=5, span_seconds=SECONDS_PER_YEAR * 3.0, resolved=(bns, bbh))

    source_mass_1 = draw.parameters["detector_frame_mass_1"] / (1.0 + draw.parameters["redshift"])
    source_mass_2 = draw.parameters["detector_frame_mass_2"] / (1.0 + draw.parameters["redshift"])
    assert np.all(source_mass_1 >= 1.2)
    assert np.all(source_mass_1 <= 50.0)
    assert np.all(source_mass_2 >= 1.1)
    assert np.all(source_mass_2 <= 40.0)
    assert np.all(draw.parameters["distance"] >= 0.05 * 1000.0)
    assert np.all(draw.parameters["distance"] <= 150.0 * 1000.0)
    assert np.all(draw.parameters["coa_time"] >= 0.0)
    assert np.all(draw.parameters["coa_time"] <= SECONDS_PER_YEAR * 3.0)


def test_band_cut_uses_the_detector_frame_chirp_mass(tmp_path: Path) -> None:
    """The band membership is exactly the comparison on the detector-frame chirp mass."""
    bns, bbh = make_pair(tmp_path)
    common = {"seed": 9, "span_seconds": SECONDS_PER_YEAR, "resolved": (bns, bbh)}

    unbounded = draw_catalogue_population(band_detector_frame_chirp_mass=1.0e9, **common)
    open_band = draw_catalogue_population(band_detector_frame_chirp_mass=0.1, **common)

    assert unbounded.n_rows > 0
    assert open_band.n_rows > 0
    assert not np.any(unbounded.in_band)
    assert np.all(open_band.in_band)
    expected = open_band.detector_frame_chirp_mass > 0.1
    assert np.array_equal(open_band.in_band, expected)


def test_classes_are_read_from_the_catalogues_and_the_heavy_tail_is_labelled(tmp_path: Path) -> None:
    """Neutron stars stay neutron stars; only heavy black holes become intermediate-mass."""
    bns, bbh = make_pair(
        tmp_path,
        bbh_mass_1=(6.0, 60.0, 80.0),
        bbh_mass_2=(3.0, 60.0, 80.0),
        bbh_redshift=(0.0, 0.0, 0.0),
    )
    draw = draw_catalogue_population(seed=3, span_seconds=SECONDS_PER_YEAR * 20.0, resolved=(bns, bbh))

    assert set(draw.source_class.tolist()) == {"BNS", "BBH"}
    assert set(draw.population_class.tolist()) == {"BNS", "BBH", "IMBH"}
    assert not np.any((draw.population_class == "IMBH") & (draw.source_class != "BBH"))
    heavy = draw.parameters["detector_frame_mass_1"] + draw.parameters["detector_frame_mass_2"] >= 100.0
    expected_imbh = (draw.source_class == "BBH") & heavy
    assert np.array_equal(draw.population_class == "IMBH", expected_imbh)


def test_expected_counts_follow_rate_and_span(tmp_path: Path) -> None:
    """The per-class Poisson mean is the catalogue row count times span over year."""
    bns, bbh = make_pair(tmp_path)
    span = SECONDS_PER_YEAR / 4.0
    draw = draw_catalogue_population(seed=0, span_seconds=span, resolved=(bns, bbh), rate_multiplier=2.0)

    assert draw.expected_per_class["BNS"] == pytest.approx(3 * (span / SECONDS_PER_YEAR) * 2.0)
    assert draw.expected_per_class["BBH"] == pytest.approx(3 * (span / SECONDS_PER_YEAR) * 2.0)
    assert sum(draw.drawn_per_class.values()) == draw.n_rows


def test_zero_span_draws_nothing(tmp_path: Path) -> None:
    """A zero span draws no source and still returns every column."""
    bns, bbh = make_pair(tmp_path)
    draw = draw_catalogue_population(seed=0, span_seconds=0.0, resolved=(bns, bbh))

    assert draw.n_rows == 0
    assert draw.in_band_rows == 0
    assert set(draw.parameters) == set(PARAMETER_NAMES)
    for values in draw.parameters.values():
        assert values.shape == (0,)
    assert draw.population_class.shape == (0,)


@pytest.mark.parametrize(
    "settings",
    [
        {"span_seconds": -1.0},
        {"span_seconds": 1.0, "rate_multiplier": -1.0},
        {"span_seconds": 1.0, "band_detector_frame_chirp_mass": 0.0},
        {"span_seconds": 1.0, "imbh_total_mass_threshold": 0.0},
    ],
)
def test_invalid_settings_are_refused(tmp_path: Path, settings: dict[str, float]) -> None:
    """Settings that cannot describe a population are refused."""
    bns, bbh = make_pair(tmp_path)
    with pytest.raises(PopulationValidationError):
        draw_catalogue_population(seed=0, resolved=(bns, bbh), **settings)


def test_read_catalogue_columns_rejects_a_missing_column(tmp_path: Path) -> None:
    """A catalogue without a required column is refused with the column named."""
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as handle:
        handle["m1_source"] = np.asarray([1.0, 2.0])
        handle["m2_source"] = np.asarray([1.0, 2.0])

    with pytest.raises(PopulationValidationError, match="z"):
        read_catalogue_columns(path)


def test_read_catalogue_columns_rejects_mismatched_lengths(tmp_path: Path) -> None:
    """Columns of different lengths are refused rather than broadcast."""
    path = tmp_path / "ragged.h5"
    with h5py.File(path, "w") as handle:
        handle["m1_source"] = np.asarray([1.0, 2.0])
        handle["z"] = np.asarray([0.1])

    with pytest.raises(PopulationValidationError, match="lengths"):
        read_catalogue_columns(path, columns=("m1_source", "z"))


def test_chirp_mass_matches_the_equal_mass_identity() -> None:
    """An equal-mass binary's chirp mass is ``m * 2 ** (-1/5)``."""
    mass = np.asarray([10.0, 10.0])
    assert chirp_mass(mass, mass) == pytest.approx(10.0 * 2.0 ** (-0.2))


def test_inband_duration_shrinks_with_chirp_mass() -> None:
    """A heavier binary spends less time in band at a fixed low-frequency cutoff."""
    light = inband_duration_seconds(np.asarray([1.22]), 10.0)
    heavy = inband_duration_seconds(np.asarray([30.0]), 10.0)

    assert np.all(np.isfinite(light))
    assert 100.0 < light[0] < 10_000.0
    assert heavy[0] < light[0]


def test_draw_uses_a_supplied_resolved_catalogue_without_fetching(tmp_path: Path) -> None:
    """A caller-supplied resolved catalogue is read directly, with no fetch."""
    path = tmp_path / "bbh.h5"
    write_catalogue(
        path,
        mass_1=np.asarray([10.0, 12.0]),
        mass_2=np.asarray([8.0, 9.0]),
        redshift=np.asarray([0.0, 0.0]),
    )
    resolved = resolved_catalogue(path, name="bbh", source_class="BBH")
    draw = draw_catalogue_population(
        seed=0,
        span_seconds=SECONDS_PER_YEAR,
        catalogues=(resolved.catalogue,),
        resolved=(resolved,),
    )

    assert draw.n_rows > 0
    assert set(draw.source_class.tolist()) == {"BBH"}


def _overwrite_column_with_nan(path: Path, column: str) -> None:
    """Set every value of one catalogue column to NaN in place.

    Args:
        path: Catalogue file to edit.
        column: Column to poison.
    """
    with h5py.File(path, "a") as handle:
        values = np.asarray(handle[column], dtype=float)
        values[:] = np.nan
        handle[column][...] = values


@pytest.mark.parametrize("column", ["chi1z", "chi2z", "dL", "ra", "dec", "psi", "iota", "Phicoal"])
def test_draw_rejects_non_finite_catalogue_columns(tmp_path: Path, column: str) -> None:
    """A non-finite value in any emitted catalogue column is refused."""
    path = tmp_path / "bns.h5"
    write_catalogue(
        path,
        mass_1=np.asarray([1.2, 1.4, 1.6]),
        mass_2=np.asarray([1.1, 1.3, 1.5]),
        redshift=np.asarray([0.05, 0.1, 0.2]),
    )
    _overwrite_column_with_nan(path, column)
    resolved = resolved_catalogue(path, name="bns", source_class="BNS")

    with pytest.raises(PopulationValidationError, match=column):
        draw_catalogue_population(
            seed=0,
            span_seconds=SECONDS_PER_YEAR * 100.0,
            resolved=(resolved,),
        )


def test_draw_rejects_resolved_catalogues_that_do_not_match_the_configured_pins(tmp_path: Path) -> None:
    """A resolved file whose pinned identity differs from the configured one is refused."""
    bns, _ = make_pair(tmp_path)
    mismatched = CatalogueFile(
        name="other",
        source_class="BNS",
        filename="other.h5",
        sha256="1" * 64,
        size_bytes=1,
        n_rows=1,
    )

    with pytest.raises(PopulationValidationError, match="match"):
        draw_catalogue_population(
            seed=0,
            span_seconds=SECONDS_PER_YEAR,
            catalogues=(mismatched,),
            resolved=(bns,),
        )
