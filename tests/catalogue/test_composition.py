"""Tests for the band-composition measurement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_pop.catalogue.composition import (
    catalogue_band_anchor,
    composition_summary,
    wilson_lower_bound,
)
from gwmock_pop.catalogue.population import PARAMETER_NAMES, CatalogueDraw
from tests.catalogue._fixtures import resolved_catalogue, write_catalogue


def _synthetic_draw(  # noqa: PLR0913  # every field the test sets is a knob
    *,
    classes: list[str],
    in_band: list[bool],
    chirp_masses: list[float],
    seed: int = 0,
    band: float = 5.0,
    threshold: float = 100.0,
) -> CatalogueDraw:
    """Build a draw with explicit classes and band membership.

    Args:
        classes: Class label per row.
        in_band: Band membership per row.
        chirp_masses: Detector-frame chirp mass per row.
        seed: The draw's seed.
        band: The band edge.
        threshold: The intermediate-mass threshold.

    Returns:
        A draw carrying the given rows.
    """
    n_rows = len(classes)
    parameters = {name: np.zeros(n_rows, dtype=float) for name in PARAMETER_NAMES}
    parameters["redshift"] = np.zeros(n_rows, dtype=float)
    label_array = np.asarray(classes, dtype=object)
    return CatalogueDraw(
        seed=seed,
        span_seconds=1.0,
        rate_multiplier=1.0,
        band_detector_frame_chirp_mass=band,
        imbh_total_mass_threshold=threshold,
        parameters=parameters,
        source_class=label_array,
        population_class=label_array,
        detector_frame_chirp_mass=np.asarray(chirp_masses, dtype=float),
        source_frame_chirp_mass=np.asarray(chirp_masses, dtype=float),
        in_band=np.asarray(in_band, dtype=bool),
        expected_per_class={"BNS": 100.0, "BBH": 50.0},
        drawn_per_class=dict.fromkeys(classes, 1),
        resolved=(),
    )


def test_wilson_lower_bound_behaves_at_the_edges() -> None:
    """The bound is zero over no data and rises toward the observed share."""
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(0, 100) < 1e-12
    assert 0.9 < wilson_lower_bound(100, 100) < 1.0
    assert 0.2 < wilson_lower_bound(3, 6) < 0.3


def test_composition_pools_counts_and_reports_the_spread() -> None:
    """Pooled counts, shares and the draw-to-draw spread are all measured."""
    first = _synthetic_draw(
        classes=["BNS", "BNS", "BBH", "IMBH"], in_band=[True, True, False, True], chirp_masses=[6, 7, 4, 200]
    )
    second = _synthetic_draw(
        classes=["BNS", "BBH", "BBH", "IMBH"], in_band=[True, True, True, False], chirp_masses=[6, 10, 20, 200], seed=1
    )

    composition = composition_summary([first, second])

    counts = {entry.name: entry for entry in composition.class_counts}
    assert counts["BNS"].drawn == 3
    assert counts["BNS"].in_band == 3
    assert counts["BNS"].in_band_fraction == pytest.approx(1.0)
    assert counts["BNS"].share_of_band == pytest.approx(0.5)
    assert counts["BBH"].share_of_band == pytest.approx(2 / 6)
    assert counts["IMBH"].share_of_band == pytest.approx(1 / 6)
    assert composition.share_of_band_mean["BNS"] == pytest.approx(0.5)
    assert composition.share_of_band_std["BNS"] == pytest.approx(1 / 6)
    assert composition.share_of_band_min["BNS"] == pytest.approx(1 / 3)
    assert composition.share_of_band_max["BNS"] == pytest.approx(2 / 3)
    assert len(composition.share_of_band_per_draw["BNS"]) == 2
    assert composition.bns_share_of_band_lower_90 == pytest.approx(0.2682, abs=1e-3)
    assert composition.seeds == (0, 1)


def test_composition_ranges_and_duration_use_the_first_draw() -> None:
    """Chirp-mass ranges and duration quantiles are reported for the archived draw."""
    draw = _synthetic_draw(
        classes=["BNS", "BNS", "BBH"],
        in_band=[True, True, False],
        chirp_masses=[6.0, 7.0, 4.0],
    )
    composition = composition_summary([draw])

    assert composition.detector_frame_chirp_mass_range["BNS"] == (6.0, 7.0)
    assert composition.detector_frame_chirp_mass_range["BBH"] is None
    assert set(composition.in_band_duration_quantiles) == {"p10", "p50", "p90"}
    assert composition.in_band_duration_quantiles["p10"] > 0.0


def test_composition_refuses_no_draws() -> None:
    """An empty draw set cannot measure a composition."""
    with pytest.raises(ValueError, match="At least one draw"):
        composition_summary([])


def test_composition_refuses_mismatched_bands() -> None:
    """Draws measured at different band edges are not pooled."""
    first = _synthetic_draw(classes=["BNS"], in_band=[True], chirp_masses=[6.0], band=5.0)
    second = _synthetic_draw(classes=["BNS"], in_band=[True], chirp_masses=[6.0], band=4.0)
    with pytest.raises(ValueError, match="band edge"):
        composition_summary([first, second])


def test_catalogue_band_anchor_counts_every_catalogue_row(tmp_path: Path) -> None:
    """The anchor counts the full catalogues exactly, in both frames' classes."""
    bns_path = tmp_path / "bns.h5"
    bbh_path = tmp_path / "bbh.h5"
    write_catalogue(
        bns_path,
        mass_1=np.asarray([1.4, 1.4, 2.0]),
        mass_2=np.asarray([1.4, 1.4, 2.0]),
        redshift=np.asarray([0.0, 5.0, 3.0]),
    )
    write_catalogue(
        bbh_path,
        mass_1=np.asarray([10.0, 60.0, 60.0]),
        mass_2=np.asarray([10.0, 60.0, 60.0]),
        redshift=np.asarray([0.0, 0.0, 0.1]),
    )
    resolved = (
        resolved_catalogue(bns_path, name="bns", source_class="BNS"),
        resolved_catalogue(bbh_path, name="bbh", source_class="BBH"),
    )

    anchor = catalogue_band_anchor(
        resolved,
        band_detector_frame_chirp_mass=5.0,
        imbh_total_mass_threshold=100.0,
    )

    counts = {entry.name: entry for entry in anchor.class_counts}
    assert counts["BNS"].drawn == 3
    assert counts["BNS"].in_band == 2
    assert counts["BNS"].share_of_band == pytest.approx(0.4)
    assert counts["BBH"].drawn == 1
    assert counts["BBH"].in_band == 1
    assert counts["IMBH"].drawn == 2
    assert counts["IMBH"].in_band == 2
    assert counts["IMBH"].share_of_band == pytest.approx(0.4)
    assert anchor.redshift_reach == pytest.approx(5.0)
