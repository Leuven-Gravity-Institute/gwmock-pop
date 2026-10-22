"""Tests for the seeded catalogue draw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import h5py
except ModuleNotFoundError as exc:  # Python 3.15: h5py publishes no wheel yet; its tests are skipped
    if exc.name != "h5py":
        raise
    h5py = None

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

# The fixture catalogues hold a few rows, fewer than most spans here draw, so the
# tests of other properties permit sampling with replacement. The refusal, and the
# draws it leaves unchanged, are tested on their own below.


@pytest.mark.requires_h5py
def test_draw_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    """The same seed, span and files reproduce the draw exactly."""
    bns, bbh = make_pair(tmp_path)
    settings = {"seed": 11, "span_seconds": SECONDS_PER_YEAR * 3.0, "resolved": (bns, bbh)}

    first = draw_catalogue_population(allow_replacement=True, **settings)
    second = draw_catalogue_population(allow_replacement=True, **settings)

    assert first.n_rows > 0
    for name in PARAMETER_NAMES:
        assert np.array_equal(first.parameters[name], second.parameters[name]), name
    assert np.array_equal(first.population_class, second.population_class)
    assert np.array_equal(first.in_band, second.in_band)


@pytest.mark.requires_h5py
def test_draw_changes_with_the_seed(tmp_path: Path) -> None:
    """A different seed draws a different realisation."""
    bns, bbh = make_pair(tmp_path)
    common = {"span_seconds": SECONDS_PER_YEAR * 3.0, "resolved": (bns, bbh)}

    first = draw_catalogue_population(allow_replacement=True, seed=1, **common)
    second = draw_catalogue_population(allow_replacement=True, seed=2, **common)

    assert first.n_rows > 0
    differs = any(not np.array_equal(first.parameters[name], second.parameters[name]) for name in PARAMETER_NAMES)
    assert differs


@pytest.mark.requires_h5py
def test_draw_converts_units_and_frames(tmp_path: Path) -> None:
    """Distances are converted to Mpc and masses are boosted to the detector frame."""
    bns, bbh = make_pair(tmp_path)
    draw = draw_catalogue_population(
        allow_replacement=True, seed=5, span_seconds=SECONDS_PER_YEAR * 3.0, resolved=(bns, bbh)
    )

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


@pytest.mark.requires_h5py
def test_band_cut_uses_the_detector_frame_chirp_mass(tmp_path: Path) -> None:
    """The band membership is exactly the comparison on the detector-frame chirp mass."""
    bns, bbh = make_pair(tmp_path)
    common = {"seed": 9, "span_seconds": SECONDS_PER_YEAR, "resolved": (bns, bbh)}

    unbounded = draw_catalogue_population(allow_replacement=True, band_detector_frame_chirp_mass=1.0e9, **common)
    open_band = draw_catalogue_population(allow_replacement=True, band_detector_frame_chirp_mass=0.1, **common)

    assert unbounded.n_rows > 0
    assert open_band.n_rows > 0
    assert not np.any(unbounded.in_band)
    assert np.all(open_band.in_band)
    expected = open_band.detector_frame_chirp_mass > 0.1
    assert np.array_equal(open_band.in_band, expected)


@pytest.mark.requires_h5py
def test_classes_are_read_from_the_catalogues_and_the_heavy_tail_is_labelled(tmp_path: Path) -> None:
    """Neutron stars stay neutron stars; only heavy black holes become intermediate-mass."""
    bns, bbh = make_pair(
        tmp_path,
        bbh_mass_1=(6.0, 60.0, 80.0),
        bbh_mass_2=(3.0, 60.0, 80.0),
        bbh_redshift=(0.0, 0.0, 0.0),
    )
    draw = draw_catalogue_population(
        allow_replacement=True, seed=3, span_seconds=SECONDS_PER_YEAR * 20.0, resolved=(bns, bbh)
    )

    assert set(draw.source_class.tolist()) == {"BNS", "BBH"}
    assert set(draw.population_class.tolist()) == {"BNS", "BBH", "IMBH"}
    assert not np.any((draw.population_class == "IMBH") & (draw.source_class != "BBH"))
    heavy = draw.parameters["detector_frame_mass_1"] + draw.parameters["detector_frame_mass_2"] >= 100.0
    expected_imbh = (draw.source_class == "BBH") & heavy
    assert np.array_equal(draw.population_class == "IMBH", expected_imbh)


@pytest.mark.requires_h5py
def test_expected_counts_follow_rate_and_span(tmp_path: Path) -> None:
    """The per-class Poisson mean is the catalogue row count times span over year."""
    bns, bbh = make_pair(tmp_path)
    span = SECONDS_PER_YEAR / 4.0
    draw = draw_catalogue_population(
        allow_replacement=True, seed=0, span_seconds=span, resolved=(bns, bbh), rate_multiplier=2.0
    )

    assert draw.expected_per_class["BNS"] == pytest.approx(3 * (span / SECONDS_PER_YEAR) * 2.0)
    assert draw.expected_per_class["BBH"] == pytest.approx(3 * (span / SECONDS_PER_YEAR) * 2.0)
    assert sum(draw.drawn_per_class.values()) == draw.n_rows


@pytest.mark.requires_h5py
def test_expected_count_of_a_class_sums_over_its_catalogues(tmp_path: Path) -> None:
    """Two catalogues of one class report the summed mean, as their counts are summed."""
    first = tmp_path / "bbh_a.h5"
    second = tmp_path / "bbh_b.h5"
    write_catalogue(
        first, mass_1=np.asarray([30.0, 40.0]), mass_2=np.asarray([20.0, 30.0]), redshift=np.asarray([0.5, 1.0])
    )
    write_catalogue(
        second,
        mass_1=np.asarray([30.0, 40.0, 50.0, 60.0]),
        mass_2=np.asarray([20.0, 30.0, 40.0, 50.0]),
        redshift=np.asarray([0.5, 1.0, 1.5, 2.0]),
    )
    resolved = (
        resolved_catalogue(first, name="bbh_a", source_class="BBH"),
        resolved_catalogue(second, name="bbh_b", source_class="BBH"),
    )
    span = SECONDS_PER_YEAR / 4.0

    draw = draw_catalogue_population(seed=0, span_seconds=span, resolved=resolved)

    assert draw.expected_per_class == {"BBH": pytest.approx((2 + 4) * span / SECONDS_PER_YEAR)}
    assert draw.drawn_per_class["BBH"] == draw.n_rows


def _reference_draw(
    resolved: tuple, *, seed: int, span_seconds: float, rate_multiplier: float = 1.0
) -> dict[str, np.ndarray]:
    """Redraw rows the documented way, independently of the package's draw.

    One generator is seeded and shared across catalogues in order; each draws a
    Poisson count, then ``choice`` of that many row indices (with replacement only
    when the count exceeds the rows), then uniform merger times over the span.

    Args:
        resolved: The resolved catalogues, in draw order.
        seed: Seed of the generator.
        span_seconds: Observation span in seconds.
        rate_multiplier: Factor applied to every catalogue's annual rate.

    Returns:
        The redrawn distance, redshift, primary detector-frame mass and merger time.
    """
    rng = np.random.default_rng(seed)
    blocks: dict[str, list[np.ndarray]] = {"distance": [], "redshift": [], "detector_frame_mass_1": [], "coa_time": []}
    for item in resolved:
        columns = read_catalogue_columns(item.path)
        n_rows = columns["m1_source"].shape[0]
        count = int(rng.poisson(n_rows * (span_seconds / SECONDS_PER_YEAR) * rate_multiplier))
        if count == 0:
            continue
        indices = rng.choice(n_rows, size=count, replace=count > n_rows)
        blocks["distance"].append(columns["dL"][indices] * 1000.0)
        blocks["redshift"].append(columns["z"][indices])
        blocks["detector_frame_mass_1"].append(columns["m1_source"][indices] * (1.0 + columns["z"][indices]))
        blocks["coa_time"].append(rng.uniform(0.0, span_seconds, size=count))
    return {name: np.concatenate(values) if values else np.empty(0) for name, values in blocks.items()}


def _large_pair(tmp_path: Path, n_rows: int = 400) -> tuple:
    """Write and resolve a BNS/BBH pair large enough that a draw never exceeds it.

    Args:
        tmp_path: Directory to write the files in.
        n_rows: Rows per catalogue.

    Returns:
        The resolved BNS and BBH catalogues, in that order.
    """
    rng = np.random.default_rng(17)
    resolved = []
    for name, source_class, low, high in (("bns", "BNS", 1.0, 2.0), ("bbh", "BBH", 5.0, 60.0)):
        path = tmp_path / f"{name}_large.h5"
        mass_2 = rng.uniform(low, high, n_rows)
        write_catalogue(
            path,
            mass_1=mass_2 + rng.uniform(0.0, high - low, n_rows),
            mass_2=mass_2,
            redshift=rng.uniform(0.0, 3.0, n_rows),
        )
        resolved.append(resolved_catalogue(path, name=name, source_class=source_class))
    return tuple(resolved)


@pytest.mark.requires_h5py
def test_a_count_larger_than_the_catalogue_is_refused(tmp_path: Path) -> None:
    """Rows are sampled without replacement, so a count the catalogue cannot meet is refused."""
    bns, bbh = make_pair(tmp_path)

    with pytest.raises(PopulationValidationError, match=r"holds 3 rows.*allow_replacement"):
        draw_catalogue_population(seed=0, span_seconds=SECONDS_PER_YEAR * 20.0, resolved=(bns, bbh))


@pytest.mark.requires_h5py
@pytest.mark.parametrize("allow_replacement", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_a_count_within_the_catalogue_draws_exactly_as_before(
    tmp_path: Path, seed: int, *, allow_replacement: bool
) -> None:
    """A count the catalogue can meet is drawn bit for bit as before, whether or not replacement is allowed."""
    resolved = _large_pair(tmp_path)
    span = SECONDS_PER_YEAR * 0.5

    draw = draw_catalogue_population(
        seed=seed, span_seconds=span, resolved=resolved, allow_replacement=allow_replacement
    )
    reference = _reference_draw(resolved, seed=seed, span_seconds=span)

    assert draw.n_rows > 0
    assert all(count <= 400 for count in draw.drawn_per_class.values())
    for name, values in reference.items():
        assert draw.parameters[name].dtype == values.dtype, name
        assert draw.parameters[name].tobytes() == values.tobytes(), name


@pytest.mark.requires_h5py
def test_allowing_replacement_draws_an_oversized_count_as_before(tmp_path: Path) -> None:
    """With replacement allowed, a count larger than the catalogue is drawn bit for bit as before."""
    bns, bbh = make_pair(tmp_path)
    span = SECONDS_PER_YEAR * 20.0

    draw = draw_catalogue_population(seed=0, span_seconds=span, resolved=(bns, bbh), allow_replacement=True)
    reference = _reference_draw((bns, bbh), seed=0, span_seconds=span)

    assert draw.allow_replacement is True
    assert all(count > 3 for count in draw.drawn_per_class.values())
    for name, values in reference.items():
        assert draw.parameters[name].tobytes() == values.tobytes(), name


@pytest.mark.requires_h5py
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


@pytest.mark.requires_h5py
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


@pytest.mark.requires_h5py
def test_read_catalogue_columns_rejects_a_missing_column(tmp_path: Path) -> None:
    """A catalogue without a required column is refused with the column named."""
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as handle:
        handle["m1_source"] = np.asarray([1.0, 2.0])
        handle["m2_source"] = np.asarray([1.0, 2.0])

    with pytest.raises(PopulationValidationError, match="z"):
        read_catalogue_columns(path)


@pytest.mark.requires_h5py
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


@pytest.mark.requires_h5py
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
        allow_replacement=True,
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


@pytest.mark.requires_h5py
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
            allow_replacement=True,
            seed=0,
            span_seconds=SECONDS_PER_YEAR * 100.0,
            resolved=(resolved,),
        )


@pytest.mark.requires_h5py
def test_draw_rejects_non_numeric_catalogue_columns(tmp_path: Path) -> None:
    """A non-numeric catalogue column is refused as malformed, not as a NumPy error."""
    path = tmp_path / "bns.h5"
    write_catalogue(
        path,
        mass_1=np.asarray([1.2, 1.4, 1.6]),
        mass_2=np.asarray([1.1, 1.3, 1.5]),
        redshift=np.asarray([0.05, 0.1, 0.2]),
    )
    with h5py.File(path, "a") as handle:
        del handle["chi1z"]
        handle.create_dataset("chi1z", data=["a", "b", "c"], dtype=h5py.string_dtype())
    resolved = resolved_catalogue(path, name="bns", source_class="BNS")

    with pytest.raises(PopulationValidationError, match="non-numeric chi1z"):
        draw_catalogue_population(
            allow_replacement=True,
            seed=0,
            span_seconds=SECONDS_PER_YEAR * 100.0,
            resolved=(resolved,),
        )


@pytest.mark.requires_h5py
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
            allow_replacement=True,
            seed=0,
            span_seconds=SECONDS_PER_YEAR,
            catalogues=(mismatched,),
            resolved=(bns,),
        )


_SPIN_COLUMNS = ("chi1x", "chi1y", "chi2x", "chi2y")
_TIDE_COLUMNS = ("Lambda1", "Lambda2")


def _pair_with_extras(tmp_path: Path) -> tuple:
    """Write a BNS file carrying deformabilities and a BBH file carrying in-plane spins.

    Each extra value is a distinct function of the row's primary mass, so a test
    can tell which catalogue row an emitted value came from.

    Args:
        tmp_path: Directory to write the files in.

    Returns:
        The resolved BNS and BBH catalogues, in that order.
    """
    bns_mass_1 = np.asarray([1.2, 1.4, 1.6])
    bbh_mass_1 = np.asarray([30.0, 40.0, 50.0])
    bns_path = tmp_path / "bns.h5"
    bbh_path = tmp_path / "bbh.h5"
    write_catalogue(
        bns_path,
        mass_1=bns_mass_1,
        mass_2=np.asarray([1.1, 1.3, 1.5]),
        redshift=np.asarray([0.05, 0.1, 0.2]),
        extra={"Lambda1": 100.0 * bns_mass_1, "Lambda2": 200.0 * bns_mass_1},
    )
    write_catalogue(
        bbh_path,
        mass_1=bbh_mass_1,
        mass_2=np.asarray([20.0, 30.0, 40.0]),
        redshift=np.asarray([0.5, 1.0, 1.5]),
        extra={column: (k + 1) * bbh_mass_1 / 1000.0 for k, column in enumerate(_SPIN_COLUMNS)},
    )
    return (
        resolved_catalogue(bns_path, name="bns", source_class="BNS", extra_columns=_TIDE_COLUMNS),
        resolved_catalogue(bbh_path, name="bbh", source_class="BBH", extra_columns=_SPIN_COLUMNS),
    )


@pytest.mark.requires_h5py
def test_declared_extra_columns_are_emitted_from_their_own_rows(tmp_path: Path) -> None:
    """Each declared column is emitted for its file's rows; undeclared ones are zero."""
    bns, bbh = _pair_with_extras(tmp_path)
    draw = draw_catalogue_population(
        allow_replacement=True, seed=3, span_seconds=SECONDS_PER_YEAR * 20.0, resolved=(bns, bbh)
    )
    parameters = draw.parameters
    source_mass_1 = parameters["detector_frame_mass_1"] / (1.0 + parameters["redshift"])
    is_bns = draw.source_class == "BNS"
    assert np.any(is_bns)
    assert np.any(~is_bns)

    np.testing.assert_allclose(parameters["lambda_1"][is_bns], 100.0 * source_mass_1[is_bns], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(parameters["lambda_2"][is_bns], 200.0 * source_mass_1[is_bns], rtol=1e-12, atol=0.0)
    for k, name in enumerate(("spin_1x", "spin_1y", "spin_2x", "spin_2y")):
        expected = (k + 1) * source_mass_1[~is_bns] / 1000.0
        np.testing.assert_allclose(parameters[name][~is_bns], expected, rtol=1e-12, atol=0.0)
        assert np.all(parameters[name][is_bns] == 0.0)
    assert np.all(parameters["lambda_1"][~is_bns] == 0.0)
    assert np.all(parameters["lambda_2"][~is_bns] == 0.0)


@pytest.mark.requires_h5py
def test_extra_columns_do_not_change_the_seeded_draw(tmp_path: Path) -> None:
    """Carrying extra columns consumes no randomness, so every other parameter is unchanged."""
    plain_dir = tmp_path / "plain"
    extra_dir = tmp_path / "extra"
    plain_dir.mkdir()
    extra_dir.mkdir()
    plain = make_pair(plain_dir)
    extra = _pair_with_extras(extra_dir)
    first = draw_catalogue_population(
        allow_replacement=True, seed=11, span_seconds=SECONDS_PER_YEAR * 5.0, resolved=plain
    )
    second = draw_catalogue_population(
        allow_replacement=True, seed=11, span_seconds=SECONDS_PER_YEAR * 5.0, resolved=extra
    )

    for name in PARAMETER_NAMES:
        if name in ("spin_1x", "spin_1y", "spin_2x", "spin_2y", "lambda_1", "lambda_2"):
            continue
        np.testing.assert_array_equal(first.parameters[name], second.parameters[name])


@pytest.mark.requires_h5py
def test_a_declared_extra_column_missing_from_the_file_is_refused(tmp_path: Path) -> None:
    """A manifest entry that declares a column the file lacks fails the draw, naming it."""
    path = tmp_path / "bbh.h5"
    write_catalogue(path, mass_1=np.asarray([30.0]), mass_2=np.asarray([20.0]), redshift=np.asarray([0.5]))
    resolved = resolved_catalogue(path, name="bbh", source_class="BBH", extra_columns=("chi1x",))

    with pytest.raises(PopulationValidationError, match="chi1x"):
        draw_catalogue_population(
            allow_replacement=True, seed=0, span_seconds=SECONDS_PER_YEAR * 100.0, resolved=(resolved,)
        )


@pytest.mark.requires_h5py
@pytest.mark.parametrize("column", [*_SPIN_COLUMNS, *_TIDE_COLUMNS])
def test_draw_rejects_non_finite_extra_columns(tmp_path: Path, column: str) -> None:
    """A non-finite value in a declared extra column is refused like any other column."""
    path = tmp_path / "cat.h5"
    write_catalogue(
        path,
        mass_1=np.asarray([1.2, 1.4, 1.6]),
        mass_2=np.asarray([1.1, 1.3, 1.5]),
        redshift=np.asarray([0.05, 0.1, 0.2]),
        extra={column: np.asarray([np.nan, np.nan, np.nan])},
    )
    resolved = resolved_catalogue(path, name="cat", source_class="BNS", extra_columns=(column,))

    with pytest.raises(PopulationValidationError, match=column):
        draw_catalogue_population(
            allow_replacement=True, seed=0, span_seconds=SECONDS_PER_YEAR * 100.0, resolved=(resolved,)
        )


def test_an_unknown_extra_column_is_refused() -> None:
    """A manifest entry naming a column the draw cannot emit is refused at construction."""
    with pytest.raises(ValueError, match="theta_jn"):
        CatalogueFile(
            name="x",
            source_class="BBH",
            filename="x.h5",
            sha256="0" * 64,
            size_bytes=1,
            n_rows=1,
            extra_columns=("theta_jn",),
        )


@pytest.mark.requires_h5py
def test_a_carried_but_undeclared_extra_column_is_refused(tmp_path: Path) -> None:
    """A file holding in-plane spins its manifest entry does not declare fails, rather than drawing zeros."""
    path = tmp_path / "bbh.h5"
    write_catalogue(
        path,
        mass_1=np.asarray([30.0, 40.0]),
        mass_2=np.asarray([20.0, 30.0]),
        redshift=np.asarray([0.5, 1.0]),
        extra={"chi1x": np.asarray([0.1, 0.2])},
    )
    resolved = resolved_catalogue(path, name="bbh", source_class="BBH")

    with pytest.raises(PopulationValidationError, match="chi1x"):
        draw_catalogue_population(
            allow_replacement=True, seed=0, span_seconds=SECONDS_PER_YEAR * 100.0, resolved=(resolved,)
        )
