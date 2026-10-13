"""Tests for the Mock Data Challenge CBC list generator."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from gwmock_pop.catalogue.mdc_list import (
    BBH_TYPE,
    BNS_TYPE,
    MDC1_SETTINGS,
    NSBH_TYPE,
    MdcListSettings,
    _chirp_time_bound,
    _event_uniforms,
    _final_spin_bound,
    _merge_time_bound,
    _natural_cubic_spline,
    _ringdown_time_bound,
    generate_mdc_cbc_list,
    resolve_mdc_inputs,
)
from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.loaders import FilePopulationLoader, write_population_catalogue
from gwmock_pop.provenance import read_provenance

_MSUN_SI = 1.9884098706980507e30

#: Directory holding the four input files of the first Mock Data Challenge.
_INPUTS_ENV = "GWMOCK_POP_MDC_INPUTS"

#: Directory holding the lists the original generator wrote for the same settings.
_REFERENCE_ENV = "GWMOCK_POP_MDC_REFERENCE_LISTS"

#: Largest signal-bound difference accepted against the original lists, in
#: seconds. The bounds depend on the LAL version the lists were made with.
_BOUND_TOLERANCE = 2e-4

#: Columns of the original lists, after the event counter.
_LIST_COLUMNS = (
    "start_time",
    "coa_time",
    "end_time",
    "mass_1",
    "mass_2",
    "spin_1",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
    "redshift",
    "distance",
    "right_ascension",
    "declination",
    "polarization_angle",
    "inclination",
    "coa_phase",
    "snr",
    "source_type",
)


def _write_inputs(directory: Path, *, n_rows: int = 200, seed: int = 3) -> Path:
    """Write small synthetic input files in the original layout."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    grid = 1.0 + np.arange(151) * 0.01
    np.savetxt(directory / "lambda.txt", np.column_stack([grid, 4000.0 * grid**-6]), delimiter="\t")

    def masses(low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        first = rng.uniform(low, high, n_rows)
        second = rng.uniform(low, high, n_rows)
        return np.maximum(first, second), np.minimum(first, second)

    redshift = rng.uniform(0.1, 3.0, n_rows)
    distance = rng.uniform(500.0, 30000.0, n_rows)
    bns_1, bns_2 = masses(1.1, 2.5)
    np.savetxt(directory / "list_BNS.txt", np.column_stack([bns_1, bns_2, redshift, distance]))
    np.savetxt(
        directory / "list_BHNS.txt",
        np.column_stack([rng.uniform(5.0, 30.0, n_rows), rng.uniform(1.1, 2.5, n_rows), redshift, distance]),
    )
    bbh_1, bbh_2 = masses(5.0, 60.0)
    spins = rng.uniform(0.0, 0.9, (n_rows, 2))
    tilts = rng.uniform(0.0, np.pi, (n_rows, 2))
    np.savetxt(directory / "list_BBH.txt", np.column_stack([bbh_1, bbh_2, spins, tilts, redshift, distance]))
    return directory


_SMALL_SETTINGS = MdcListSettings(
    start_time=1000.0,
    segment_duration=64.0,
    n_segments=4,
    mean_interval=4.0,
    bns_fraction=0.5,
    bbh_fraction=0.3,
    seed=7,
    f_min=40.0,
)


@pytest.fixture
def small_inputs(tmp_path: Path) -> dict:
    """Return synthetic inputs, fingerprinted without a pinned digest."""
    return resolve_mdc_inputs(_write_inputs(tmp_path / "inputs"), expected_sha256=None)


def test_event_uniforms_follow_the_reference_mersenne_twister() -> None:
    """The first words of init_genrand(5489) are the published reference outputs."""
    generator = np.random.RandomState()
    draws = _event_uniforms(generator, 5489)

    assert draws[0] == 3499211612 / 2**32
    assert draws[1] == 581869302 / 2**32
    assert draws.shape == (9,)


def test_event_uniforms_reduce_the_seed_like_the_original() -> None:
    """A zero seed becomes 4357 and seeds wrap at 32 bits."""
    generator = np.random.RandomState()

    assert np.array_equal(_event_uniforms(generator, 0), _event_uniforms(generator, 4357))
    assert np.array_equal(_event_uniforms(generator, 2**32), _event_uniforms(generator, 4357))
    assert np.array_equal(_event_uniforms(generator, 2**32 + 11), _event_uniforms(generator, 11))
    assert np.array_equal(_event_uniforms(generator, -1), _event_uniforms(generator, 2**32 - 1))
    assert not np.array_equal(_event_uniforms(generator, 1), _event_uniforms(generator, 2))


# Reference values from lalsimulation 6.2.1 (lal 7.7.1): masses in solar masses,
# spin magnitudes, then chirp, merge, final-spin and ringdown bounds.
_LAL_BOUNDS = [
    (4.202228, 3.818891, 0.0, 0.0, 1135.6213959706963, 0.00670236565033563, 0.686, 0.005304216028954602),
    (1.1, 1.1, 0.0, 0.0, 9627.479440754421, 0.001838297677760221, 0.686, 0.0014548188680033453),
    (36.7, 25.4, 0.1424, 0.1037, 42.37162976163036, 0.05189012990404987, 0.7229150000000001, 0.04187328999492545),
    (60.0, 8.0, 0.9, -0.3, 88.71108197854738, 0.05682011003986137, 0.9, 0.05726910093172924),
]


@pytest.mark.parametrize("reference", _LAL_BOUNDS)
def test_duration_bounds_match_lal(reference: tuple[float, ...]) -> None:
    """The signal-duration bounds agree with LAL's own implementation."""
    m1, m2, s1, s2, chirp, merge, final_spin, ringdown = reference
    mass_1 = np.asarray(m1 * _MSUN_SI)
    mass_2 = np.asarray(m2 * _MSUN_SI)
    larger = np.asarray(max(abs(s2), abs(s1)))
    spin = _final_spin_bound(np.asarray(s1), np.asarray(s2))

    assert float(_chirp_time_bound(5.0, mass_1, mass_2, larger)) == pytest.approx(chirp, rel=1e-13, abs=0.0)
    assert float(_merge_time_bound(mass_1, mass_2)) == pytest.approx(merge, rel=1e-13, abs=0.0)
    assert float(spin) == pytest.approx(final_spin, rel=1e-15, abs=0.0)
    assert float(_ringdown_time_bound(mass_1 + mass_2, spin)) == pytest.approx(ringdown, rel=1e-13, abs=0.0)


def test_final_spin_bound_is_capped() -> None:
    """The remnant-spin over-estimate never exceeds 0.998."""
    assert float(_final_spin_bound(np.asarray(0.999), np.asarray(0.5))) == 0.998


def test_natural_cubic_spline_matches_an_independent_spline() -> None:
    """The spline agrees with scipy's natural cubic spline and hits the knots."""
    grid = 1.0 + np.arange(151) * 0.01
    values = 4000.0 * grid**-6 + 50.0 * np.sin(7.0 * grid)
    x = np.linspace(1.0, 2.5, 997)

    expected = CubicSpline(grid, values, bc_type="natural")(x)
    actual = _natural_cubic_spline(grid, values, x)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(_natural_cubic_spline(grid, values, grid), values, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("mass", [0.999, 2.501])
def test_natural_cubic_spline_refuses_masses_outside_the_table(mass: float) -> None:
    """A mass outside the table is an error, not an extrapolation."""
    grid = 1.0 + np.arange(151) * 0.01
    with pytest.raises(PopulationValidationError, match="tidal-deformability table"):
        _natural_cubic_spline(grid, grid, np.asarray([1.5, mass]))


def test_generation_is_deterministic_and_seeded(small_inputs: dict) -> None:
    """The same settings reproduce the list; another seed changes it."""
    first = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs)
    second = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs)
    other = generate_mdc_cbc_list(replace(_SMALL_SETTINGS, seed=8), small_inputs)

    assert first.n_listed > 10
    for name, values in first.columns.items():
        assert np.array_equal(values, second.columns[name]), name
    assert not np.array_equal(first.columns["coa_time"], other.columns["coa_time"])


def test_catalogue_rows_are_consumed_in_order(small_inputs: dict) -> None:
    """The i-th event of a class takes the i-th row of that class's catalogue."""
    generated = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs)
    columns = generated.columns
    # Every event that merges inside the data is listed, so these are the first
    # events of each class, without gaps.
    inside = columns["coa_time"] < _SMALL_SETTINGS.stop_time
    for code, name in ((BNS_TYPE, "bns"), (NSBH_TYPE, "nsbh"), (BBH_TYPE, "bbh")):
        selected = inside & (columns["source_type"] == code)
        catalogue = np.loadtxt(small_inputs[name].path)
        n = int(np.count_nonzero(selected))
        assert n > 0, name
        np.testing.assert_array_equal(columns["source_frame_mass_1"][selected], catalogue[:n, 0])
        np.testing.assert_array_equal(columns["source_frame_mass_2"][selected], catalogue[:n, 1])
        np.testing.assert_array_equal(columns["distance"][selected], catalogue[:n, -1])
        np.testing.assert_array_equal(columns["redshift"][selected], catalogue[:n, -2])
    assert sum(generated.rows_consumed.values()) == generated.n_events


def test_class_specific_parameters(small_inputs: dict) -> None:
    """Only binary neutron stars carry tidal deformability, only black-hole binaries carry spin."""
    columns = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs).columns
    bns = columns["source_type"] == BNS_TYPE
    bbh = columns["source_type"] == BBH_TYPE

    assert np.all(columns["lambda_1"][bns] > 0.0)
    assert np.all(columns["lambda_2"][bns] >= columns["lambda_1"][bns])
    assert np.all(columns["lambda_1"][~bns] == 0.0)
    assert np.all(columns["lambda_2"][~bns] == 0.0)
    for index in (1, 2):
        components = np.stack([columns[f"spin_{index}{axis}"] for axis in "xyz"])
        assert np.all(components[:, ~bbh] == 0.0)
        np.testing.assert_allclose(np.linalg.norm(components[:, bbh], axis=0), columns[f"spin_{index}"][bbh])
    np.testing.assert_array_equal(
        columns["detector_frame_mass_1"], columns["source_frame_mass_1"] * (1.0 + columns["redshift"])
    )


def test_listing_and_segments(small_inputs: dict) -> None:
    """Listed signals start before the end of the data and sit in the segments they overlap."""
    generated = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs)
    columns = generated.columns
    start, duration = _SMALL_SETTINGS.start_time, _SMALL_SETTINGS.segment_duration

    assert generated.n_events > generated.n_listed
    assert np.all(columns["start_time"] < _SMALL_SETTINGS.stop_time)
    assert np.all(np.diff(columns["event_id"]) > 0)
    assert np.all(np.diff(columns["coa_time"]) > 0.0)
    assert np.all(columns["start_time"] < columns["coa_time"])
    assert np.all(columns["coa_time"] < columns["end_time"])
    for segment in range(_SMALL_SETTINGS.n_segments):
        mask = generated.in_segment(segment)
        segment_start = start + segment * duration
        overlaps = (columns["start_time"] < segment_start + duration) & (columns["end_time"] + 3.0 > segment_start)
        np.testing.assert_array_equal(mask, overlaps)


def test_population_round_trips_through_the_file_loader(small_inputs: dict, tmp_path: Path) -> None:
    """A class population is written with its record and loads as that class."""
    generated = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs)
    output = tmp_path / "bns.h5"
    write_population_catalogue(
        output, generated.population("bns"), provenance=generated.provenance("bns", file_format="hdf5")
    )

    loader = FilePopulationLoader("bns", output)
    loaded = loader.simulate()
    expected = generated.population("bns")
    order = np.argsort(np.asarray(loaded["coa_time"]))
    for name, values in expected.items():
        np.testing.assert_array_equal(np.asarray(loaded[name])[order], values)

    record = read_provenance(output)
    assert record["run"]["seed"] == 7
    configuration = record["origin"]["configuration"]
    assert configuration["bns_fraction"] == 0.5
    assert configuration["bbh_fraction"] == 0.3
    assert configuration["nsbh_fraction"] == pytest.approx(0.2)
    assert configuration["mean_interval"] == 4.0
    assert configuration["start_time"] == 1000.0
    assert configuration["span_seconds"] == 256.0
    digests = {item["name"]: item["sha256"] for item in record["origin"]["files"]}
    assert digests == {name: item.sha256 for name, item in small_inputs.items()}
    assert record["origin"]["environment"]["numpy"] == np.__version__
    assert record["catalogue"]["n_samples"] == len(expected["coa_time"])


def test_population_refuses_an_unknown_class(small_inputs: dict) -> None:
    """Only the three list classes exist."""
    generated = generate_mdc_cbc_list(_SMALL_SETTINGS, small_inputs)
    with pytest.raises(ValueError, match="source_type"):
        generated.population("imbh")


def test_exhausted_catalogue_is_an_error(tmp_path: Path) -> None:
    """Running out of catalogue rows stops the run instead of repeating a row."""
    inputs = resolve_mdc_inputs(_write_inputs(tmp_path, n_rows=3), expected_sha256=None)
    with pytest.raises(PopulationValidationError, match="holds 3 rows"):
        generate_mdc_cbc_list(_SMALL_SETTINGS, inputs)


def test_inputs_are_checked_against_their_digests(tmp_path: Path) -> None:
    """A file that does not match its pinned digest is refused, as is a missing one."""
    directory = _write_inputs(tmp_path)
    with pytest.raises(PopulationValidationError, match="SHA-256"):
        resolve_mdc_inputs(directory)
    (directory / "lambda.txt").unlink()
    with pytest.raises(PopulationValidationError, match="does not exist"):
        resolve_mdc_inputs(directory, expected_sha256=None)


def test_missing_input_role_is_an_error(small_inputs: dict) -> None:
    """Generation needs all four inputs."""
    partial = {name: item for name, item in small_inputs.items() if name != "lambda"}
    with pytest.raises(PopulationValidationError, match="lambda"):
        generate_mdc_cbc_list(_SMALL_SETTINGS, partial)


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 1.5},
        {"n_segments": 0},
        {"n_segments": True},
        {"segment_duration": 0.0},
        {"mean_interval": float("nan")},
        {"f_min": -1.0},
        {"min_component_mass": 0.0},
        {"start_time": float("inf")},
        {"bns_fraction": -0.1},
        {"bbh_fraction": 1.1},
        {"bns_fraction": 0.8, "bbh_fraction": 0.3},
    ],
)
def test_settings_are_validated(changes: dict, small_inputs: dict) -> None:
    """Settings that cannot describe a run are refused."""
    with pytest.raises(PopulationValidationError):
        generate_mdc_cbc_list(replace(_SMALL_SETTINGS, **changes), small_inputs)


def test_mdc1_settings_derive_the_nsbh_fraction() -> None:
    """The first challenge's fractions leave 0.0279 for neutron-star--black-hole binaries."""
    assert MDC1_SETTINGS.nsbh_fraction == pytest.approx(0.0279, abs=1e-12)
    assert MDC1_SETTINGS.stop_time == 1_000_000_000.0 + 1300 * 2048.0


def _read_list(path: Path) -> list[list[str]]:
    """Return the whitespace-separated fields of every row of a list file."""
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def _compare_rows(rows: list[list[str]], columns: dict, indices: np.ndarray, *, mass_frame: str) -> None:
    """Assert list rows equal the generated events at the original printed precision."""
    assert len(rows) == len(indices)
    for row, index in zip(rows, indices, strict=True):
        assert int(row[0]) == columns["event_id"][index]
        for name, field in zip(_LIST_COLUMNS, row[1:], strict=True):
            if name in {"mass_1", "mass_2"}:
                value = columns[f"{mass_frame}_mass_{name[-1]}"][index]
            elif name == "snr":
                value = 0.0
            else:
                value = columns[name][index]
            if name == "source_type":
                assert int(field) == value, (row[0], name)
            elif name in {"start_time", "end_time"}:
                assert abs(float(field) - value) <= _BOUND_TOLERANCE, (row[0], name, field, value)
            else:
                assert field == f"{value:f}", (row[0], name, field, value)


@pytest.mark.skipif(
    not (os.environ.get(_INPUTS_ENV) and os.environ.get(_REFERENCE_ENV)),
    reason=f"set {_INPUTS_ENV} and {_REFERENCE_ENV} to compare against the original lists",
)
def test_mdc1_lists_match_the_original_generator() -> None:
    """The first challenge's lists are reproduced event for event.

    Every column equals the original program's printed value, except the
    signal start and end, which depend on the LAL version and must agree to
    within 2e-4 s.
    """
    inputs = resolve_mdc_inputs(os.environ[_INPUTS_ENV])
    reference = Path(os.environ[_REFERENCE_ENV])
    generated = generate_mdc_cbc_list(MDC1_SETTINGS, inputs)
    columns = generated.columns

    _compare_rows(
        _read_list(reference / "list_cbc.txt"), columns, np.arange(generated.n_listed), mass_frame="source_frame"
    )
    n_segment_rows = 0
    for segment in range(MDC1_SETTINGS.n_segments):
        rows = _read_list(reference / f"cbc_{segment}.dat")
        _compare_rows(rows, columns, np.flatnonzero(generated.in_segment(segment)), mass_frame="detector_frame")
        n_segment_rows += len(rows)

    counts = {code: int(np.count_nonzero(columns["source_type"] == code)) for code in (BNS_TYPE, NSBH_TYPE, BBH_TYPE)}
    assert generated.n_listed == 69781
    assert counts == {BNS_TYPE: 61031, NSBH_TYPE: 2025, BBH_TYPE: 6725}
    assert n_segment_rows > generated.n_listed
