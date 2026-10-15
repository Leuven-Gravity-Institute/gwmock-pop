"""Tests for the IMBH binary list generator."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy import units
from astropy.cosmology import FlatLambdaCDM
from scipy import integrate, stats

from gwmock_pop.catalogue.imbh_list import (
    IMBH_ROUTING_KEY,
    IMBH_TYPE,
    IMRI_TYPE,
    JULIAN_YEAR_SECONDS,
    ImbhListSettings,
    ImbhPopulation,
    generate_imbh_list,
    imbh_rate,
    surviving_mass_fraction,
)
from gwmock_pop.catalogue.mdc_list import _event_uniforms
from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.loaders import FilePopulationLoader, write_population_catalogue
from gwmock_pop.provenance import read_provenance

_MONTH_SETTINGS = ImbhListSettings(start_time=1_000_000_000.0, segment_duration=2048.0, n_segments=1300, seed=11)

#: About 40 years of data, so the distributions can be tested on ~2e4 events.
_LONG_SETTINGS = ImbhListSettings(start_time=0.0, segment_duration=2**20, n_segments=1200, seed=3)


def _reference_rate(population: ImbhPopulation, *, capped: bool) -> float:
    """Integrate the rate with astropy distances and adaptive quadrature."""
    cosmology = FlatLambdaCDM(H0=population.hubble_constant, Om0=population.omega_m, Tcmb0=0.0)
    exponent = population.gamma + population.kappa

    def density(z: float) -> float:
        shape = (1.0 + (1.0 + population.z_peak) ** -exponent) * (1.0 + z) ** population.gamma
        shape /= 1.0 + ((1.0 + z) / (1.0 + population.z_peak)) ** exponent
        volume = 4.0 * math.pi * cosmology.differential_comoving_volume(z).to_value(units.Gpc**3 / units.sr)
        value = population.local_rate * shape / (1.0 + z) * volume
        if capped:
            upper = min(population.max_total_mass, population.max_detector_frame_total_mass / (1.0 + z))
            ratio = math.log(max(upper, population.min_total_mass) / population.min_total_mass)
            value *= min(max(ratio / math.log(population.max_total_mass / population.min_total_mass), 0.0), 1.0)
        return value

    return integrate.quad(density, 0.0, population.max_redshift, limit=500, epsrel=1e-10)[0]


@pytest.fixture(scope="module")
def long_list():
    """A long list for distribution tests, with the IMRI sub-population on."""
    return generate_imbh_list(_LONG_SETTINGS, ImbhPopulation(imri_fraction=0.25))


@pytest.mark.parametrize("cap", [1e12, 3000.0, 1500.0])
def test_rate_integral_matches_an_independent_cosmology(cap: float) -> None:
    """The rate agrees with astropy distances integrated by adaptive quadrature."""
    population = ImbhPopulation(max_detector_frame_total_mass=cap)
    rate = imbh_rate(population)
    assert rate.uncapped_events_per_year == pytest.approx(_reference_rate(population, capped=False), rel=1e-5)
    assert rate.events_per_year == pytest.approx(_reference_rate(population, capped=True), rel=1e-4)


def test_rate_has_the_euclidean_limit_at_low_redshift() -> None:
    """With a constant rate and a small volume, the count is R0 * 4/3 pi (c z / H0)^3."""
    population = ImbhPopulation(gamma=0.0, kappa=0.0, max_redshift=1e-3, local_rate=2.0)
    radius_gpc = 299792.458 / population.hubble_constant * population.max_redshift / 1e3
    euclidean = population.local_rate * 4.0 / 3.0 * math.pi * radius_gpc**3
    # Leading corrections are of order z, from the expansion and time dilation.
    assert imbh_rate(population).events_per_year == pytest.approx(euclidean, rel=5e-3)


def test_rate_is_linear_in_the_local_rate() -> None:
    """The local rate is a pure prefactor of the event rate."""
    base = imbh_rate(ImbhPopulation()).events_per_year
    assert imbh_rate(ImbhPopulation(local_rate=0.39)).events_per_year == pytest.approx(3.0 * base, rel=1e-12)


def test_fiducial_rate_per_month() -> None:
    """The fiducial population gives about 43 mergers a month before the cap and 40 after it."""
    rate = imbh_rate(ImbhPopulation())
    assert rate.uncapped_events_per_year / 12.0 == pytest.approx(43.20, abs=0.01)
    assert rate.events_per_year / 12.0 == pytest.approx(39.94, abs=0.01)
    assert rate.mean_interval == pytest.approx(JULIAN_YEAR_SECONDS / rate.events_per_year)


def test_surviving_mass_fraction_limits() -> None:
    """The cap keeps the whole mass range at low redshift and none of it once (1+z) M_min exceeds it."""
    population = ImbhPopulation(max_detector_frame_total_mass=2000.0)
    fraction = surviving_mass_fraction(population, np.array([0.0, 1.0, 3.0, 19.0, 25.0]))
    np.testing.assert_allclose(fraction, [1.0, 1.0, math.log(5.0) / math.log(10.0), 0.0, 0.0], rtol=1e-12, atol=0.0)


def test_listed_events_respect_the_population_bounds(long_list) -> None:
    """Every listed event lies inside the mass, ratio, spin and cap bounds."""
    population = long_list.population
    columns = long_list.columns
    total = columns["source_frame_mass_1"] + columns["source_frame_mass_2"]
    ratio = columns["source_frame_mass_1"] / columns["source_frame_mass_2"]
    assert np.all((total >= population.min_total_mass) & (total <= population.max_total_mass))
    detector_total = columns["detector_frame_mass_1"] + columns["detector_frame_mass_2"]
    assert np.all(detector_total <= population.max_detector_frame_total_mass * (1.0 + 1e-12))
    np.testing.assert_allclose(detector_total, total * (1.0 + columns["redshift"]), rtol=1e-12, atol=0.0)
    main = columns["source_type"] == IMBH_TYPE
    imri = columns["source_type"] == IMRI_TYPE
    assert np.all(main | imri)
    assert np.all((ratio[main] >= 1.0) & (ratio[main] <= 10.0 * (1.0 + 1e-12)))
    assert np.all((ratio[imri] >= 10.0 * (1.0 - 1e-12)) & (ratio[imri] <= 100.0 * (1.0 + 1e-12)))
    for index in (1, 2):
        vector = np.stack([columns[f"spin_{index}{axis}"] for axis in "xyz"])
        np.testing.assert_allclose(np.linalg.norm(vector, axis=0), columns[f"spin_{index}"], rtol=1e-12, atol=1e-15)
        assert np.all(columns[f"spin_{index}"] <= population.max_spin)
    assert np.all(columns["lambda_1"] == 0.0)
    assert np.all(columns["lambda_2"] == 0.0)
    assert np.all((columns["redshift"] >= 0.0) & (columns["redshift"] <= population.max_redshift))


def test_distributions_follow_the_model(long_list) -> None:
    """Redshift, mass, mass ratio, spins and the IMRI fraction follow their stated distributions."""
    columns = long_list.columns
    population = long_list.population
    rate = long_list.rate
    assert long_list.n_listed > 15000

    redshift = columns["redshift"]
    assert stats.kstest(redshift, lambda z: np.interp(z, rate.redshift, rate.cumulative)).pvalue > 1e-3

    # Given z, log M is uniform up to the capped bound.
    total = columns["source_frame_mass_1"] + columns["source_frame_mass_2"]
    upper = np.minimum(population.max_total_mass, population.max_detector_frame_total_mass / (1.0 + redshift))
    position = np.log(total / population.min_total_mass) / np.log(upper / population.min_total_mass)
    assert stats.kstest(position, "uniform").pvalue > 1e-3

    ratio = columns["source_frame_mass_1"] / columns["source_frame_mass_2"]
    main = columns["source_type"] == IMBH_TYPE
    assert stats.kstest((ratio[main] - 1.0) / 9.0, "uniform").pvalue > 1e-3
    assert stats.kstest((ratio[~main] - 10.0) / 90.0, "uniform").pvalue > 1e-3
    imri_share = np.count_nonzero(~main) / long_list.n_listed
    assert abs(imri_share - 0.25) < 4.0 * math.sqrt(0.25 * 0.75 / long_list.n_listed)

    for index in (1, 2):
        assert stats.kstest(columns[f"spin_{index}"] / population.max_spin, "uniform").pvalue > 1e-3
        cos_tilt = columns[f"spin_{index}z"] / columns[f"spin_{index}"]
        assert stats.kstest((cos_tilt + 1.0) / 2.0, "uniform").pvalue > 1e-3


def test_event_count_follows_the_rate(long_list) -> None:
    """The number of mergers in the data is the rate integral times the span, within Poisson scatter."""
    settings = long_list.settings
    span_years = settings.n_segments * settings.segment_duration / JULIAN_YEAR_SECONDS
    expected = long_list.rate.events_per_year * span_years
    coa_time = long_list.columns["coa_time"]
    observed = np.count_nonzero((coa_time >= settings.start_time) & (coa_time < settings.stop_time))
    assert abs(observed - expected) < 4.0 * math.sqrt(expected)


def test_draws_follow_the_mock_data_challenge_arrival_process() -> None:
    """Each event re-seeds with seed + k, waits -mean * log(u0) and maps uniforms 11-15 to the angles."""
    generated = generate_imbh_list(_MONTH_SETTINGS)
    columns = generated.columns
    mean_interval = generated.rate.mean_interval
    generator = np.random.RandomState()
    coa_time = _MONTH_SETTINGS.start_time
    listed = iter(range(generated.n_listed))
    row = next(listed)
    for k in range(1, generated.n_events + 1):
        generator.seed(_MONTH_SETTINGS.seed + k)
        words = generator.randint(0, 2**32, size=16, dtype=np.uint64)
        u = words / 2.0**32
        coa_time += -mean_interval * math.log(u[0])
        if row is None or columns["event_id"][row] != k:
            continue
        assert columns["coa_time"][row] == coa_time
        assert columns["right_ascension"][row] == pytest.approx(2.0 * math.pi * u[11] - math.pi, abs=1e-15)
        assert math.sin(columns["declination"][row]) == pytest.approx(2.0 * u[12] - 1.0, abs=1e-12)
        assert math.cos(columns["inclination"][row]) == pytest.approx(2.0 * u[13] - 1.0, abs=1e-12)
        assert columns["polarization_angle"][row] == pytest.approx(2.0 * math.pi * u[14], abs=1e-15)
        assert columns["coa_phase"][row] == pytest.approx(2.0 * math.pi * u[15], abs=1e-15)
        row = next(listed, None)
    assert row is None


def test_event_streams_extend_the_cbc_streams() -> None:
    """An event's first nine uniforms are the ones the CBC list draws for the same seed."""
    generator = np.random.RandomState()
    for seed in (0, 1, 101, 2**32 + 5):
        np.testing.assert_array_equal(_event_uniforms(generator, seed, 16)[:9], _event_uniforms(generator, seed))


def test_signal_bounds_and_segments() -> None:
    """Signals start before and end after the merger, and each listed event overlaps its segments."""
    generated = generate_imbh_list(_MONTH_SETTINGS)
    columns = generated.columns
    assert generated.n_listed > 0
    assert np.all(columns["start_time"] < columns["coa_time"])
    assert np.all(columns["coa_time"] < columns["end_time"])
    assert np.all(columns["start_time"] < _MONTH_SETTINGS.stop_time)
    assert np.all(columns["first_segment"] < columns["stop_segment"])
    for segment in (0, 650, 1299):
        mask = generated.in_segment(segment)
        lower = _MONTH_SETTINGS.start_time + segment * _MONTH_SETTINGS.segment_duration
        upper = lower + _MONTH_SETTINGS.segment_duration
        overlaps = (columns["start_time"] < upper) & (columns["end_time"] + 3.0 > lower)
        np.testing.assert_array_equal(mask, overlaps)


def test_generation_is_reproducible() -> None:
    """The same seed gives the same list, and another seed a different one."""
    first = generate_imbh_list(_MONTH_SETTINGS)
    second = generate_imbh_list(_MONTH_SETTINGS)
    other = generate_imbh_list(replace(_MONTH_SETTINGS, seed=12))
    for name, values in first.columns.items():
        np.testing.assert_array_equal(values, second.columns[name])
    assert not np.array_equal(first.columns["coa_time"][:5], other.columns["coa_time"][:5])


def test_population_round_trips_through_the_file_loader(tmp_path: Path) -> None:
    """The population is written with its record and loads under the BBH routing key."""
    generated = generate_imbh_list(_MONTH_SETTINGS, ImbhPopulation(imri_fraction=0.5))
    output = tmp_path / "imbh.h5"
    write_population_catalogue(
        output, generated.population_columns(), provenance=generated.provenance(file_format="hdf5")
    )

    loaded = FilePopulationLoader(IMBH_ROUTING_KEY, output).simulate()
    expected = generated.population_columns()
    order = np.argsort(np.asarray(loaded["coa_time"]))
    for name, values in expected.items():
        np.testing.assert_array_equal(np.asarray(loaded[name])[order], values)

    record = read_provenance(output)
    assert record["run"]["seed"] == _MONTH_SETTINGS.seed
    configuration = record["origin"]["configuration"]
    assert configuration["population"]["local_rate"] == 0.13
    assert configuration["population"]["imri_fraction"] == 0.5
    assert configuration["population"]["waveform_approximant"] == "IMRPhenomXPHM"
    assert configuration["settings"]["f_min"] == 5.0
    assert configuration["events_per_year"] == generated.rate.events_per_year
    assert configuration["mean_interval_seconds"] == generated.rate.mean_interval
    assert configuration["n_events_listed"] == generated.n_listed
    assert configuration["n_imri_listed"] == np.count_nonzero(generated.columns["source_type"] == IMRI_TYPE)
    assert record["origin"]["environment"]["numpy"] == np.__version__
    assert record["catalogue"]["n_samples"] == generated.n_listed


@pytest.mark.parametrize(
    "change",
    [
        {"min_total_mass": 0.0},
        {"max_total_mass": 50.0},
        {"min_mass_ratio": 0.5},
        {"max_mass_ratio": math.nan},
        {"imri_min_mass_ratio": 200.0},
        {"imri_fraction": 1.5},
        {"max_spin": 1.2},
        {"local_rate": -1.0},
        {"max_redshift": math.inf},
        {"max_detector_frame_total_mass": 50.0},
        {"omega_m": 2.0},
    ],
)
def test_population_refuses_out_of_range_models(change: dict) -> None:
    """A model that cannot describe a population is refused."""
    with pytest.raises(PopulationValidationError):
        imbh_rate(replace(ImbhPopulation(), **change))


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1.5},
        {"seed": True},
        {"n_segments": 0},
        {"segment_duration": 0.0},
        {"f_min": -5.0},
        {"start_time": math.nan},
    ],
)
def test_settings_refuse_out_of_range_values(change: dict) -> None:
    """Settings that cannot describe a run are refused."""
    with pytest.raises(PopulationValidationError):
        generate_imbh_list(replace(_MONTH_SETTINGS, **change))
