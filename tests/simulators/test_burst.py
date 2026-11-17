"""Tests for the standard-candle burst population."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.burst import (
    PARAMETER_NAMES,
    STANDARD_CANDLE_LABEL,
    StandardCandleBurstPopulation,
    draw_standard_candle_population,
    standard_candle_hrss,
)

START = 1_419_724_818.0
SPAN = 30 * 86400.0
YEAR_S = 365.25 * 86400.0
SG = {"burst_model": "sine_gaussian", "quality_factor": 9.0}
WNB = {"burst_model": "white_noise_burst", "low_frequency": 250.0, "bandwidth": 100.0, "duration": 0.1}
# 1e12 Gpc^-3 yr^-1 within 1 Mpc over 30 days: a mean of about 344 sources per component.
RATE = 1.0e12


def _components(rate=RATE, max_distance=1.0, families=(SG, WNB)):
    return [{"family": dict(f), "rate_density": rate, "max_distance": max_distance} for f in families]


def _draw(**overrides):
    kwargs = {
        "start_time": START,
        "end_time": START + SPAN,
        "energy": 1.0e-8,
        "components": _components(),
        "rng": np.random.default_rng(2026),
    }
    kwargs.update(overrides)
    return draw_standard_candle_population(**kwargs)


def _isotropic_hrss_reference(energy, distance_mpc, frequency):
    """O3 all-sky search Eq. (2) solved for hrss, with astropy's constants: an independent evaluation."""
    e = (energy * u.M_sun * const.c**2).to(u.J)
    r = (distance_mpc * u.Mpc).to(u.m)
    hrss2 = e * const.G / (math.pi**2 * const.c**3 * r**2 * (frequency * u.Hz) ** 2)
    return math.sqrt(hrss2.to(u.Hz**-1).value)


# ---------------------------------------------------------------- the energy-to-hrss relation


@pytest.mark.parametrize("frequency", [70.0, 235.0, 1304.0])
def test_isotropic_hrss_matches_the_o3_relation(frequency):
    """A white-noise burst's hrss is Eq. (2) of the O3 all-sky search solved for hrss."""
    hrss = standard_candle_hrss(energy=1e-2, distance=10.0, frequency=frequency, inclination=0.3, elliptical=False)
    assert hrss == pytest.approx(_isotropic_hrss_reference(1e-2, 10.0, frequency), rel=1e-9, abs=0.0)


def test_elliptical_hrss_face_on_carries_the_rotating_source_factor():
    """Face on, E_rot = (2/5) E_iso(hrss), so hrss is sqrt(5/2) times the isotropic value."""
    hrss = standard_candle_hrss(energy=1e-2, distance=10.0, frequency=235.0, inclination=0.0, elliptical=True)
    assert hrss == pytest.approx(math.sqrt(2.5) * _isotropic_hrss_reference(1e-2, 10.0, 235.0), rel=1e-9, abs=0.0)


def test_elliptical_hrss_follows_the_rotating_pattern():
    """hrss^2 follows ((1 + cos^2 i) / 2)^2 + cos^2 i, relative to face on (where it is 2)."""
    face_on = standard_candle_hrss(energy=1.0, distance=1.0, frequency=100.0, inclination=0.0, elliptical=True)
    for iota in (0.4, 1.0, math.pi / 2, 2.5, math.pi):
        c = math.cos(iota)
        pattern = ((1 + c**2) / 2) ** 2 + c**2
        hrss = standard_candle_hrss(energy=1.0, distance=1.0, frequency=100.0, inclination=iota, elliptical=True)
        assert (hrss / face_on) ** 2 == pytest.approx(pattern / 2, rel=1e-12, abs=0.0)


def test_elliptical_hrss_radiates_the_stated_energy_over_the_sphere():
    """The average of hrss^2 over isotropic orientations is the isotropic hrss^2: energy is conserved."""
    cos_iota, weights = np.polynomial.legendre.leggauss(16)  # exact for this polynomial in cos(iota)
    hrss2 = [
        standard_candle_hrss(energy=1e-3, distance=3.0, frequency=554.0, inclination=math.acos(c), elliptical=True) ** 2
        for c in cos_iota
    ]
    mean = float(np.dot(weights, hrss2)) / 2.0
    assert mean == pytest.approx(_isotropic_hrss_reference(1e-3, 3.0, 554.0) ** 2, rel=1e-9, abs=0.0)


def test_hrss_scales_as_the_inverse_distance_and_root_energy():
    """Hrss is proportional to sqrt(E) / r."""
    base = standard_candle_hrss(energy=1e-4, distance=2.0, frequency=300.0, inclination=1.0, elliptical=True)
    scaled = standard_candle_hrss(energy=4e-4, distance=4.0, frequency=300.0, inclination=1.0, elliptical=True)
    assert scaled == pytest.approx(base, rel=1e-14, abs=0.0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"energy": 0.0}, "energy"),
        ({"energy": math.inf}, "energy"),
        ({"distance": -1.0}, "distance"),
        ({"frequency": math.nan}, "frequency"),
        ({"inclination": math.inf}, "inclination"),
    ],
)
def test_hrss_rejects_invalid_arguments(overrides, match):
    """Non-positive or non-finite inputs are refused."""
    kwargs = {"energy": 1e-2, "distance": 1.0, "frequency": 100.0, "inclination": 0.0, "elliptical": True}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        standard_candle_hrss(**kwargs)


# ---------------------------------------------------------------- the draw


def test_each_component_count_is_poisson_with_its_own_volume_rate_mean():
    """Each component's count is Poisson with mean R * (4/3) pi d^3 * T, independently of the others."""
    components = [
        {"family": SG, "rate_density": RATE, "max_distance": 1.0},
        {"family": WNB, "rate_density": RATE / 8, "max_distance": 2.0},  # the same mean: 1/8 * 2^3
        {"family": SG | {"frequency": 100.0}, "rate_density": 0.0, "max_distance": 1.0},
    ]
    mean = RATE * 1e-9 * (4.0 / 3.0) * math.pi * (SPAN / YEAR_S)
    assert mean == pytest.approx(344.0, rel=2e-3)
    draws = 200
    sg, wnb, silent = [], [], 0
    for seed in range(draws):
        events = _draw(components=components, rng=np.random.default_rng(seed))
        sg.append(sum(e["burst_model"] == "sine_gaussian" for e in events))
        wnb.append(sum(e["burst_model"] == "white_noise_burst" for e in events))
        silent += sum(e.get("frequency") == 100.0 for e in events)
    # Means within 5 standard errors; a Poisson variance equals its mean (the sample variance has
    # relative standard deviation sqrt(2 / draws) = 0.1, so 0.5 is 5 sigma).
    for counts in (sg, wnb):
        assert abs(np.mean(counts) - mean) < 5 * math.sqrt(mean / draws)
        assert np.var(counts, ddof=1) == pytest.approx(mean, rel=0.5)
    assert silent == 0


def test_distances_are_uniform_in_volume_and_times_in_the_span():
    """(d / d_max)^3, times and log frequencies are uniform; events come in time order."""
    events = _draw(components=_components(rate=10 * RATE, max_distance=2.0, families=(SG,)))
    n = len(events)
    assert n > 20_000
    grid = np.arange(1, n + 1) / n
    critical = 1.95 / math.sqrt(n)  # Kolmogorov 99.9% critical value
    d3 = np.sort([(e["distance"] / 2.0) ** 3 for e in events])
    times = np.array([e["peak_time"] for e in events])
    log_f = np.sort([math.log(e["frequency"] / 30.0) / math.log(60.0) for e in events])
    assert np.all(np.diff(times) >= 0)
    assert np.max(np.abs(d3 - grid)) < critical
    assert np.max(np.abs(np.sort((times - START) / SPAN) - grid)) < critical
    assert np.max(np.abs(log_f - grid)) < critical
    assert d3[0] > 0
    assert d3[-1] <= 1.0
    assert times.min() >= START
    assert times.max() < START + SPAN


def test_each_events_hrss_follows_from_its_energy_distance_and_orientation():
    """Every event's hrss is standard_candle_hrss of its own energy, distance, frequency and inclination."""
    events = _draw()
    assert {e["burst_model"] for e in events} == {"sine_gaussian", "white_noise_burst"}
    for event in events:
        if event["burst_model"] == "sine_gaussian":
            frequency, elliptical = event["frequency"], True
        else:
            frequency, elliptical = event["low_frequency"] + 0.5 * event["bandwidth"], False
        expected = standard_candle_hrss(
            energy=event["energy"],
            distance=event["distance"],
            frequency=frequency,
            inclination=event["inclination"],
            elliptical=elliptical,
        )
        assert event["hrss"] == expected
        assert event["energy"] == 1.0e-8
        assert event["injection_set"] == STANDARD_CANDLE_LABEL
        assert ("seed" in event) == (event["burst_model"] == "white_noise_burst")
        assert -math.pi / 2 <= event["declination"] <= math.pi / 2
        assert 0 <= event["inclination"] <= math.pi


def test_a_family_may_fix_its_frequency():
    """A family fixing its frequency keeps it, and the hrss follows from it."""
    events = _draw(components=_components(families=(SG | {"frequency": 235.0},)))
    assert events
    for event in events:
        assert event["frequency"] == 235.0
        assert event["hrss"] == standard_candle_hrss(
            energy=1.0e-8,
            distance=event["distance"],
            frequency=235.0,
            inclination=event["inclination"],
            elliptical=True,
        )


def test_the_draw_is_reproducible_and_may_be_empty():
    """The same seed gives the same events; a negligible rate gives none."""
    assert _draw() == _draw()
    assert _draw(components=_components(rate=1e-30)) == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"end_time": START}, "end_time"),
        ({"energy": 0.0}, "energy"),
        ({"components": []}, "components"),
        ({"components": [{"family": SG, "rate_density": 1.0}]}, "max_distance"),
        ({"components": [{"family": SG, "rate_density": 1.0, "max_distance": 1.0, "x": 1}]}, "component 0"),
        ({"components": _components(rate=-1.0)}, "rate_density"),
        ({"components": _components(rate=math.nan)}, "rate_density"),
        ({"components": _components(max_distance=0.0)}, "max_distance"),
        ({"components": _components(max_distance=150.0)}, "100 Mpc"),
        ({"components": _components(families=({"burst_model": "gaussian", "duration": 1e-3},))}, "gaussian"),
        ({"components": _components(families=({"quality_factor": 3.0},))}, "None"),
        ({"components": _components(families=(SG | {"distance": 0.5},))}, "distance"),
        ({"components": _components(families=(SG | {"inclination": 0.0},))}, "inclination"),
        ({"components": _components(families=(SG | {"hrss": 1e-22},))}, "hrss"),
        ({"components": _components(families=(WNB | {"seed": 1},))}, "seed"),
        ({"sine_gaussian_frequency_range": (0.0, 10.0)}, "sine_gaussian_frequency_range"),
        ({"components": _components(families=({"burst_model": "white_noise_burst"},))}, "low_frequency"),
        ({"components": _components(families=({"burst_model": "sine_gaussian"},))}, "quality_factor"),
        ({"components": _components(families=(WNB | {"bandwidth": -100.0},))}, "bandwidth"),
        ({"components": _components(families=(SG | {"frequency": math.nan},))}, "frequency"),
        ({"components": _components(families=(SG | {"quality_factor": 0.0},))}, "quality_factor"),
        ({"components": [{"family": "sine_gaussian", "rate_density": 1.0, "max_distance": 1.0}]}, "mapping"),
    ],
)
def test_the_draw_rejects_invalid_settings(overrides, match):
    """Invalid spans, energies, components, rates, distances and families are refused."""
    with pytest.raises(ValueError, match=match):
        _draw(**overrides)


# ---------------------------------------------------------------- the population interface


def _population(**overrides):
    kwargs = {"start_time": START, "end_time": START + SPAN, "energy": 1.0e-8, "components": _components(), "seed": 7}
    kwargs.update(overrides)
    return StandardCandleBurstPopulation(**kwargs)


def test_the_population_satisfies_the_protocol_and_returns_its_columns():
    """The population is a GWPopSimulator of source type burst; columns are the events, coa_time at the peak."""
    population = _population()
    assert isinstance(population, GWPopSimulator)
    assert population.source_type == "burst"
    columns = population.simulate()
    assert tuple(columns) == PARAMETER_NAMES
    events = _draw(rng=np.random.default_rng(7))
    assert all(len(column) == len(events) for column in columns.values())
    for index, event in enumerate(events):
        assert columns["coa_time"][index] == event["peak_time"]
        for name in PARAMETER_NAMES:
            if name != "coa_time":
                assert columns[name][index] == event.get(name)


def test_the_population_ignores_a_callers_seed_and_refuses_a_size():
    """Its own seed fixes the population; a requested size is refused."""
    population = _population()
    first = population.simulate(seed=1)
    second = population.simulate(seed=2)
    assert all(np.array_equal(first[name], second[name]) for name in PARAMETER_NAMES)
    with pytest.raises(ValueError, match="n_samples"):
        population.simulate(10)
    assert population.metadata["seed"] == 7


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"components": _components(max_distance=150.0)}, "100 Mpc"),
        ({"energy": -1.0}, "energy"),
    ],
)
def test_the_population_checks_its_settings_when_built(overrides, match):
    """A bad setting fails when the population is built, before anything is simulated."""
    with pytest.raises(ValueError, match=match):
        _population(**overrides)
