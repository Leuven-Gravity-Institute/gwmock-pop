"""The cosmoRate convolution against cases with an analytic answer, and the mergers it draws."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value

from gwmock_pop.constants import PLANCK18_H0_KM_S_MPC, PLANCK18_OMEGA_M
from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.synthesis.convolution import (
    CosmicHistory,
    MergingBinaries,
    draw_mergers,
    expected_mergers,
    merger_rate_density,
)

pytestmark = pytest.mark.unit

_COSMOLOGY = FlatLambdaCDM(H0=PLANCK18_H0_KM_S_MPC, Om0=PLANCK18_OMEGA_M)
_HISTORY = CosmicHistory(n_redshift=301)
#: Merging binaries per solar mass of star formation, before the binary-fraction and IMF factors.
_PER_MASS = 1e-5
#: The test runs' IMF fraction.
_IMF_FRACTION = 0.3


def _madau_fragos(redshift: np.ndarray) -> np.ndarray:
    return 0.01 * (1.0 + redshift) ** 2.6 / (1.0 + ((1.0 + redshift) / 3.2) ** 6.2)


def _run(
    delays: list[float], metallicities: list[float], evolved_mass: float = 1.0, weight: list[float] | None = None
) -> MergingBinaries:
    return MergingBinaries(
        delay_time=np.asarray(delays, dtype=float),
        metallicity=np.asarray(metallicities, dtype=float),
        evolved_mass=evolved_mass,
        imf_fraction=_IMF_FRACTION,
        properties={"label": np.arange(len(delays))},
        weight=None if weight is None else np.asarray(weight, dtype=float),
    )


def _single_metallicity(delay: float, count: int = 10) -> MergingBinaries:
    return _run([delay] * count, [0.01] * count, evolved_mass=count / _PER_MASS)


def _efficiency(history: CosmicHistory) -> float:
    return _PER_MASS * history.binary_fraction * _IMF_FRACTION


def test_immediate_mergers_follow_star_formation() -> None:
    """With no delay and one metallicity, R(z) is the efficiency times the star-formation rate.

    The bins hold the formed mass psi |dt/dz| dz over the bin's duration, which differ at second
    order in the bin width: 3e-4 for bins 0.05 wide.
    """
    rate = merger_rate_density(_single_metallicity(0.0), _HISTORY)
    centres = rate.redshift_centres
    expected = _efficiency(_HISTORY) * _madau_fragos(centres) * 1e9
    assert np.allclose(rate.rate, expected, rtol=1e-3, atol=0.0)


def test_every_binary_formed_long_enough_ago_has_merged() -> None:
    """Mergers per volume, summed over all time, are the efficiency times the mass formed over tau ago.

    Binaries form at bin centres, so the sum counts each formation bin whole or not at all: the bin
    that straddles tau ago is the bound.
    """
    tau = 1.0 * u.Gyr
    rate = merger_rate_density(_single_metallicity(tau.to(u.yr).value), _HISTORY)
    merger_durations = np.diff(rate.lookback_edges)[::-1]
    merged = np.sum(rate.rate / 1e9 * merger_durations)
    grid = np.linspace(z_at_value(_COSMOLOGY.lookback_time, tau).value, _HISTORY.max_redshift, 20001)
    years_per_redshift = 1.0 / (_COSMOLOGY.H(grid).to(1 / u.yr).value * (1.0 + grid))
    formed = np.trapezoid(_madau_fragos(grid) * years_per_redshift, grid)
    z_tau = grid[0]
    width = _HISTORY.max_redshift / (_HISTORY.n_redshift - 1)
    straddling_bin = _madau_fragos(z_tau) * years_per_redshift[0] * width
    assert abs(merged - _efficiency(_HISTORY) * formed) <= _efficiency(_HISTORY) * straddling_bin


def test_a_spread_of_delays_shifts_star_formation_in_time() -> None:
    """With delays uniform in [0.5, 1.5] Gyr, R(z) is the efficiency times psi averaged tau before z.

    The bins are 0.015 wide, so that the delays span many formation bins even at low redshift.
    """
    history = CosmicHistory()
    delays = np.linspace(0.5, 1.5, 400) * 1e9
    run = _run(list(delays), [0.01] * delays.size, evolved_mass=delays.size / _PER_MASS)
    rate = merger_rate_density(run, history)
    # at z = 3 the longest delay reaches back to z of about 9; by z = 4 it would pass z_max, where no binary forms
    for redshift in (0.3, 1.0, 2.0, 3.0):
        d = int(np.argmin(np.abs(rate.redshift_centres - redshift)))
        centre = rate.redshift_centres[d]
        merger_lookback = _COSMOLOGY.lookback_time(centre)
        formation = [
            z_at_value(_COSMOLOGY.lookback_time, merger_lookback + tau * u.Gyr).value
            for tau in np.linspace(0.5, 1.5, 41)
        ]
        expected = _efficiency(history) * np.mean(_madau_fragos(np.asarray(formation))) * 1e9
        assert rate.rate[d] == pytest.approx(expected, rel=0.01)


def test_metallicities_are_weighted_by_the_shifted_log_normal() -> None:
    """Two metallicities share each redshift's star formation at the log-midpoint, about the shifted mean."""
    history = _HISTORY
    low, high = 0.002, 0.02
    # three binaries at the low metallicity and one at the high, all merging at once
    run = _run([0.0] * 4, [low, low, low, high], evolved_mass=1.0 / _PER_MASS)
    rate = merger_rate_density(run, history)
    centres = rate.redshift_centres
    midpoint = 0.5 * (math.log10(low / history.solar_metallicity) + math.log10(high / history.solar_metallicity))
    spread = history.metallicity_spread
    mean = 0.153 - 0.074 * centres**1.34 - math.log(10.0) / 2.0 * spread**2
    below = 0.5 * (1.0 + np.array([math.erf((midpoint - m) / (spread * math.sqrt(2.0))) for m in mean]))
    factor = history.binary_fraction * _IMF_FRACTION * _PER_MASS
    expected = factor * _madau_fragos(centres) * 1e9 * (3.0 * below + 1.0 * (1.0 - below))
    assert np.allclose(rate.rate, expected, rtol=1e-3, atol=0.0)


def test_the_binary_fraction_and_imf_fraction_scale_the_rate() -> None:
    """The rate is proportional to each fraction, so only their product matters."""
    history = CosmicHistory(n_redshift=101)
    run = _single_metallicity(1e8)
    base = merger_rate_density(run, history).rate
    halved = merger_rate_density(run, CosmicHistory(n_redshift=101, binary_fraction=0.25)).rate
    assert np.allclose(halved, 0.5 * base, rtol=1e-12, atol=0.0)
    lighter = MergingBinaries(
        delay_time=run.delay_time,
        metallicity=run.metallicity,
        evolved_mass=run.evolved_mass,
        imf_fraction=_IMF_FRACTION / 2.0,
        properties=run.properties,
    )
    assert np.allclose(merger_rate_density(lighter, history).rate, 0.5 * base, rtol=1e-12, atol=0.0)


def test_a_weighted_binary_counts_as_that_fraction_of_one() -> None:
    """Weights 1 and 0.5 give the rate of two copies of the first and one of the second, from twice the mass."""
    history = CosmicHistory(n_redshift=101)
    weighted = _run([0.0, 3e9], [0.01, 0.01], evolved_mass=1e5, weight=[1.0, 0.5])
    copies = _run([0.0, 0.0, 3e9], [0.01, 0.01, 0.01], evolved_mass=2e5)
    assert np.allclose(
        merger_rate_density(weighted, history).rate, merger_rate_density(copies, history).rate, rtol=1e-12, atol=0.0
    )


def test_binaries_are_drawn_in_proportion_to_their_weights() -> None:
    """Of two otherwise identical binaries weighted 1 and 0.25, the first is drawn four times as often."""
    history = CosmicHistory(n_redshift=101)
    run = _run([0.0, 0.0], [0.01, 0.01], evolved_mass=2e5, weight=[1.0, 0.25])
    rate = merger_rate_density(run, history)
    labels = draw_mergers(run, rate, np.random.default_rng(5), history=history)["label"]
    first, second = np.sum(labels == 0), np.sum(labels == 1)
    assert first + second > 5000
    expected_share = 0.8
    share = first / (first + second)
    assert abs(share - expected_share) < 5.0 * np.sqrt(expected_share * (1 - expected_share) / (first + second))


def test_a_delay_longer_than_the_universe_gives_no_mergers() -> None:
    """A binary whose delay exceeds the age of the Universe at z_max never merges."""
    run = _single_metallicity(1.4e10)
    rate = merger_rate_density(run, _HISTORY)
    assert np.all(rate.rate == 0.0)
    assert expected_mergers(rate, _HISTORY) == 0.0
    mergers = draw_mergers(run, rate, np.random.default_rng(0), history=_HISTORY)
    assert set(mergers) == {"label", "redshift", "formation_redshift", "delay_time", "metallicity"}
    assert all(values.size == 0 for values in mergers.values())


def test_expected_mergers_integrate_the_rate_over_the_volume() -> None:
    """The expected count is R(z) dV_c/dz / (1 + z), integrated: checked for immediate mergers."""
    rate = merger_rate_density(_single_metallicity(0.0), _HISTORY)
    grid = np.linspace(0.0, _HISTORY.max_redshift, 20001)
    density = (
        _efficiency(_HISTORY)
        * _madau_fragos(grid)
        * 1e9
        * _COSMOLOGY.differential_comoving_volume(grid).to(u.Gpc**3 / u.sr).value
        * 4.0
        * np.pi
        / (1.0 + grid)
    )
    assert expected_mergers(rate, _HISTORY) == pytest.approx(np.trapezoid(density, grid), rel=1e-3)
    assert expected_mergers(rate, _HISTORY, years=2.5) == pytest.approx(2.5 * expected_mergers(rate, _HISTORY))


@pytest.fixture(scope="module")
def two_delays() -> tuple[MergingBinaries, object, dict[str, np.ndarray]]:
    """A run with prompt mergers at one metallicity and 5 Gyr delays at another, and a draw from it."""
    history = _HISTORY
    run = _run([1e6] * 5 + [5e9] * 5, [0.002] * 5 + [0.02] * 5, evolved_mass=5.0 / 2e-6)
    rate = merger_rate_density(run, history)
    return run, rate, draw_mergers(run, rate, np.random.default_rng(7), history=history)


def test_the_number_drawn_is_poisson_about_the_expectation(two_delays) -> None:
    """The draw holds the expected number of mergers, within five Poisson deviations."""
    _, rate, mergers = two_delays
    expected = expected_mergers(rate, _HISTORY)
    assert expected > 1000
    assert abs(mergers["redshift"].size - expected) < 5.0 * math.sqrt(expected)


def test_drawn_redshifts_follow_the_rate(two_delays) -> None:
    """Counts in coarse redshift bins agree with the rate's expectation, within five deviations."""
    _, rate, mergers = two_delays
    edges = rate.redshift_edges[::-1][::30]
    counts, _ = np.histogram(mergers["redshift"], edges)
    for low, high, count in zip(edges[:-1], edges[1:], counts, strict=True):
        inside = (rate.redshift_centres > low) & (rate.redshift_centres < high)
        sub = type(rate)(
            redshift_edges=rate.redshift_edges,
            rate=np.where(inside, rate.rate, 0.0),
            weights=rate.weights,
            metallicities=rate.metallicities,
            formation_lookback=rate.formation_lookback,
            lookback_edges=rate.lookback_edges,
        )
        expected = expected_mergers(sub, _HISTORY)
        assert abs(count - expected) < 5.0 * math.sqrt(expected) + 1.0


def test_a_merger_comes_only_from_binaries_that_can_merge_then(two_delays) -> None:
    """Where 5 Gyr before the merger precedes z_max, the merger is from the prompt metallicity."""
    _, _, mergers = two_delays
    too_early = (
        _COSMOLOGY.lookback_time(mergers["redshift"]).to(u.yr).value + 5e9
        > _COSMOLOGY.lookback_time(_HISTORY.max_redshift).to(u.yr).value
    )
    assert np.any(too_early)
    assert np.all(mergers["metallicity"][too_early] == 0.002)
    assert np.all(mergers["delay_time"][too_early] == 1e6)
    late = mergers["redshift"] < 0.5
    assert np.any(mergers["delay_time"][late] == 5e9)


def test_properties_follow_their_binary(two_delays) -> None:
    """Each merger carries the properties, delay and metallicity of one binary of the run."""
    run, _, mergers = two_delays
    labels = mergers["label"]
    assert np.array_equal(run.delay_time[labels], mergers["delay_time"])
    assert np.array_equal(run.metallicity[labels], mergers["metallicity"])


def test_formation_is_the_delay_before_the_merger(two_delays) -> None:
    """The formation redshift is where the lookback time is the merger's plus the delay."""
    _, _, mergers = two_delays
    merger_lookback = _COSMOLOGY.lookback_time(mergers["redshift"]).to(u.yr).value
    formation_lookback = _COSMOLOGY.lookback_time(mergers["formation_redshift"]).to(u.yr).value
    assert np.allclose(formation_lookback - merger_lookback, mergers["delay_time"], rtol=1e-3, atol=1e5)
    assert np.all(mergers["formation_redshift"] >= mergers["redshift"])


def test_a_seed_repeats_the_draw() -> None:
    """The same generator seed gives the same mergers."""
    run = _single_metallicity(1e8)
    rate = merger_rate_density(run, CosmicHistory(n_redshift=101))
    first = draw_mergers(run, rate, np.random.default_rng(3), history=CosmicHistory(n_redshift=101))
    second = draw_mergers(run, rate, np.random.default_rng(3), history=CosmicHistory(n_redshift=101))
    assert all(np.array_equal(first[name], second[name]) for name in first)


@pytest.mark.parametrize(
    ("delays", "metallicities", "mass", "match"),
    [
        ([], [], 1.0, "at least one"),
        ([1.0, 2.0], [0.01], 1.0, "do not match"),
        ([-1.0], [0.01], 1.0, "non-negative"),
        ([np.nan], [0.01], 1.0, "non-negative"),
        ([1.0], [0.0], 1.0, "positive"),
        ([1.0], [0.01], 0.0, "evolved mass"),
    ],
)
def test_an_inconsistent_run_is_refused(delays, metallicities, mass, match) -> None:
    """A run that is empty, misaligned, or carries a non-physical value is refused."""
    with pytest.raises(PopulationValidationError, match=match):
        _run(delays, metallicities, evolved_mass=mass) if delays else MergingBinaries(
            delay_time=np.asarray(delays, dtype=float),
            metallicity=np.asarray(metallicities, dtype=float),
            evolved_mass=mass,
            imf_fraction=_IMF_FRACTION,
            properties={},
        )


@pytest.mark.parametrize(
    ("imf_fraction", "weight", "match"),
    [
        (0.0, None, "IMF fraction"),
        (1.5, None, "IMF fraction"),
        (np.nan, None, "IMF fraction"),
        (0.3, [1.0], "do not match"),
        (0.3, [1.0, 0.0], "Weights"),
        (0.3, [1.0, np.inf], "Weights"),
    ],
)
def test_an_invalid_imf_fraction_or_weight_is_refused(imf_fraction, weight, match) -> None:
    """The IMF fraction lies in (0, 1]; there is one finite, positive weight per binary."""
    with pytest.raises(PopulationValidationError, match=match):
        MergingBinaries(
            delay_time=np.array([1.0, 2.0]),
            metallicity=np.array([0.01, 0.01]),
            evolved_mass=1.0,
            imf_fraction=imf_fraction,
            properties={},
            weight=None if weight is None else np.asarray(weight),
        )


def test_within_a_bin_redshifts_follow_the_observed_volume() -> None:
    """Within a bin the rate is constant, so redshifts follow dV_c/dz / (1 + z) there: bins 5 wide show it."""
    history = CosmicHistory(n_redshift=4)
    run = _single_metallicity(0.0)
    rate = merger_rate_density(run, history)
    redshift = draw_mergers(run, rate, np.random.default_rng(11), history=history)["redshift"]
    nearest = redshift[redshift < 5.0]
    grid = np.linspace(0.0, 5.0, 2001)
    density = _COSMOLOGY.differential_comoving_volume(grid).value / (1.0 + grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    empirical = np.searchsorted(np.sort(nearest), grid, side="right") / nearest.size
    assert nearest.size > 1000
    assert np.max(np.abs(empirical - cdf)) < 1.63 / np.sqrt(nearest.size)  # KS at 1 %


def test_a_formation_beyond_the_lookback_table_is_refused() -> None:
    """A formation redshift past twice z_max, where the table ends, is refused rather than clamped."""
    from gwmock_pop.synthesis.convolution import _formation_redshift

    history = CosmicHistory(max_redshift=1.0, n_redshift=11)
    assert _formation_redshift(np.array([0.5]), np.array([1e9]), history)[0] > 0.5
    with pytest.raises(PopulationValidationError, match="lookback table ends"):
        _formation_redshift(np.array([0.5]), np.array([1e10]), history)
