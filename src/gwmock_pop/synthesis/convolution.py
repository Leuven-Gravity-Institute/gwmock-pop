r"""Cosmic merger rate of a population-synthesis run, and mergers drawn from it.

A population-synthesis code evolves binaries at a few metallicities and reports, for each binary
that becomes a merging compact pair, its delay from formation to merger. This module turns such a
run into a merger-rate density :math:`R(z)` and into individual mergers, by the algorithm of
cosmoRate (Santoliquido et al. 2020, 2021; ``merger_rate_density`` and the ``SFR_pw`` catalogue
sampling in its ``cosmo_functions.py``, branch ``v2``), with that code's isolated-channel settings:

- The star-formation-rate density and mean metallicity are Madau & Fragos (2017):
  :math:`\psi(z) = 0.01\,(1+z)^{2.6} / [1 + ((1+z)/3.2)^{6.2}]` in
  :math:`M_\odot\,\mathrm{yr^{-1}\,Mpc^{-3}}`, and
  :math:`\langle\log_{10} Z/Z_\odot\rangle = 0.153 - 0.074\,z^{1.34}`.
- At each redshift :math:`\log_{10} Z/Z_\odot` is normal with spread :math:`\sigma_Z` about that
  mean less :math:`\ln(10)\,\sigma_Z^2/2`, as cosmoRate shifts it. Each simulated metallicity takes
  the probability between the midpoints (in :math:`\log Z`) to its neighbours; the lowest and
  highest take the tails.
- Redshift is cut into ``n_redshift - 1`` equal bins from ``max_redshift`` to 0 (by default 15
  and 1001 edges, so 0.015 wide; see below for why not cosmoRate's 100). Binaries form at the
  centre of each bin with mass :math:`\psi(z)\,|\mathrm{d}t/\mathrm{d}z|\,\Delta z` per comoving
  volume, and the binaries of metallicity :math:`Z` merge at the formation lookback time less their
  delay. Mergers with a negative lookback time, which have not happened yet, are dropped.
- Each simulated metallicity :math:`Z` contributes :math:`\eta_Z = N_Z / M \cdot f_\mathrm{bin}
  f_\mathrm{IMF}` merging binaries per solar mass of star formation. Here :math:`N_Z` is the number
  of merging binaries at :math:`Z`, :math:`M` the mass evolved there, :math:`f_\mathrm{bin}` the
  binary fraction (a property of the stars, in :class:`CosmicHistory`), and :math:`f_\mathrm{IMF}`
  the evolved mass's share of the full population it is drawn from (a property of the run's
  sampling, in :class:`MergingBinaries`).

The binary fraction defaults to 0.4, the value behind Iorio et al.'s (2023) published merger rates
(see :mod:`gwmock_pop.synthesis.iorio2023`). The metallicity relation is Madau & Fragos's fit to the mass-weighted mean of
:math:`Z/Z_\odot`, on the Anders & Grevesse (1989) solar scale. That is why the log-normal is shifted
down by :math:`\ln(10)\,\sigma_Z^2/2`, which keeps its mean at the fit, and why
:math:`Z_\odot = 0.019`.

Zenodo record 21682194 (Dupletsa et al. 2026) holds a population built from Iorio et al.'s merging
binary neutron stars (``simulation_F``, :math:`\alpha_\mathrm{CE} = 1`; Loffredo et al. 2025). It
has 351,205 mergers in one year, with a local rate of 107 Gpc^-3 yr^-1. From the same binaries, read
by :func:`gwmock_pop.synthesis.iorio2023.read_iorio2023`:

- The redshift distribution's largest difference from the record's is 0.005 in cumulative
  probability; two draws of this module differ from each other by 0.0015.
- cosmoRate's own 100 edges give 0.010 instead: a rate constant over bins 0.15 wide misplaces
  mergers around the peak at :math:`z \approx 2`. The difference stops falling at about 750 edges.
- With :func:`gwmock_pop.synthesis.iorio2023.read_iorio2023`'s normalisation and the defaults, the
  count is 350,400 mergers per year, 0.23 % below the record, and the rate at :math:`z = 0.0075` is
  106.0 Gpc^-3 yr^-1.

Two departures from cosmoRate, each deliberate:

- **Redshift grid.** 1001 edges by default, against cosmoRate's 100; see above.
- **Cosmology.** One flat Planck 2018 cosmology, with matter density 0.3111 (cold matter and massive
  neutrinos), gives both the lookback times and :math:`\mathrm{d}t/\mathrm{d}z`. cosmoRate takes its
  lookback times from astropy's ``Planck18``, which includes the neutrinos and radiation, but
  computes :math:`\mathrm{d}t/\mathrm{d}z` by hand from ``Planck18.Om0`` = 0.30966 with
  :math:`\Omega_\Lambda = 1 - \Omega_m`, leaving them out. The two disagree; here they cannot.
  The rate is 0.25 % lower than cosmoRate's for the same run, nearly uniformly in redshift.
  Substituting cosmoRate's two expressions makes the two codes agree bin for bin.

The cosmology is :mod:`gwmock_pop.cosmology.flat_lambda_cdm` at the Planck 2018 parameters.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from gwmock_pop.constants import PLANCK18_H0_KM_S_MPC, PLANCK18_OMEGA_M, SPEED_OF_LIGHT
from gwmock_pop.cosmology.flat_lambda_cdm import compute_normalized_hubble_parameter
from gwmock_pop.exceptions import PopulationValidationError

#: Cubic megaparsecs per cubic gigaparsec.
_MPC3_PER_GPC3 = 1e9
#: Probability the metallicity bins must hold at every formation redshift, as cosmoRate requires.
_MIN_METALLICITY_PROBABILITY = 0.98
#: Fine redshift points per bin when drawing a merger's redshift within its bin.
_POINTS_PER_BIN = 64
#: Kilometres per megaparsec (IAU 2015 parsec), and seconds per year (Julian).
_KM_PER_MPC = 3.0856775814913673e19
_SECONDS_PER_YEAR = 365.25 * 86400.0
#: Points of the cosmology table, from redshift 0 to twice the history's highest redshift. Its step,
#: 1e-4 at the default 15, puts the table's integration and interpolation errors below 1e-8.
_TABLE_POINTS = 300_001


@dataclass(frozen=True)
class MergingBinaries:
    r"""The merging compact binaries of one population-synthesis run.

    Attributes:
        delay_time: Time from the binary's formation to its merger, in years.
        metallicity: Absolute metallicity :math:`Z` the binary was evolved at.
        evolved_mass: Total initial mass, in solar masses, of the binaries evolved at each
            metallicity (the same for every metallicity).
        imf_fraction: :math:`f_\mathrm{IMF}`: ``evolved_mass`` divided by the mass of the full
            population the run is drawn from.
        properties: Further per-binary columns, carried to each merger drawn from the binary.
    """

    delay_time: np.ndarray
    metallicity: np.ndarray
    evolved_mass: float
    imf_fraction: float
    properties: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        """Refuse an empty run, misaligned columns, or a non-positive mass or delay.

        Raises:
            PopulationValidationError: If the run is empty or inconsistent.
        """
        n_rows = self.delay_time.shape[0]
        if n_rows == 0:
            raise PopulationValidationError("A population-synthesis run must hold at least one merging binary.")
        lengths = {"metallicity": self.metallicity.shape[0], **{k: v.shape[0] for k, v in self.properties.items()}}
        misaligned = {name: length for name, length in lengths.items() if length != n_rows}
        if misaligned:
            raise PopulationValidationError(f"Columns {misaligned} do not match the {n_rows} delay times.")
        if not np.all(np.isfinite(self.delay_time)) or np.any(self.delay_time < 0):
            raise PopulationValidationError("Delay times must be finite and non-negative.")
        if not np.all(self.metallicity > 0):
            raise PopulationValidationError("Metallicities must be positive.")
        if not (np.isfinite(self.evolved_mass) and self.evolved_mass > 0):
            raise PopulationValidationError("The evolved mass must be finite and positive.")
        if not (np.isfinite(self.imf_fraction) and 0 < self.imf_fraction <= 1):
            raise PopulationValidationError("The IMF fraction must lie in (0, 1].")


@dataclass(frozen=True)
class CosmicHistory:
    r"""Star formation, metallicity and cosmology a run is convolved with.

    Attributes:
        metallicity_spread: :math:`\sigma_Z`, the spread of :math:`\log_{10} Z` at a redshift.
        solar_metallicity: :math:`Z_\odot` (cosmoRate's 0.019).
        binary_fraction: :math:`f_\mathrm{bin}`, the fraction of the stellar mass formed in binaries;
            0.4 by default, as in Iorio et al.'s published rates.
        max_redshift: Highest formation redshift.
        n_redshift: Number of redshift bin edges, from ``max_redshift`` to 0.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
    """

    metallicity_spread: float = 0.2
    solar_metallicity: float = 0.019
    binary_fraction: float = 0.4
    max_redshift: float = 15.0
    n_redshift: int = 1001
    hubble_constant: float = PLANCK18_H0_KM_S_MPC
    omega_m: float = PLANCK18_OMEGA_M

    def star_formation_rate(self, redshift: np.ndarray) -> np.ndarray:
        """Madau & Fragos (2017) star-formation-rate density, in Msun / yr / Mpc^3.

        Args:
            redshift: Redshift.

        Returns:
            The star-formation-rate density.
        """
        return 0.01 * (1.0 + redshift) ** 2.6 / (1.0 + ((1.0 + redshift) / 3.2) ** 6.2)

    def mean_log_metallicity(self, redshift: np.ndarray) -> np.ndarray:
        """Mean of the normal in log10(Z / Zsun) at a redshift, with cosmoRate's shift.

        Args:
            redshift: Redshift.

        Returns:
            The mean, in log10 of solar units.
        """
        return 0.153 - 0.074 * redshift**1.34 - np.log(10.0) / 2.0 * self.metallicity_spread**2


@dataclass(frozen=True)
class MergerRate:
    """A run's merger-rate density on its redshift bins.

    Attributes:
        redshift_edges: Bin edges, decreasing from the history's ``max_redshift`` to 0.
        rate: Merger-rate density in each merger bin, in Gpc^-3 yr^-1 (source frame).
        weights: ``weights[d, i, m]`` is the part of ``rate[d]`` from binaries formed in bin ``i``
            at the ``m``-th metallicity of ``metallicities``, in Mpc^-3 yr^-1.
        metallicities: The run's distinct metallicities, ascending.
        formation_lookback: Lookback time, in years, at the centre of each formation bin.
        lookback_edges: Lookback time, in years, at each redshift edge, increasing.
    """

    redshift_edges: np.ndarray
    rate: np.ndarray
    weights: np.ndarray
    metallicities: np.ndarray
    formation_lookback: np.ndarray
    lookback_edges: np.ndarray

    @property
    def redshift_centres(self) -> np.ndarray:
        """Centre of each merger bin."""
        return 0.5 * (self.redshift_edges[1:] + self.redshift_edges[:-1])


@functools.cache
def _cosmology_table(history: CosmicHistory) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Redshift, lookback time (yr), comoving distance (Mpc) and E(z), from 0 to twice z_max.

    One cumulative trapezoid of 1 / E(z) and 1 / ((1 + z) E(z)) on a uniform grid, so a call costs a
    few megabytes, and a history computes its table once.
    """
    redshift = np.linspace(0.0, 2.0 * history.max_redshift, _TABLE_POINTS)
    e_z = np.asarray(compute_normalized_hubble_parameter(jnp.asarray(redshift), history.omega_m))

    def cumulative(integrand: np.ndarray) -> np.ndarray:
        steps = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(redshift)
        return np.concatenate([[0.0], np.cumsum(steps)])

    hubble_time_yr = _KM_PER_MPC / history.hubble_constant / _SECONDS_PER_YEAR
    hubble_distance_mpc = SPEED_OF_LIGHT / 1000.0 / history.hubble_constant
    lookback = hubble_time_yr * cumulative(1.0 / ((1.0 + redshift) * e_z))
    comoving = hubble_distance_mpc * cumulative(1.0 / e_z)
    return redshift, lookback, comoving, e_z


def _lookback_years(redshift: np.ndarray, history: CosmicHistory) -> np.ndarray:
    table_redshift, lookback, _, _ = _cosmology_table(history)
    return np.interp(redshift, table_redshift, lookback)


def _years_per_unit_redshift(redshift: np.ndarray, history: CosmicHistory) -> np.ndarray:
    hubble_time_yr = _KM_PER_MPC / history.hubble_constant / _SECONDS_PER_YEAR
    e_z = np.asarray(compute_normalized_hubble_parameter(jnp.asarray(redshift), history.omega_m))
    return hubble_time_yr / ((1.0 + redshift) * e_z)


def _metallicity_probabilities(
    formation_redshift: float, metallicities: np.ndarray, edges: np.ndarray, history: CosmicHistory
) -> np.ndarray:
    scaled = (edges - history.mean_log_metallicity(formation_redshift)) / (history.metallicity_spread * np.sqrt(2.0))
    cdf = 0.5 * (1.0 + np.asarray(erf(jnp.asarray(scaled))))
    probabilities = np.diff(cdf)
    if probabilities.sum() < _MIN_METALLICITY_PROBABILITY:
        raise PopulationValidationError(
            f"The metallicity bins hold {probabilities.sum():.3f} of the probability at z = {formation_redshift:.3f}."
        )
    return probabilities


def merger_rate_density(binaries: MergingBinaries, history: CosmicHistory | None = None) -> MergerRate:
    """Convolve a population-synthesis run with the cosmic star-formation history.

    Args:
        binaries: The run's merging binaries.
        history: Star formation, metallicity and cosmology; cosmoRate's isolated-channel settings
            when omitted.

    Returns:
        The merger-rate density on the history's redshift bins.
    """
    history = history or CosmicHistory()
    edges = np.linspace(history.max_redshift, 0.0, history.n_redshift)
    n_bins = edges.size - 1
    widths = edges[:-1] - edges[1:]
    formation_redshift = edges[:-1] - widths / 2.0
    lookback_edges = np.sort(_lookback_years(edges, history))
    # merger bin d spans lookback_edges[n_bins - d - 1] to lookback_edges[n_bins - d]
    merger_widths = (lookback_edges[1:] - lookback_edges[:-1])[::-1]
    formation_lookback = _lookback_years(formation_redshift, history)
    formed_mass = (
        history.star_formation_rate(formation_redshift) * _years_per_unit_redshift(formation_redshift, history) * widths
    )

    metallicities = np.unique(binaries.metallicity)
    log_solar = np.log10(metallicities) - np.log10(history.solar_metallicity)
    mean_high = history.mean_log_metallicity(edges[-1])
    mean_low = history.mean_log_metallicity(edges[0])
    probability_edges = np.concatenate(
        [
            [mean_low - 1000.0 * history.metallicity_spread],
            0.5 * (log_solar[1:] + log_solar[:-1]),
            [mean_high + 1000.0 * history.metallicity_spread],
        ]
    )
    delays = [binaries.delay_time[binaries.metallicity == z] for z in metallicities]
    efficiency = (
        np.array([d.size for d in delays]) / binaries.evolved_mass * history.binary_fraction * binaries.imf_fraction
    )

    weights = np.zeros((n_bins, n_bins, metallicities.size))
    for i in range(n_bins):
        probabilities = _metallicity_probabilities(formation_redshift[i], metallicities, probability_edges, history)
        for m, delay in enumerate(delays):
            counts, _ = np.histogram(formation_lookback[i] - delay, bins=lookback_edges)
            weights[:, i, m] = (
                counts[::-1] / delay.size * efficiency[m] * probabilities[m] * formed_mass[i] / merger_widths
            )
    rate = weights.sum(axis=(1, 2)) * _MPC3_PER_GPC3
    return MergerRate(
        redshift_edges=edges,
        rate=rate,
        weights=weights,
        metallicities=metallicities,
        formation_lookback=formation_lookback,
        lookback_edges=lookback_edges,
    )


def _observed_rate_density(redshift: np.ndarray, history: CosmicHistory) -> np.ndarray:
    """dV_c/dz / (1 + z), in Gpc^3: mergers per observer year per unit z per unit source-frame rate."""
    table_redshift, _, comoving, _ = _cosmology_table(history)
    distance = np.interp(redshift, table_redshift, comoving)
    e_z = np.asarray(compute_normalized_hubble_parameter(jnp.asarray(redshift), history.omega_m))
    hubble_distance_mpc = SPEED_OF_LIGHT / 1000.0 / history.hubble_constant
    volume_mpc3 = 4.0 * np.pi * hubble_distance_mpc * distance**2 / e_z
    return volume_mpc3 / _MPC3_PER_GPC3 / (1.0 + redshift)


def expected_mergers(rate: MergerRate, history: CosmicHistory | None = None, years: float = 1.0) -> float:
    """Mergers an observer sees in ``years`` of detector-frame time, over all redshifts.

    The rate is constant within each bin, as the convolution gives it.

    Args:
        rate: The merger-rate density.
        history: The cosmology the rate was computed with.
        years: Observing time, in years.

    Returns:
        The expected number of mergers.
    """
    history = history or CosmicHistory()
    return float(np.sum(rate.rate * _bin_volumes(rate.redshift_edges, history)) * years)


def _fine_grid(edges: np.ndarray) -> np.ndarray:
    """Fine redshift points within each bin: shape (n_bins, _POINTS_PER_BIN), each row decreasing."""
    fractions = np.linspace(0.0, 1.0, _POINTS_PER_BIN)
    return edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * fractions[None, :]


def _bin_volumes(edges: np.ndarray, history: CosmicHistory) -> np.ndarray:
    """Integral of dV_c/dz / (1 + z) over each bin, in Gpc^3."""
    fine = _fine_grid(edges)
    density = _observed_rate_density(fine.ravel(), history).reshape(fine.shape)
    return np.trapezoid(density[:, ::-1], fine[:, ::-1], axis=1)


def _lookback_bin(lookback: np.ndarray, lookback_edges: np.ndarray) -> np.ndarray:
    """Index of the lookback bin each time falls in, as :func:`numpy.histogram` bins it, or -1 outside."""
    index = np.searchsorted(lookback_edges, lookback, side="right") - 1
    index[lookback == lookback_edges[-1]] = lookback_edges.size - 2
    index[(lookback < lookback_edges[0]) | (lookback > lookback_edges[-1])] = -1
    return index


def _formation_redshift(redshift: np.ndarray, delay_time: np.ndarray, history: CosmicHistory) -> np.ndarray:
    """Redshift at which a binary merging at ``redshift`` after ``delay_time`` years formed.

    Raises:
        PopulationValidationError: If a formation would lie beyond twice the history's highest
            redshift, where the lookback table ends.
    """
    grid, lookback, _, _ = _cosmology_table(history)
    formation_lookback = np.interp(redshift, grid, lookback) + delay_time
    if np.any(formation_lookback > lookback[-1]):
        raise PopulationValidationError(f"A merger formed beyond z = {grid[-1]}, where the lookback table ends.")
    return np.interp(formation_lookback, lookback, grid)


def draw_mergers(
    binaries: MergingBinaries,
    rate: MergerRate,
    rng: np.random.Generator,
    *,
    history: CosmicHistory | None = None,
    years: float = 1.0,
) -> dict[str, np.ndarray]:
    """Draw the mergers an observer sees in ``years``, one row per merger.

    The number is Poisson about :func:`expected_mergers`. Each merger's bin is drawn by its share
    of that number, and its redshift within the bin in proportion to dV_c/dz / (1 + z). Then, as
    cosmoRate's ``SFR_pw`` sampling does, its formation bin and metallicity are drawn by their
    part of the bin's rate, and its binary from those of that metallicity whose merger falls in the
    bin, uniformly.

    Args:
        binaries: The run the rate was computed from.
        rate: :func:`merger_rate_density` of ``binaries``.
        rng: Random-number generator.
        history: The history ``rate`` was computed with.
        years: Observing time, in years.

    Returns:
        The binaries' ``properties`` for each merger, with ``redshift``, ``delay_time`` (years),
        ``metallicity``, and ``formation_redshift``: the redshift ``delay_time`` before the merger.
        It is not the formation bin's centre: the merger's redshift is drawn within its bin, and a
        delay short against the bin would otherwise put the formation after the merger.
    """
    history = history or CosmicHistory()
    edges = rate.redshift_edges
    per_bin = rate.rate * _bin_volumes(edges, history) * years
    total = per_bin.sum()
    n_mergers = rng.poisson(total)
    # a run none of whose binaries merges by z = 0 has no rate to share the (zero) mergers out by
    bins = rng.choice(per_bin.size, size=n_mergers, p=per_bin / total) if n_mergers else np.empty(0, dtype=np.int64)

    fine = _fine_grid(edges)
    density = _observed_rate_density(fine.ravel(), history).reshape(fine.shape)
    rows = np.empty(n_mergers, dtype=np.int64)
    redshift = np.empty(n_mergers)
    metallicity_index = np.empty(n_mergers, dtype=np.int64)
    by_metallicity = [np.flatnonzero(binaries.metallicity == z) for z in rate.metallicities]
    n_metallicities = rate.metallicities.size
    for d in np.unique(bins):
        members = np.flatnonzero(bins == d)
        z_fine, w_fine = fine[d, ::-1], density[d, ::-1]
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w_fine[1:] + w_fine[:-1]) * np.diff(z_fine))])
        redshift[members] = np.interp(rng.uniform(0.0, cdf[-1], members.size), cdf, z_fine)

        share = rate.weights[d].ravel()
        cells = rng.choice(share.size, size=members.size, p=share / share.sum())
        metallicity_bin = cells % n_metallicities
        lookback_bin = edges.size - 2 - d
        for cell in np.unique(cells):
            chosen = members[cells == cell]
            i, m = divmod(int(cell), n_metallicities)
            candidates = by_metallicity[m]
            in_bin = _lookback_bin(rate.formation_lookback[i] - binaries.delay_time[candidates], rate.lookback_edges)
            rows[chosen] = rng.choice(candidates[in_bin == lookback_bin], size=chosen.size)
        metallicity_index[members] = metallicity_bin

    columns = {name: values[rows] for name, values in binaries.properties.items()}
    columns["redshift"] = redshift
    columns["formation_redshift"] = _formation_redshift(redshift, binaries.delay_time[rows], history)
    columns["delay_time"] = binaries.delay_time[rows]
    columns["metallicity"] = rate.metallicities[metallicity_index]
    return columns
