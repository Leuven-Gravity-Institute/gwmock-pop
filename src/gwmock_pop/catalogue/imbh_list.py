r"""Generate intermediate-mass black-hole binary injection lists.

The lists have the same per-event format as the regenerated Einstein Telescope
Mock Data Challenge CBC lists (:mod:`gwmock_pop.catalogue.mdc_list`), and the
arrival process and the angle draws are the same code, so an IMBH list can be
merged with a CBC list event by event. What differs is where the intrinsic
parameters come from: they are drawn from a parametric population here rather
than read from a catalogue.

**Population.** The source-frame total mass ``M`` is log-uniform on
``[min_total_mass, max_total_mass]``. The mass ratio ``q = m1 / m2 >= 1`` is
uniform on ``[min_mass_ratio, max_mass_ratio]``; an optional intermediate-mass-
ratio-inspiral (IMRI) sub-population, a fraction ``imri_fraction`` of the
events, takes ``q`` uniform on ``[imri_min_mass_ratio, imri_max_mass_ratio]``
instead. Spin magnitudes are uniform on ``[0, max_spin]`` and spin directions
are isotropic. The source-frame merger-rate density is
``R(z) = local_rate * psi(z)``, where ``psi`` is the Madau-like star-formation
shape with ``psi(0) = 1`` (:func:`~gwmock_pop.distributions.madau_dickinson.
madau_dickinson_rate`), up to ``max_redshift``.

**Detector-frame mass cap.** Events with ``(1 + z) M`` above
``max_detector_frame_total_mass`` are excluded from the population, so that the
merger and ringdown stay above the low-frequency cutoff of the data. The cap
removes events rather than moving them, so it lowers the event rate. At redshift
``z`` the mass is drawn log-uniformly on ``[min_total_mass, M_hi(z)]`` with
``M_hi(z) = min(max_total_mass, max_detector_frame_total_mass / (1 + z))``, and
the redshift density carries the fraction of the log-uniform mass range that
survives the cap:

.. math::

    \frac{\mathrm{d}\dot N}{\mathrm{d}z} = R_0\,\frac{\psi(z)}{1+z}\,
        \frac{\mathrm{d}V_c}{\mathrm{d}z}\,F(z),
    \qquad
    F(z) = \frac{\ln[M_\mathrm{hi}(z) / M_\mathrm{min}]}{\ln(M_\mathrm{max} / M_\mathrm{min})},

clipped to ``[0, 1]``. The total observed-frame rate ``N_dot`` is the integral
of this density over ``[0, max_redshift]``, and the mean waiting time between
mergers is ``1 / N_dot``. The factor ``1 / (1 + z)`` converts the source-frame
rate to the observer's clock.

**Arrival process and draws.** Each event ``k`` (from 1) has its own Mersenne
Twister generator seeded with ``seed + k``, and it draws its uniforms as 32-bit
words, exactly as in the CBC lists. The uniforms are used in this fixed order:
the waiting time since the previous merger (exponential with mean ``1 / N_dot``),
the sub-population, the redshift (inverse CDF of the density above), the total
mass, the mass ratio, then for each component the spin magnitude, the cosine of
the tilt and the spin azimuth, and finally right ascension, the cosine of the
polar angle, the cosine of the inclination, the polarisation angle and the
coalescence phase. The signal bounds, listing rule and segment assignment are
those of the CBC lists.
"""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from gwmock_pop.catalogue.mdc_list import (
    _LAL_MSUN_SI,
    _POPULATION_COLUMNS,
    _angles_from_uniforms,
    _chirp_time_bound,
    _event_uniforms,
    _segment_range,
    _signal_bounds,
    _waiting_time,
)
from gwmock_pop.constants import (
    MADAU_DICKINSON_GAMMA,
    MADAU_DICKINSON_KAPPA,
    MADAU_DICKINSON_Z_PEAK,
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
    SPEED_OF_LIGHT,
)
from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.provenance import build_provenance_record, run_metadata, simulator_origin

#: Class code written in the lists for an IMBH binary from the main population.
#: Codes 1-3 are the CBC classes of :mod:`gwmock_pop.catalogue.mdc_list`.
IMBH_TYPE = 4

#: Class code written in the lists for an event of the IMRI sub-population.
IMRI_TYPE = 5

#: Source-type routing key of the population files. Both classes are binary
#: black holes as far as the signal simulators are concerned.
IMBH_ROUTING_KEY = "bbh"

#: Seconds in a Julian year, the year of the rate density's unit.
JULIAN_YEAR_SECONDS = 365.25 * 86400.0

#: Uniforms an event consumes: waiting time, sub-population, redshift, total
#: mass, mass ratio, three per spin and five extrinsic angles.
_DRAWS_PER_EVENT = 16
_ANGLE_DRAWS = slice(11, 16)

#: Redshift tabulation points of the rate density and its inverse CDF.
_REDSHIFT_GRID_SIZE = 16384

_MPC3_PER_GPC3 = 1e9


@dataclass(frozen=True, slots=True)
class ImbhPopulation:
    """Astrophysical model of the IMBH binaries.

    Attributes:
        min_total_mass: Lower bound of the source-frame total mass, in solar masses.
        max_total_mass: Upper bound of the source-frame total mass, in solar masses.
        min_mass_ratio: Lower bound of ``m1 / m2`` in the main population.
        max_mass_ratio: Upper bound of ``m1 / m2`` in the main population.
        imri_fraction: Fraction of the events drawn from the IMRI sub-population;
            ``0`` switches it off.
        imri_min_mass_ratio: Lower bound of ``m1 / m2`` in the IMRI sub-population.
        imri_max_mass_ratio: Upper bound of ``m1 / m2`` in the IMRI sub-population.
        max_spin: Upper bound of the dimensionless spin magnitudes.
        local_rate: Source-frame merger-rate density at ``z = 0`` of the whole
            population (both sub-populations, the full mass range before the
            detector-frame cap), in ``Gpc^-3 yr^-1``.
        max_redshift: Largest source redshift.
        max_detector_frame_total_mass: Largest ``(1 + z) M`` kept, in solar masses.
        gamma: Low-redshift slope of the star-formation shape.
        kappa: High-redshift fall-off of the star-formation shape.
        z_peak: Redshift of the star-formation peak.
        hubble_constant: Hubble constant, in km/s/Mpc.
        omega_m: Matter density of the flat Lambda-CDM cosmology.
        waveform_approximant: Waveform model the population is meant to be
            simulated with. It is recorded, not used, by the generator.
    """

    min_total_mass: float = 100.0
    max_total_mass: float = 1000.0
    min_mass_ratio: float = 1.0
    max_mass_ratio: float = 10.0
    imri_fraction: float = 0.0
    imri_min_mass_ratio: float = 10.0
    imri_max_mass_ratio: float = 100.0
    max_spin: float = 0.99
    local_rate: float = 0.13
    max_redshift: float = 10.0
    max_detector_frame_total_mass: float = 3000.0
    gamma: float = MADAU_DICKINSON_GAMMA
    kappa: float = MADAU_DICKINSON_KAPPA
    z_peak: float = MADAU_DICKINSON_Z_PEAK
    hubble_constant: float = PLANCK18_H0_KM_S_MPC
    omega_m: float = PLANCK18_OMEGA_M
    waveform_approximant: str = "IMRPhenomXPHM"

    def validate(self) -> None:
        """Refuse a model that cannot describe a population.

        Raises:
            PopulationValidationError: If a parameter is out of range.
        """
        positive = (
            "min_total_mass",
            "max_total_mass",
            "local_rate",
            "max_redshift",
            "max_detector_frame_total_mass",
            "hubble_constant",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise PopulationValidationError(f"{name} must be finite and positive, got {value!r}.")
        if not self.min_total_mass < self.max_total_mass:
            raise PopulationValidationError("min_total_mass must be below max_total_mass.")
        for low, high in (("min_mass_ratio", "max_mass_ratio"), ("imri_min_mass_ratio", "imri_max_mass_ratio")):
            low_value, high_value = getattr(self, low), getattr(self, high)
            if not (math.isfinite(low_value) and math.isfinite(high_value) and 1.0 <= low_value <= high_value):
                raise PopulationValidationError(
                    f"{low} and {high} must be finite with 1 <= {low} <= {high}, got {low_value!r} and {high_value!r}."
                )
        for name, upper in (("imri_fraction", 1.0), ("max_spin", 1.0), ("omega_m", 1.0)):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= upper:
                raise PopulationValidationError(f"{name} must lie in [0, {upper}], got {value!r}.")
        if self.max_detector_frame_total_mass < self.min_total_mass:
            raise PopulationValidationError(
                "max_detector_frame_total_mass must be at least min_total_mass, or no event survives the cap."
            )

    @property
    def largest_mass_ratio(self) -> float:
        """Return the largest mass ratio any event can have.

        Returns:
            ``imri_max_mass_ratio`` when the IMRI sub-population is on, else ``max_mass_ratio``.
        """
        return max(self.max_mass_ratio, self.imri_max_mass_ratio) if self.imri_fraction > 0.0 else self.max_mass_ratio


@dataclass(frozen=True, slots=True)
class ImbhListSettings:
    """Settings of one list generation: the data span, the seed and ``f_min``.

    Attributes:
        start_time: GPS time of the start of the data, in seconds.
        segment_duration: Duration of one data segment, in seconds.
        n_segments: Number of data segments.
        seed: Base seed; event ``k`` is generated with ``seed + k``.
        f_min: Lower frequency of the chirp-duration bound, in hertz.
    """

    start_time: float
    segment_duration: float
    n_segments: int
    seed: int = 100
    f_min: float = 5.0

    @property
    def stop_time(self) -> float:
        """Return the GPS time of the end of the data.

        Returns:
            ``start_time + n_segments * segment_duration``.
        """
        return self.start_time + self.n_segments * self.segment_duration

    def validate(self) -> None:
        """Refuse settings that cannot describe a run.

        Raises:
            PopulationValidationError: If a setting is out of range.
        """
        for name in ("seed", "n_segments"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise PopulationValidationError(f"{name} must be an integer, got {value!r}.")
        if self.n_segments < 1:
            raise PopulationValidationError(f"n_segments must be at least 1, got {self.n_segments}.")
        for name in ("segment_duration", "f_min"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise PopulationValidationError(f"{name} must be finite and positive, got {value!r}.")
        if not math.isfinite(self.start_time):
            raise PopulationValidationError(f"start_time must be finite, got {self.start_time!r}.")


@dataclass(frozen=True, slots=True)
class ImbhRate:
    """The rate integral of a population and the tables it is drawn from.

    Attributes:
        redshift: Redshift grid, from 0 to ``max_redshift``.
        rate_density: ``dN_dot/dz`` on the grid, in events per year per unit redshift.
        cumulative: Normalised cumulative distribution of the redshift on the grid.
        luminosity_distance: Luminosity distance on the grid, in Mpc.
        uncapped_events_per_year: Observed-frame rate without the detector-frame cap.
        events_per_year: Observed-frame rate of the population, after the cap.
    """

    redshift: np.ndarray = field(repr=False)
    rate_density: np.ndarray = field(repr=False)
    cumulative: np.ndarray = field(repr=False)
    luminosity_distance: np.ndarray = field(repr=False)
    uncapped_events_per_year: float
    events_per_year: float

    @property
    def mean_interval(self) -> float:
        """Return the mean waiting time between mergers, in seconds.

        Returns:
            One Julian year divided by :attr:`events_per_year`.
        """
        return JULIAN_YEAR_SECONDS / self.events_per_year


def _mass_upper_bound(population: ImbhPopulation, redshift: np.ndarray) -> np.ndarray:
    """Return the largest source-frame total mass kept at each redshift."""
    return np.minimum(population.max_total_mass, population.max_detector_frame_total_mass / (1.0 + redshift))


def surviving_mass_fraction(population: ImbhPopulation, redshift: np.ndarray) -> np.ndarray:
    """Return the fraction of the log-uniform mass range below the detector-frame cap.

    Args:
        population: The population model.
        redshift: Redshift(s).

    Returns:
        ``F(z)`` in ``[0, 1]``.
    """
    upper = _mass_upper_bound(population, np.asarray(redshift, dtype=float))
    fraction = np.log(np.maximum(upper, population.min_total_mass) / population.min_total_mass) / math.log(
        population.max_total_mass / population.min_total_mass
    )
    return np.clip(fraction, 0.0, 1.0)


def _trapezoid_cumulative(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return the cumulative trapezoid integral of ``values`` over ``grid``, starting at zero."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))))


def imbh_rate(population: ImbhPopulation) -> ImbhRate:
    """Integrate the population's merger rate over redshift.

    The comoving volume element and the luminosity distance come from one
    distance table, so the rate and the distances written to the lists share a
    single cosmology.

    Args:
        population: The population model.

    Returns:
        The rate integral, with and without the detector-frame cap, and the
        redshift tables the events are drawn from.

    Raises:
        PopulationValidationError: If the model is out of range.
    """
    # The tables are JAX functions; importing them here keeps JAX out of the
    # import closure of the package and its command line.
    from gwmock_pop.cosmology.flat_lambda_cdm import (  # noqa: PLC0415
        build_distance_lookup,
        compute_normalized_hubble_parameter,
    )
    from gwmock_pop.distributions.madau_dickinson import madau_dickinson_rate  # noqa: PLC0415

    population.validate()
    redshift, comoving_distance, luminosity_distance = (
        np.asarray(table, dtype=float)
        for table in build_distance_lookup(
            hubble_constant=population.hubble_constant,
            omega_m=population.omega_m,
            max_redshift=population.max_redshift,
            n_grid=_REDSHIFT_GRID_SIZE,
        )
    )
    hubble_distance = SPEED_OF_LIGHT / 1000.0 / population.hubble_constant
    efunc = np.asarray(compute_normalized_hubble_parameter(redshift=redshift, omega_m=population.omega_m))
    # dV_c/dz over the full sky, in Gpc^3 per unit redshift.
    volume_element = 4.0 * np.pi * hubble_distance * comoving_distance**2 / efunc / _MPC3_PER_GPC3
    shape = np.asarray(
        madau_dickinson_rate(redshift, gamma=population.gamma, kappa=population.kappa, z_peak=population.z_peak)
    )
    uncapped = population.local_rate * shape / (1.0 + redshift) * volume_element
    rate_density = uncapped * surviving_mass_fraction(population, redshift)
    cumulative = _trapezoid_cumulative(rate_density, redshift)
    events_per_year = float(cumulative[-1])
    return ImbhRate(
        redshift=redshift,
        rate_density=rate_density,
        cumulative=cumulative / events_per_year,
        luminosity_distance=luminosity_distance,
        uncapped_events_per_year=float(_trapezoid_cumulative(uncapped, redshift)[-1]),
        events_per_year=events_per_year,
    )


def _draw_schedule(
    settings: ImbhListSettings, population: ImbhPopulation, mean_interval: float
) -> tuple[np.ndarray, np.ndarray]:
    """Draw every event's coalescence time and uniforms.

    Mergers are drawn while the previous coalescence time is no later than the
    end of the data plus the longest chirp the population can produce: the
    lightest total mass at the largest mass ratio and the largest spin, at
    ``z = 0``.

    Returns:
        Coalescence times and the per-event uniforms.
    """
    q = population.largest_mass_ratio
    light = _LAL_MSUN_SI * population.min_total_mass
    longest_chirp = float(
        _chirp_time_bound(
            settings.f_min,
            np.asarray(light * q / (1.0 + q)),
            np.asarray(light / (1.0 + q)),
            np.asarray(population.max_spin),
        )
    )
    horizon = settings.stop_time + longest_chirp

    generator = np.random.RandomState()
    coa_times: list[float] = []
    uniforms: list[np.ndarray] = []
    coa_time = float(settings.start_time)
    k = 0
    while coa_time <= horizon:
        k += 1
        draws = _event_uniforms(generator, int(settings.seed) + k, _DRAWS_PER_EVENT)
        coa_time += _waiting_time(mean_interval, draws[0])
        coa_times.append(coa_time)
        uniforms.append(draws)
    return np.asarray(coa_times), np.asarray(uniforms).reshape(-1, _DRAWS_PER_EVENT)


def _intrinsic_columns(uniforms: np.ndarray, population: ImbhPopulation, rate: ImbhRate) -> dict[str, np.ndarray]:
    """Draw each event's class, redshift, masses and spins from its uniforms.

    Returns:
        The columns, one entry per drawn event.
    """
    imri = uniforms[:, 1] < population.imri_fraction
    redshift = np.interp(uniforms[:, 2], rate.cumulative, rate.redshift)
    upper = np.maximum(_mass_upper_bound(population, redshift), population.min_total_mass)
    total_mass = population.min_total_mass * (upper / population.min_total_mass) ** uniforms[:, 3]
    q_low = np.where(imri, population.imri_min_mass_ratio, population.min_mass_ratio)
    q_high = np.where(imri, population.imri_max_mass_ratio, population.max_mass_ratio)
    mass_ratio = q_low + (q_high - q_low) * uniforms[:, 4]
    columns = {
        "source_type": np.where(imri, IMRI_TYPE, IMBH_TYPE).astype(np.int64),
        "redshift": redshift,
        "distance": np.interp(redshift, rate.redshift, rate.luminosity_distance),
        "source_frame_mass_1": total_mass * mass_ratio / (1.0 + mass_ratio),
        "source_frame_mass_2": total_mass / (1.0 + mass_ratio),
        "lambda_1": np.zeros(len(redshift)),
        "lambda_2": np.zeros(len(redshift)),
    }
    for index, first in ((1, 5), (2, 8)):
        magnitude = population.max_spin * uniforms[:, first]
        cos_tilt = 2.0 * uniforms[:, first + 1] - 1.0
        sin_tilt = np.sqrt(1.0 - cos_tilt**2)
        azimuth = 2.0 * np.pi * uniforms[:, first + 2]
        columns[f"spin_{index}"] = magnitude
        columns[f"spin_{index}x"] = magnitude * sin_tilt * np.cos(azimuth)
        columns[f"spin_{index}y"] = magnitude * sin_tilt * np.sin(azimuth)
        columns[f"spin_{index}z"] = magnitude * cos_tilt
    return columns


@dataclass(frozen=True, slots=True)
class ImbhList:
    """One generated IMBH injection list.

    Attributes:
        settings: Settings the list was generated with.
        population: The population model.
        rate: The rate integral the arrival process was set from.
        columns: One 1-D array per column, one entry per listed event, in event
            order, with the same names as :attr:`gwmock_pop.catalogue.mdc_list.
            MdcCbcList.columns`. ``source_type`` is :data:`IMBH_TYPE` or
            :data:`IMRI_TYPE`.
        n_events: Events drawn, including those whose signal starts after the
            end of the data and so are not listed.
    """

    settings: ImbhListSettings
    population: ImbhPopulation
    rate: ImbhRate
    columns: dict[str, np.ndarray] = field(repr=False)
    n_events: int

    @property
    def n_listed(self) -> int:
        """Return the number of listed events.

        Returns:
            The length of every column.
        """
        return int(self.columns["event_id"].shape[0])

    def in_segment(self, segment: int) -> np.ndarray:
        """Return which listed events belong to a data segment.

        Args:
            segment: Zero-based segment index.

        Returns:
            Boolean mask over the listed events.
        """
        return (self.columns["first_segment"] <= segment) & (segment < self.columns["stop_segment"])

    def population_columns(self) -> dict[str, np.ndarray]:
        """Return the listed events as a canonical CBC population.

        Returns:
            Mapping from canonical parameter name to a 1-D array, in event order,
            for :data:`IMBH_ROUTING_KEY`. Both sub-populations are included; the
            class codes stay on :attr:`columns`.
        """
        return {name: self.columns[name] for name in _POPULATION_COLUMNS}

    def provenance(self, *, file_format: str) -> dict[str, Any]:
        """Return the provenance record of the population file.

        Args:
            file_format: Format the population is written in, ``"csv"`` or ``"hdf5"``.

        Returns:
            A record for :func:`gwmock_pop.loaders.write_population_catalogue`
            naming the population model, the run settings, the rate integral and
            the derived mean interval, the event counts and the package versions.
        """
        configuration = {
            "generator": "gwmock_pop.catalogue.imbh_list.generate_imbh_list",
            "population": asdict(self.population),
            "settings": asdict(self.settings),
            "span_seconds": self.settings.n_segments * self.settings.segment_duration,
            "uncapped_events_per_year": self.rate.uncapped_events_per_year,
            "events_per_year": self.rate.events_per_year,
            "mean_interval_seconds": self.rate.mean_interval,
            "julian_year_seconds": JULIAN_YEAR_SECONDS,
            "redshift_grid_size": _REDSHIFT_GRID_SIZE,
            "n_events_drawn": self.n_events,
            "n_events_listed": self.n_listed,
            "n_imri_listed": int(np.count_nonzero(self.columns["source_type"] == IMRI_TYPE)),
        }
        origin = simulator_origin(
            component="gwmock_pop.catalogue.imbh_list.generate_imbh_list", configuration=configuration
        )
        origin["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
        population = self.population_columns()
        return build_provenance_record(
            origin=origin,
            source_type=IMBH_ROUTING_KEY,
            parameter_names=list(population),
            n_samples=self.n_listed,
            file_format=file_format,
            writer="gwmock_pop.loaders.file_loader.write_population_catalogue",
            run=run_metadata(name=None, seed=self.settings.seed, seed_source="library"),
        )


def generate_imbh_list(settings: ImbhListSettings, population: ImbhPopulation | None = None) -> ImbhList:
    """Generate an IMBH injection list.

    Args:
        settings: Data span, seed and ``f_min``.
        population: Population model; the defaults when omitted.

    Returns:
        The listed events and the record of how they were generated.

    Raises:
        PopulationValidationError: If the settings or the model are out of range.
    """
    population = ImbhPopulation() if population is None else population
    settings.validate()
    rate = imbh_rate(population)

    coa_time, uniforms = _draw_schedule(settings, population, rate.mean_interval)
    n_events = len(coa_time)

    columns = _intrinsic_columns(uniforms, population, rate)
    columns.update(_signal_bounds(columns, coa_time, settings.f_min))
    columns.update(_angles_from_uniforms(uniforms[:, _ANGLE_DRAWS]))
    columns.update(_segment_range(columns["start_time"], columns["end_time"], settings))
    columns["event_id"] = np.arange(1, n_events + 1, dtype=np.int64)

    listed = columns["start_time"] < settings.stop_time
    return ImbhList(
        settings=settings,
        population=population,
        rate=rate,
        columns={name: values[listed] for name, values in columns.items()},
        n_events=n_events,
    )


__all__ = [
    "IMBH_ROUTING_KEY",
    "IMBH_TYPE",
    "IMRI_TYPE",
    "JULIAN_YEAR_SECONDS",
    "ImbhList",
    "ImbhListSettings",
    "ImbhPopulation",
    "ImbhRate",
    "generate_imbh_list",
    "imbh_rate",
    "surviving_mass_fraction",
]
