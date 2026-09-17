"""Deterministic draws from a published merger catalogue into injection parameters.

The draw is the step that turns a source-population catalogue at real distances
into the parameter set a search engine injects. It answers "which mergers
happen in this window, and with what parameters" without a population model of
its own: the masses, spins, distances and sky positions are read from the
catalogue row by row, and only the number of rows and the merger times are
drawn.

Three properties are deliberate and are what makes the result usable as a
producer artefact.

**The draw is seeded and cell-counted.** Each class's row count is Poisson with
mean ``n_rows / year * span * rate_multiplier`` -- the catalogue holds one year
of mergers, so its row count is its annual rate -- and the rows are then sampled
without replacement (with replacement only when more are asked for than the
catalogue holds, where a Poisson tail can exceed a small class). The same seed,
span and files reproduce the draw exactly.

**Distances are the catalogue's own.** No rescaling is applied. A source's
luminosity distance is its catalogue value converted from Gpc to Mpc, and its
redshift is retained, so the detector-frame masses are ``m_source * (1 + z)``
and the band a template bank searches can be measured in the frame the search
sees. A population drawn at rescaled distances would instead measure the
rescaling.

**Classes are labelled, not reweighted.** The binary-neutron-star and
binary-black-hole labels are the catalogues the rows came from. The
intermediate-mass label is a mass cut of the black-hole catalogue -- the
standard lower bound on an intermediate-mass black hole, applied to the
source-frame total mass -- because the published catalogue product has no
separate intermediate-mass file. It is a labelling convention and is recorded as
such in the draw's provenance, not a claim about the catalogue.

Only the aligned spin components are carried. The catalogue's precessing
degrees of freedom are dropped because every waveform family the reference
population is injected with is aligned-spin; the alternative would be a
component the waveform does not consume.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from gwmock_pop.catalogue.manifest import COBA_CATALOGUES, CatalogueFile
from gwmock_pop.catalogue.source import ResolvedCatalogue, fetch_catalogues
from gwmock_pop.exceptions import PopulationValidationError

#: Seconds in a Julian year, matching the catalogue's one-year row count.
SECONDS_PER_YEAR = 365.25 * 86400.0

#: Solar mass in geometric seconds, GM_sun / c^3.
MSUN_SECONDS = 4.925_491e-6

#: Catalogue column names the draw reads.
_CATALOGUE_COLUMNS = (
    "m1_source",
    "m2_source",
    "chi1z",
    "chi2z",
    "dL",
    "ra",
    "dec",
    "psi",
    "iota",
    "Phicoal",
    "z",
)

#: Output parameter names, in the order they are written.
PARAMETER_NAMES = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1z",
    "spin_2z",
    "distance",
    "right_ascension",
    "declination",
    "inclination",
    "coa_phase",
    "polarization_angle",
    "coa_time",
    "redshift",
)

#: Catalogue distance unit to the output unit (Gpc to Mpc).
_GPC_TO_MPC = 1000.0

#: Label of a row drawn from a binary-neutron-star catalogue.
BNS_CLASS = "BNS"

#: Label of a row drawn from a binary-black-hole catalogue.
BBH_CLASS = "BBH"

#: Label of a heavy black-hole row, above the intermediate-mass threshold.
IMBH_CLASS = "IMBH"


def chirp_mass(mass_1: np.ndarray, mass_2: np.ndarray) -> np.ndarray:
    """Return the chirp mass of a binary, ``(m1 m2)^(3/5) / (m1 + m2)^(1/5)``.

    Args:
        mass_1: First component mass.
        mass_2: Second component mass.

    Returns:
        The chirp mass, in the unit of the inputs.
    """
    mass_1 = np.asarray(mass_1, dtype=float)
    mass_2 = np.asarray(mass_2, dtype=float)
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2


def inband_duration_seconds(detector_frame_chirp_mass: np.ndarray, f_low: float) -> np.ndarray:
    """Return the leading-order in-band duration from ``f_low`` to merger.

    The detector-frame chirp mass and the observed low-frequency cutoff give the
    observed time to coalescence directly: a source at redshift ``z`` enters at
    an observed frequency ``f_low`` when the emitted frequency is
    ``f_low (1 + z)``, and the two factors of ``(1 + z)`` cancel against the
    detector-frame chirp mass's ``(1 + z)`` boost, so no redshift enters here.

    Args:
        detector_frame_chirp_mass: Detector-frame chirp mass in solar masses.
        f_low: Observed low-frequency cutoff in hertz.

    Returns:
        The time from entering the band to merger, in seconds.
    """
    chirp_seconds = np.asarray(detector_frame_chirp_mass, dtype=float) * MSUN_SECONDS
    return (5.0 / 256.0) * (np.pi * f_low) ** (-8.0 / 3.0) * chirp_seconds ** (-5.0 / 3.0)


@dataclass(frozen=True, slots=True)
class CatalogueDraw:
    """One seeded draw of a merger catalogue into injection parameters.

    Attributes:
        seed: Seed the draw's generator was initialized with.
        span_seconds: Observation span the draw covers, in seconds.
        rate_multiplier: Factor applied to every catalogue's annual rate.
        band_detector_frame_chirp_mass: Lower edge of the searched band, in the
            detector frame.
        imbh_total_mass_threshold: Source-frame total mass at or above which a
            black-hole-catalogue row is labelled intermediate-mass.
        parameters: Mapping from output parameter name to a 1-D array, in
            :data:`PARAMETER_NAMES` order.
        source_class: Catalogue class of each drawn row.
        population_class: Labelled class of each drawn row -- a catalogue class,
            or ``"IMBH"`` for a heavy black-hole row.
        detector_frame_chirp_mass: Detector-frame chirp mass of each row.
        source_frame_chirp_mass: Source-frame chirp mass of each row.
        in_band: Whether the row's detector-frame chirp mass is above the band.
        expected_per_class: Poisson mean count per catalogue class.
        drawn_per_class: Realized count per catalogue class.
        resolved: The verified catalogue files the draw was made from.
    """

    seed: int
    span_seconds: float
    rate_multiplier: float
    band_detector_frame_chirp_mass: float
    imbh_total_mass_threshold: float
    parameters: dict[str, np.ndarray] = field(repr=False)
    source_class: np.ndarray = field(repr=False)
    population_class: np.ndarray = field(repr=False)
    detector_frame_chirp_mass: np.ndarray = field(repr=False)
    source_frame_chirp_mass: np.ndarray = field(repr=False)
    in_band: np.ndarray = field(repr=False)
    expected_per_class: Mapping[str, float]
    drawn_per_class: Mapping[str, int]
    resolved: tuple[ResolvedCatalogue, ...] = field(repr=False)

    @property
    def n_rows(self) -> int:
        """Return the number of drawn rows.

        Returns:
            The number of rows, zero when the span admitted no merger.
        """
        return int(self.in_band.shape[0])

    @property
    def in_band_rows(self) -> int:
        """Return the number of drawn rows inside the searched band.

        Returns:
            The count of rows whose detector-frame chirp mass is above the band.
        """
        return int(np.count_nonzero(self.in_band))


def _validate_draw_inputs(
    *,
    seed: int,
    span_seconds: float,
    rate_multiplier: float,
    band_detector_frame_chirp_mass: float,
    imbh_total_mass_threshold: float,
) -> None:
    """Refuse draw settings that cannot describe a population.

    Args:
        seed: Seed for the draw's generator.
        span_seconds: Observation span in seconds.
        rate_multiplier: Factor applied to every catalogue's annual rate.
        band_detector_frame_chirp_mass: Lower edge of the searched band.
        imbh_total_mass_threshold: Source-frame total mass of an
            intermediate-mass black hole.

    Raises:
        PopulationValidationError: If any setting is out of range.
    """
    if not np.isfinite(span_seconds) or span_seconds < 0.0:
        raise PopulationValidationError(f"span_seconds must be finite and non-negative, got {span_seconds!r}.")
    if not np.isfinite(rate_multiplier) or rate_multiplier < 0.0:
        raise PopulationValidationError(f"rate_multiplier must be finite and non-negative, got {rate_multiplier!r}.")
    if not np.isfinite(band_detector_frame_chirp_mass) or band_detector_frame_chirp_mass <= 0.0:
        raise PopulationValidationError(
            f"band_detector_frame_chirp_mass must be finite and positive, got {band_detector_frame_chirp_mass!r}."
        )
    if not np.isfinite(imbh_total_mass_threshold) or imbh_total_mass_threshold <= 0.0:
        raise PopulationValidationError(
            f"imbh_total_mass_threshold must be finite and positive, got {imbh_total_mass_threshold!r}."
        )
    if not isinstance(seed, (int, np.integer)):
        raise PopulationValidationError(f"seed must be an integer, got {seed!r}.")


def read_catalogue_columns(
    path: str | os.PathLike[str],
    columns: Sequence[str] = _CATALOGUE_COLUMNS,
) -> dict[str, np.ndarray]:
    """Read named columns from a catalogue file.

    Args:
        path: Path of the HDF5 catalogue.
        columns: Columns to read.

    Returns:
        Mapping from column name to a 1-D array.

    Raises:
        PopulationValidationError: If a required column is missing or empty, or
            if the columns do not all have the same length.
    """
    catalogue_path = Path(path)
    with h5py.File(catalogue_path, "r") as handle:
        missing = [name for name in columns if name not in handle]
        if missing:
            raise PopulationValidationError(
                f"Catalogue {catalogue_path} is missing required columns: {', '.join(missing)}."
            )
        read_columns = {name: np.asarray(handle[name]) for name in columns}

    lengths = {name: values.shape[0] for name, values in read_columns.items()}
    if not lengths:
        raise PopulationValidationError(f"Catalogue {catalogue_path} holds no columns.")
    distinct = set(lengths.values())
    if len(distinct) != 1:
        raise PopulationValidationError(f"Catalogue {catalogue_path} has mismatched column lengths: {lengths}.")
    if next(iter(distinct)) == 0:
        raise PopulationValidationError(f"Catalogue {catalogue_path} is empty.")
    return read_columns


def _draw_from_catalogue(
    resolved: ResolvedCatalogue,
    *,
    rng: np.random.Generator,
    span_seconds: float,
    rate_multiplier: float,
) -> dict[str, np.ndarray]:
    """Draw one catalogue's rows for an observation span.

    Args:
        resolved: The verified catalogue to draw from.
        rng: Random generator, shared across classes so the draw is determined
            by the seed and the class order alone.
        span_seconds: Observation span in seconds.
        rate_multiplier: Factor applied to the catalogue's annual rate.

    Returns:
        Mapping from output parameter name to a 1-D array of drawn values.

    Raises:
        PopulationValidationError: If the catalogue cannot be read or holds
            non-finite mass, spin, distance or redshift values.
    """
    columns = read_catalogue_columns(resolved.path)
    n_catalogue = columns["m1_source"].shape[0]
    expected = n_catalogue * (span_seconds / SECONDS_PER_YEAR) * rate_multiplier
    count = int(rng.poisson(expected))
    if count == 0:
        return {name: np.empty(0, dtype=float) for name in PARAMETER_NAMES}

    indices = rng.choice(n_catalogue, size=count, replace=count > n_catalogue)

    mass_1_source = columns["m1_source"][indices]
    mass_2_source = columns["m2_source"][indices]
    redshift = columns["z"][indices]
    for name, values in (
        ("m1_source", mass_1_source),
        ("m2_source", mass_2_source),
        ("z", redshift),
    ):
        if not np.all(np.isfinite(values)):
            raise PopulationValidationError(f"Catalogue {resolved.catalogue.filename} holds non-finite {name} values.")

    one_plus_redshift = 1.0 + redshift
    coa_time = rng.uniform(0.0, span_seconds, size=count)
    return {
        "detector_frame_mass_1": mass_1_source * one_plus_redshift,
        "detector_frame_mass_2": mass_2_source * one_plus_redshift,
        "spin_1z": columns["chi1z"][indices],
        "spin_2z": columns["chi2z"][indices],
        "distance": columns["dL"][indices] * _GPC_TO_MPC,
        "right_ascension": columns["ra"][indices],
        "declination": columns["dec"][indices],
        "inclination": columns["iota"][indices],
        "coa_phase": columns["Phicoal"][indices],
        "polarization_angle": columns["psi"][indices],
        "coa_time": coa_time,
        "redshift": redshift,
    }


def draw_catalogue_population(  # noqa: PLR0913  # the draw's settings, named at its one call site
    *,
    seed: int,
    span_seconds: float,
    catalogues: Sequence[CatalogueFile] | None = None,
    resolved: Sequence[ResolvedCatalogue] | None = None,
    rate_multiplier: float = 1.0,
    band_detector_frame_chirp_mass: float = 5.0,
    imbh_total_mass_threshold: float = 100.0,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    timeout: int = 300,
) -> CatalogueDraw:
    """Draw a seeded population from pinned catalogues into injection parameters.

    Classes are drawn in ``catalogues`` order, each with a Poisson count and a
    without-replacement row sample, so the same seed, span and files reproduce
    the draw exactly. Pass ``resolved`` to draw from already-resolved files
    (a cached or locally built catalogue); otherwise the files are fetched and
    digest-verified on demand.

    Args:
        seed: Seed for the draw's generator.
        span_seconds: Observation span the draw covers, in seconds.
        catalogues: Pinned catalogues to draw from, in draw order. Defaults to
            the catalogues of ``resolved`` when those are given, otherwise to
            the design-comparison pair.
        resolved: Already-resolved catalogue files, in the same order. When
            given, no fetch is performed.
        rate_multiplier: Factor applied to every catalogue's annual rate.
        band_detector_frame_chirp_mass: Lower edge of the searched band, in the
            detector frame.
        imbh_total_mass_threshold: Source-frame total mass at or above which a
            black-hole-catalogue row is labelled intermediate-mass.
        cache_dir: Cache directory for on-demand fetches.
        refresh: Whether to re-download catalogues even when cached.
        offline: Whether to refuse network access and use only matching caches.
        timeout: Timeout in seconds for each catalogue download.

    Returns:
        The draw, its parameters and its class labels.

    Raises:
        PopulationValidationError: If the settings are out of range, or a
            catalogue is malformed.
        PopulationFetchError: If a catalogue cannot be fetched or does not match
            its pinned digest.
    """
    _validate_draw_inputs(
        seed=seed,
        span_seconds=span_seconds,
        rate_multiplier=rate_multiplier,
        band_detector_frame_chirp_mass=band_detector_frame_chirp_mass,
        imbh_total_mass_threshold=imbh_total_mass_threshold,
    )
    if catalogues is None:
        catalogues = tuple(item.catalogue for item in resolved) if resolved is not None else COBA_CATALOGUES
    resolved_catalogues = (
        tuple(resolved)
        if resolved is not None
        else fetch_catalogues(catalogues, cache_dir=cache_dir, refresh=refresh, offline=offline, timeout=timeout)
    )
    if len(resolved_catalogues) != len(catalogues):
        raise PopulationValidationError(
            f"Received {len(resolved_catalogues)} resolved catalogues for {len(catalogues)} configured ones."
        )

    rng = np.random.default_rng(seed)
    expected_per_class: dict[str, float] = {}
    drawn_per_class: dict[str, int] = {}
    parameter_blocks: dict[str, list[np.ndarray]] = {name: [] for name in PARAMETER_NAMES}
    source_classes: list[str] = []
    for catalogue, resolved_catalogue in zip(catalogues, resolved_catalogues, strict=True):
        expected = catalogue.n_rows * (span_seconds / SECONDS_PER_YEAR) * rate_multiplier
        block = _draw_from_catalogue(
            resolved_catalogue,
            rng=rng,
            span_seconds=span_seconds,
            rate_multiplier=rate_multiplier,
        )
        count = block["detector_frame_mass_1"].shape[0]
        expected_per_class[catalogue.source_class] = expected
        drawn_per_class[catalogue.source_class] = drawn_per_class.get(catalogue.source_class, 0) + count
        for name in PARAMETER_NAMES:
            parameter_blocks[name].append(block[name])
        source_classes.extend([catalogue.source_class] * count)

    parameters = {
        name: np.concatenate(blocks) if blocks else np.empty(0, dtype=float)
        for name, blocks in parameter_blocks.items()
    }
    source_class = np.asarray(source_classes, dtype=object)
    mass_1 = parameters["detector_frame_mass_1"]
    mass_2 = parameters["detector_frame_mass_2"]
    redshift = parameters["redshift"]
    detector_frame_chirp_mass = chirp_mass(mass_1, mass_2)
    source_frame_chirp_mass = detector_frame_chirp_mass / (1.0 + redshift)

    # The source-frame total mass is the same combination in both classes; the
    # label only decides whether the cut applies, not how the mass is formed.
    source_total_mass = (mass_1 + mass_2) / (1.0 + redshift)
    population_class = np.array(source_class, dtype=object)
    imbh_rows = (source_class == BBH_CLASS) & (source_total_mass >= imbh_total_mass_threshold)
    population_class[imbh_rows] = IMBH_CLASS

    return CatalogueDraw(
        seed=int(seed),
        span_seconds=float(span_seconds),
        rate_multiplier=float(rate_multiplier),
        band_detector_frame_chirp_mass=float(band_detector_frame_chirp_mass),
        imbh_total_mass_threshold=float(imbh_total_mass_threshold),
        parameters=parameters,
        source_class=source_class,
        population_class=population_class,
        detector_frame_chirp_mass=detector_frame_chirp_mass,
        source_frame_chirp_mass=source_frame_chirp_mass,
        in_band=detector_frame_chirp_mass > band_detector_frame_chirp_mass,
        expected_per_class=expected_per_class,
        drawn_per_class=drawn_per_class,
        resolved=resolved_catalogues,
    )


def draw_catalogue_population_many(  # noqa: PLR0913  # the draw's settings, named at its one call site
    *,
    seeds: Sequence[int],
    span_seconds: float,
    catalogues: Sequence[CatalogueFile] | None = None,
    resolved: Sequence[ResolvedCatalogue] | None = None,
    rate_multiplier: float = 1.0,
    band_detector_frame_chirp_mass: float = 5.0,
    imbh_total_mass_threshold: float = 100.0,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
    offline: bool = False,
    timeout: int = 300,
) -> tuple[CatalogueDraw, ...]:
    """Draw the same population once per seed.

    The catalogues are resolved once and reused across seeds, so a repeated-draw
    measurement pays for one download and reseeds only the draw.

    Args:
        seeds: Seeds to draw, one draw each.
        span_seconds: Observation span each draw covers, in seconds.
        catalogues: Pinned catalogues to draw from, in draw order. Defaults to
            the catalogues of ``resolved`` when those are given, otherwise to
            the design-comparison pair.
        resolved: Already-resolved catalogue files, in the same order.
        rate_multiplier: Factor applied to every catalogue's annual rate.
        band_detector_frame_chirp_mass: Lower edge of the searched band.
        imbh_total_mass_threshold: Source-frame total mass at or above which a
            black-hole row is labelled intermediate-mass.
        cache_dir: Cache directory for on-demand fetches.
        refresh: Whether to re-download catalogues even when cached.
        offline: Whether to refuse network access and use only matching caches.
        timeout: Timeout in seconds for each catalogue download.

    Returns:
        One draw per seed, in seed order.
    """
    if catalogues is None:
        catalogues = tuple(item.catalogue for item in resolved) if resolved is not None else COBA_CATALOGUES
    resolved_catalogues = (
        tuple(resolved)
        if resolved is not None
        else fetch_catalogues(catalogues, cache_dir=cache_dir, refresh=refresh, offline=offline, timeout=timeout)
    )
    return tuple(
        draw_catalogue_population(
            seed=seed,
            span_seconds=span_seconds,
            catalogues=catalogues,
            resolved=resolved_catalogues,
            rate_multiplier=rate_multiplier,
            band_detector_frame_chirp_mass=band_detector_frame_chirp_mass,
            imbh_total_mass_threshold=imbh_total_mass_threshold,
        )
        for seed in seeds
    )


__all__ = [
    "BBH_CLASS",
    "BNS_CLASS",
    "IMBH_CLASS",
    "MSUN_SECONDS",
    "PARAMETER_NAMES",
    "SECONDS_PER_YEAR",
    "CatalogueDraw",
    "chirp_mass",
    "draw_catalogue_population",
    "draw_catalogue_population_many",
    "inband_duration_seconds",
    "read_catalogue_columns",
]
