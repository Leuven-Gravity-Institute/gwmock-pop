"""Per-class composition of the detector-frame band and the draw's other summaries.

A band cut is a statement about a population: how much of what a search admits
is black holes and how much is the redshift-boosted neutron-star tail. Counting
it needs the draw, the cut and the frame all fixed, so this module reports the
composition, the drawn rates, the chirp-mass ranges, the redshift reach and the
in-band duration together -- a composition quoted without the band, the frame
and the span is not a measurement of anything.

Two conventions are deliberate.

**The detector frame is the measurement; the source frame is the control.**
The band is defined on the detector-frame chirp mass, because that is what a
search's templates are sorted by. The source frame is reported next to it, and
the difference between the two is the ``(1 + z)`` boost's contribution and
nothing else -- the internal control for the claim that the tail enters the band
through that boost.

**The spread is draw to draw, not a fitted uncertainty.** Each draw is a
different realisation of the Poisson counts and row samples at the same seed
family, and the reported spread is the spread of the per-draw shares. The lower
bound on the aggregate share is a one-sided Wilson interval, computed in closed
form so no optional statistics dependency is introduced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gwmock_pop.catalogue.manifest import CatalogueFile
from gwmock_pop.catalogue.population import (
    BBH_CLASS,
    BNS_CLASS,
    IMBH_CLASS,
    SECONDS_PER_YEAR,
    CatalogueDraw,
    chirp_mass,
    inband_duration_seconds,
    read_catalogue_columns,
)
from gwmock_pop.catalogue.source import ResolvedCatalogue

#: Class order every composition table uses.
CLASS_ORDER = (BNS_CLASS, BBH_CLASS, IMBH_CLASS)

#: Quantiles of a distribution this module reports.
QUANTILES = (0.1, 0.5, 0.9)

#: One-sided z for a 90 per cent lower bound.
_Z_90 = 1.2815515655446004


def wilson_lower_bound(successes: int, trials: int, *, z: float = _Z_90) -> float:
    """Return the one-sided Wilson lower bound on a binomial proportion.

    The closed form needs no optional dependency and behaves at the edges: it
    returns zero rather than a negative number when the observed share is small,
    and it is defined for zero trials.

    Args:
        successes: Number of successes.
        trials: Number of trials.
        z: One-sided normal quantile; the default is the 90 per cent bound.

    Returns:
        The lower bound, in ``[0, 1]``.
    """
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    half_width = z * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
    return float(max(0.0, (centre - half_width) / denominator))


@dataclass(frozen=True, slots=True)
class ClassBandCounts:
    """One class's counts in and out of the band, pooled across draws.

    Attributes:
        name: Class label.
        drawn: Rows drawn across all draws.
        in_band: Rows drawn inside the band across all draws.
        in_band_fraction: ``in_band / drawn`` for a class with rows, else zero.
        share_of_band: ``in_band`` over the total in-band count, else zero.
    """

    name: str
    drawn: int
    in_band: int
    in_band_fraction: float
    share_of_band: float


@dataclass(frozen=True, slots=True)
class BandComposition:
    """The measured composition and summaries of a set of draws.

    Attributes:
        band_detector_frame_chirp_mass: Lower edge of the band.
        imbh_total_mass_threshold: Intermediate-mass total-mass threshold.
        n_draws: Number of draws pooled.
        seeds: Seeds of the draws, in order.
        class_counts: Per-class pooled counts, in :data:`CLASS_ORDER`.
        share_of_band_mean: Per class, the mean per-draw share of the band.
        share_of_band_std: Per class, the standard deviation of the per-draw
            share of the band.
        share_of_band_min: Per class, the smallest per-draw share of the band.
        share_of_band_max: Per class, the largest per-draw share of the band.
        share_of_band_per_draw: Per class, the per-draw share of the band.
        bns_share_of_band_lower_90: One-sided 90 per cent Wilson lower bound on
            the pooled binary-neutron-star share of the band.
        detector_frame_chirp_mass_range: Per class, ``(min, max)`` detector-frame
            chirp mass over the in-band rows, or ``None`` when none are in band.
        source_frame_chirp_mass_range: The same range in the source frame.
        redshift_reach: Largest redshift over all drawn rows, or ``None``.
        drawn_rate_per_year: Per class, the annual rate the draw was made at.
        in_band_duration_quantiles: Leading-order in-band duration quantiles, in
            seconds, over the in-band rows.
    """

    band_detector_frame_chirp_mass: float
    imbh_total_mass_threshold: float
    n_draws: int
    seeds: tuple[int, ...]
    class_counts: tuple[ClassBandCounts, ...]
    share_of_band_mean: Mapping[str, float]
    share_of_band_std: Mapping[str, float]
    share_of_band_min: Mapping[str, float]
    share_of_band_max: Mapping[str, float]
    share_of_band_per_draw: Mapping[str, tuple[float, ...]]
    bns_share_of_band_lower_90: float
    detector_frame_chirp_mass_range: Mapping[str, tuple[float, float] | None]
    source_frame_chirp_mass_range: Mapping[str, tuple[float, float] | None]
    redshift_reach: float | None
    drawn_rate_per_year: Mapping[str, float]
    in_band_duration_quantiles: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the composition as a JSON-ready mapping.

        Returns:
            A plain mapping with lists where the dataclass holds tuples, so it
            can be written as a stamp beside a draw.
        """

        def flatten_ranges(ranges: Mapping[str, tuple[float, float] | None]) -> dict[str, list[float] | None]:
            return {name: None if bounds is None else list(bounds) for name, bounds in ranges.items()}

        return {
            "band_detector_frame_chirp_mass": self.band_detector_frame_chirp_mass,
            "imbh_total_mass_threshold": self.imbh_total_mass_threshold,
            "n_draws": self.n_draws,
            "seeds": list(self.seeds),
            "class_counts": [
                {
                    "name": counts.name,
                    "drawn": counts.drawn,
                    "in_band": counts.in_band,
                    "in_band_fraction": counts.in_band_fraction,
                    "share_of_band": counts.share_of_band,
                }
                for counts in self.class_counts
            ],
            "share_of_band_mean": dict(self.share_of_band_mean),
            "share_of_band_std": dict(self.share_of_band_std),
            "share_of_band_min": dict(self.share_of_band_min),
            "share_of_band_max": dict(self.share_of_band_max),
            "share_of_band_per_draw": {name: list(values) for name, values in self.share_of_band_per_draw.items()},
            "bns_share_of_band_lower_90": self.bns_share_of_band_lower_90,
            "detector_frame_chirp_mass_range": flatten_ranges(self.detector_frame_chirp_mass_range),
            "source_frame_chirp_mass_range": flatten_ranges(self.source_frame_chirp_mass_range),
            "redshift_reach": self.redshift_reach,
            "drawn_rate_per_year": dict(self.drawn_rate_per_year),
            "in_band_duration_quantiles": dict(self.in_band_duration_quantiles),
        }


def _present_classes(draws: Sequence[CatalogueDraw]) -> tuple[str, ...]:
    """Return the class labels present across draws, in the canonical order.

    Args:
        draws: The draws to inspect.

    Returns:
        The class labels present, ordered by :data:`CLASS_ORDER`.
    """
    present = {label for draw in draws for label in draw.population_class.tolist()}
    ordered = [name for name in CLASS_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _catalogue_identity(draw: CatalogueDraw) -> tuple[CatalogueFile, ...]:
    """Return the pinned catalogue identity a draw's rows were sampled from.

    The identity of the population is the pinned catalogues, not the filesystem
    location or the fetch metadata: two resolutions of the same pins are the
    same population even when one was a cache hit and the other a download.

    Args:
        draw: The draw to describe.

    Returns:
        The pinned catalogue of each resolved input, in draw order.
    """
    return tuple(resolved.catalogue for resolved in draw.resolved)


def _pooled_counts(draw: CatalogueDraw, name: str) -> tuple[int, int]:
    """Return a class's drawn and in-band counts in one draw.

    Args:
        draw: The draw to count.
        name: Class label.

    Returns:
        ``(drawn, in_band)`` for the class.
    """
    rows = draw.population_class == name
    return int(np.count_nonzero(rows)), int(np.count_nonzero(rows & draw.in_band))


def _mass_range(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float] | None:
    """Return the ``(min, max)`` of a masked array, or ``None`` when empty.

    Args:
        values: The values to summarise.
        mask: Boolean selection.

    Returns:
        The bounds, or ``None`` when nothing is selected.
    """
    selected = values[mask]
    if selected.size == 0:
        return None
    return float(np.min(selected)), float(np.max(selected))


def _duration_quantiles(draw: CatalogueDraw, *, f_low: float) -> dict[str, float]:
    """Return in-band duration quantiles over the in-band rows of one draw.

    Args:
        draw: The draw to summarise.
        f_low: Observed low-frequency cutoff in hertz.

    Returns:
        Mapping from quantile label to duration in seconds; empty when no row is
        in band.
    """
    if not np.any(draw.in_band):
        return {}
    durations = inband_duration_seconds(draw.detector_frame_chirp_mass[draw.in_band], f_low)
    return {f"p{int(quantile * 100)}": float(np.quantile(durations, quantile)) for quantile in QUANTILES}


def composition_summary(
    draws: Sequence[CatalogueDraw],
    *,
    f_low: float = 5.0,
    black_hole_rate_split: Mapping[str, float] | None = None,
) -> BandComposition:
    """Summarise the composition of several draws of the same population.

    Args:
        draws: Draws to summarise. All must sample the same population under the
            same settings -- the band, the intermediate-mass threshold, the
            observation span, the rate multiplier and the pinned catalogues; the
            seeds must not be empty.
        f_low: Observed low-frequency cutoff, for the in-band duration.
        black_hole_rate_split: Fraction of the black-hole catalogue's annual
            rate each mutually exclusive black-hole-derived class takes, keyed
            by class label (for example ``{"BBH": 0.9847, "IMBH": 0.0153}``).
            Pass the fractions measured from the full catalogue when they are
            available; omit them to split the rate by the draw's own labelled
            counts. Either way the classes partition the black-hole rate.

    Returns:
        The composition and the draw-to-draw spread.

    Raises:
        ValueError: If no draws are given, they disagree on the band edge, the
            intermediate-mass threshold, the observation span, the rate
            multiplier or the pinned catalogues, or ``black_hole_rate_split``
            names a class that is not black-hole-derived, does not sum to one,
            or holds a negative or non-finite fraction.
    """
    if not draws:
        raise ValueError("At least one draw is needed to measure a composition.")
    band = draws[0].band_detector_frame_chirp_mass
    threshold = draws[0].imbh_total_mass_threshold
    if any(draw.band_detector_frame_chirp_mass != band for draw in draws):
        raise ValueError("Every draw in a composition must share the same band edge.")
    if any(draw.imbh_total_mass_threshold != threshold for draw in draws):
        raise ValueError("Every draw in a composition must share the same intermediate-mass threshold.")
    span = draws[0].span_seconds
    if any(draw.span_seconds != span for draw in draws):
        raise ValueError("Every draw in a composition must share the same observation span.")
    rate_multiplier = draws[0].rate_multiplier
    if any(draw.rate_multiplier != rate_multiplier for draw in draws):
        raise ValueError("Every draw in a composition must share the same rate multiplier.")
    identity = _catalogue_identity(draws[0])
    if any(_catalogue_identity(draw) != identity for draw in draws):
        raise ValueError("Every draw in a composition must be drawn from the same catalogues.")

    classes = _present_classes(draws)
    _validate_black_hole_rate_split(black_hole_rate_split)
    pooled_drawn = dict.fromkeys(classes, 0)
    pooled_in_band = dict.fromkeys(classes, 0)
    per_draw_shares: dict[str, list[float]] = {name: [] for name in classes}
    for draw in draws:
        total_in_band = draw.in_band_rows
        for name in classes:
            drawn, in_band = _pooled_counts(draw, name)
            pooled_drawn[name] += drawn
            pooled_in_band[name] += in_band
            per_draw_shares[name].append(in_band / total_in_band if total_in_band else 0.0)

    total_pooled_in_band = sum(pooled_in_band.values())
    class_counts = tuple(
        ClassBandCounts(
            name=name,
            drawn=pooled_drawn[name],
            in_band=pooled_in_band[name],
            in_band_fraction=pooled_in_band[name] / pooled_drawn[name] if pooled_drawn[name] else 0.0,
            share_of_band=pooled_in_band[name] / total_pooled_in_band if total_pooled_in_band else 0.0,
        )
        for name in classes
    )

    first = draws[0]
    return BandComposition(
        band_detector_frame_chirp_mass=band,
        imbh_total_mass_threshold=threshold,
        n_draws=len(draws),
        seeds=tuple(draw.seed for draw in draws),
        class_counts=class_counts,
        share_of_band_mean={name: float(np.mean(values)) for name, values in per_draw_shares.items()},
        share_of_band_std={name: float(np.std(values)) for name, values in per_draw_shares.items()},
        share_of_band_min={name: float(np.min(values)) for name, values in per_draw_shares.items()},
        share_of_band_max={name: float(np.max(values)) for name, values in per_draw_shares.items()},
        share_of_band_per_draw={name: tuple(values) for name, values in per_draw_shares.items()},
        bns_share_of_band_lower_90=wilson_lower_bound(pooled_in_band.get(BNS_CLASS, 0), total_pooled_in_band),
        detector_frame_chirp_mass_range={
            name: _mass_range(first.detector_frame_chirp_mass, (first.population_class == name) & first.in_band)
            for name in classes
        },
        source_frame_chirp_mass_range={
            name: _mass_range(first.source_frame_chirp_mass, (first.population_class == name) & first.in_band)
            for name in classes
        },
        redshift_reach=float(np.max(first.parameters["redshift"])) if first.n_rows else None,
        drawn_rate_per_year={
            name: _class_rate(
                name,
                draw=first,
                pooled_drawn=pooled_drawn,
                black_hole_rate_split=black_hole_rate_split,
            )
            for name in classes
        },
        in_band_duration_quantiles=_duration_quantiles(first, f_low=f_low),
    )


def _validate_black_hole_rate_split(split: Mapping[str, float] | None) -> None:
    """Refuse a black-hole rate split that does not partition the rate.

    The rates are reported per mutually exclusive class, so the fractions handed
    in must be for black-hole-derived classes and sum to one. A split that sums
    to less than one would silently leave part of the black-hole rate
    unattributed -- the same double-counting the split exists to remove, in the
    other direction.

    A black-hole class named by the split but absent from the draws is allowed:
    the catalogue cut can hold rows the draw did not produce, and its rate is
    simply not reported.

    Args:
        split: Fraction of the black-hole rate per class, or ``None``.

    Raises:
        ValueError: If the split names a class that is not black-hole-derived,
            holds a negative or non-finite fraction, or does not sum to one.
    """
    if split is None:
        return
    unknown = sorted(set(split) - {BBH_CLASS, IMBH_CLASS})
    if unknown:
        raise ValueError(f"black_hole_rate_split names classes that are not black-hole-derived: {', '.join(unknown)}.")
    for name, value in split.items():
        fraction = float(value)
        if not np.isfinite(fraction) or fraction < 0.0:
            raise ValueError(f"black_hole_rate_split fractions must be finite and non-negative, got {name}={value!r}.")
    total = sum(float(value) for value in split.values())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"black_hole_rate_split must sum to 1 over the black-hole classes, got {total}.")


def _class_rate(
    name: str,
    *,
    draw: CatalogueDraw,
    pooled_drawn: Mapping[str, int],
    black_hole_rate_split: Mapping[str, float] | None = None,
) -> float:
    """Return the annual rate a class was drawn at.

    The Poisson mean is ``n_rows * span / year * multiplier``, so dividing it by
    the span in years recovers the annual rate -- the catalogue's own row count
    times the multiplier, independent of the span. The intermediate-mass label
    is a cut of the black-hole catalogue, so ``BBH`` and ``IMBH`` share the one
    black-hole rate rather than each taking it: their fractions of it come from
    ``black_hole_rate_split`` when the caller measured them on the full
    catalogue, and otherwise from the draw's own labelled counts. Either way the
    mutually exclusive classes partition the black-hole catalogue's rate.

    Args:
        name: Class label.
        draw: A draw from the set, supplying the span and the expectations.
        pooled_drawn: Rows drawn per class, pooled across the set.
        black_hole_rate_split: Fraction of the black-hole rate per
            black-hole-derived class, or ``None`` to use the draw's counts.

    Returns:
        The annual rate for the class, or zero when the class is absent.
    """
    if draw.span_seconds <= 0.0:
        return 0.0
    span_years = draw.span_seconds / SECONDS_PER_YEAR
    if name not in (BBH_CLASS, IMBH_CLASS):
        return draw.expected_per_class.get(name, 0.0) / span_years

    black_hole_rate = draw.expected_per_class.get(BBH_CLASS, 0.0) / span_years
    if black_hole_rate_split is not None:
        return black_hole_rate * float(black_hole_rate_split.get(name, 0.0))
    black_holes = pooled_drawn.get(BBH_CLASS, 0) + pooled_drawn.get(IMBH_CLASS, 0)
    if black_holes == 0:
        return 0.0
    return black_hole_rate * pooled_drawn.get(name, 0) / black_holes


@dataclass(frozen=True, slots=True)
class CatalogueBandAnchor:
    """Band membership counted over every catalogue row, not over a draw.

    Attributes:
        band_detector_frame_chirp_mass: Lower edge of the band.
        class_counts: Per-class counts over the full catalogues, in
            :data:`CLASS_ORDER`.
        redshift_reach: Largest redshift over all catalogue rows.
    """

    band_detector_frame_chirp_mass: float
    class_counts: tuple[ClassBandCounts, ...]
    redshift_reach: float | None


def catalogue_band_anchor(
    resolved: Sequence[ResolvedCatalogue],
    *,
    band_detector_frame_chirp_mass: float,
    imbh_total_mass_threshold: float,
) -> CatalogueBandAnchor:
    """Count the band composition of every catalogue row.

    A draw measures the composition of a sample; this measures it over the whole
    published product. The two are not the same number, and reporting them
    together is the check that the draw samples the catalogue rather than
    something shaped like it -- the draw's spread should contain the catalogue
    count, and a systematic offset is a sampling bug rather than a fluctuation.

    Args:
        resolved: The verified catalogue files to count.
        band_detector_frame_chirp_mass: Lower edge of the band.
        imbh_total_mass_threshold: Source-frame total mass at or above which a
            black-hole row is labelled intermediate-mass.

    Returns:
        The full-catalogue counts and redshift reach.

    Raises:
        PopulationValidationError: If a catalogue cannot be read.
    """
    class_totals: dict[str, int] = {}
    class_in_band: dict[str, int] = {}
    redshift_max: float | None = None
    for catalogue in resolved:
        columns = read_catalogue_columns(
            catalogue.path,
            columns=("m1_source", "m2_source", "z"),
        )
        mass_1 = columns["m1_source"]
        mass_2 = columns["m2_source"]
        redshift = columns["z"]
        detector_frame_chirp_mass = chirp_mass(mass_1, mass_2) * (1.0 + redshift)
        in_band = detector_frame_chirp_mass > band_detector_frame_chirp_mass
        source_total_mass = mass_1 + mass_2

        source_class = catalogue.catalogue.source_class
        class_totals[source_class] = class_totals.get(source_class, 0) + int(mass_1.shape[0])
        class_in_band[source_class] = class_in_band.get(source_class, 0) + int(np.count_nonzero(in_band))
        if source_class == BBH_CLASS:
            # The intermediate-mass label splits the black-hole catalogue rather
            # than adding to it: a heavy row is intermediate-mass and not also
            # black-hole, or it would be counted in both shares.
            heavy = source_total_mass >= imbh_total_mass_threshold
            heavy_rows = int(np.count_nonzero(heavy))
            heavy_in_band = int(np.count_nonzero(heavy & in_band))
            class_totals[IMBH_CLASS] = class_totals.get(IMBH_CLASS, 0) + heavy_rows
            class_in_band[IMBH_CLASS] = class_in_band.get(IMBH_CLASS, 0) + heavy_in_band
            class_totals[BBH_CLASS] -= heavy_rows
            class_in_band[BBH_CLASS] -= heavy_in_band
        if redshift.size:
            current = float(np.max(redshift))
            redshift_max = current if redshift_max is None else max(redshift_max, current)

    total_in_band = sum(class_in_band.values())
    class_counts = tuple(
        ClassBandCounts(
            name=name,
            drawn=class_totals[name],
            in_band=class_in_band[name],
            in_band_fraction=class_in_band[name] / class_totals[name] if class_totals[name] else 0.0,
            share_of_band=class_in_band[name] / total_in_band if total_in_band else 0.0,
        )
        for name in CLASS_ORDER
        if name in class_totals
    )
    return CatalogueBandAnchor(
        band_detector_frame_chirp_mass=band_detector_frame_chirp_mass,
        class_counts=class_counts,
        redshift_reach=redshift_max,
    )


__all__ = [
    "CLASS_ORDER",
    "QUANTILES",
    "BandComposition",
    "CatalogueBandAnchor",
    "ClassBandCounts",
    "catalogue_band_anchor",
    "composition_summary",
    "wilson_lower_bound",
]
