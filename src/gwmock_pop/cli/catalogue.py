"""Draw a population from a published catalogue and report its band composition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from gwmock_pop.catalogue.composition import (
    BandComposition,
    CatalogueBandAnchor,
    catalogue_band_anchor,
    composition_summary,
)
from gwmock_pop.catalogue.manifest import (
    CATALOGUE_SOURCE_CITATION,
    CITATION_ROLES,
    COBA_CATALOGUES,
    SCIENCE_REFERENCE_CITATION,
    TDS_CATALOGUE_DOCUMENT,
    TDS_CATALOGUE_RECORD,
    TDS_SENSITIVITY_DOCUMENT,
    TDS_SENSITIVITY_RECORD,
)
from gwmock_pop.catalogue.population import (
    BBH_CLASS,
    IMBH_CLASS,
    PARAMETER_NAMES,
    CatalogueDraw,
    draw_catalogue_population_many,
)
from gwmock_pop.loaders.file_loader import infer_population_file_format, write_population_catalogue
from gwmock_pop.provenance import build_provenance_record, catalogue_draw_origin, run_metadata

_WRITER = "gwmock_pop.cli.catalogue.catalogue_command"

#: Observation span of the reference draw: fifty 512 s segments.
_REFERENCE_SPAN_SECONDS = 50 * 512.0

#: Number of draws whose spread is reported.
_DEFAULT_DRAWS = 10


def _fetch_block(draw: CatalogueDraw) -> list[dict[str, Any]]:
    """Return one provenance block per input catalogue.

    Args:
        draw: The draw whose input files are described.

    Returns:
        A list of file blocks, each naming the pinned digest the bytes matched.
    """
    return [
        {
            "name": resolved.catalogue.name,
            "source_class": resolved.catalogue.source_class,
            "filename": resolved.catalogue.filename,
            "url": resolved.catalogue.url,
            "sha256": resolved.catalogue.sha256,
            "size_bytes": resolved.catalogue.size_bytes,
            "n_rows": resolved.catalogue.n_rows,
            "verified": bool(resolved.metadata.get("verified", False)),
            "cache_hit": bool(resolved.metadata.get("cache_hit", False)),
        }
        for resolved in draw.resolved
    ]


def _draw_configuration(draw: CatalogueDraw) -> dict[str, Any]:
    """Return the draw settings as a provenance configuration block.

    Args:
        draw: The draw whose settings are described.

    Returns:
        The configuration, with the catalogue order made explicit.
    """
    return {
        "seed": draw.seed,
        "span_seconds": draw.span_seconds,
        "rate_multiplier": draw.rate_multiplier,
        "band_detector_frame_chirp_mass": draw.band_detector_frame_chirp_mass,
        "imbh_total_mass_threshold": draw.imbh_total_mass_threshold,
        "catalogue_order": [resolved.catalogue.name for resolved in draw.resolved],
    }


def _catalogue_block() -> dict[str, Any]:
    """Return the published-catalogue identity for a provenance record.

    Every citation is listed with the role it plays, so the source paper that
    publishes the catalogue files cannot be read as the reference for the
    detector configuration the draw serves, or the reverse.

    Returns:
        The record id, document code, role-labelled citations and sensitivity
        references.
    """
    return {
        "record": TDS_CATALOGUE_RECORD,
        "document": TDS_CATALOGUE_DOCUMENT,
        "citations": [
            {
                "reference": CATALOGUE_SOURCE_CITATION,
                "role": "catalogue_source_paper",
                "role_description": CITATION_ROLES["catalogue_source_paper"],
            },
            {
                "reference": SCIENCE_REFERENCE_CITATION,
                "role": "science_reference",
                "role_description": CITATION_ROLES["science_reference"],
            },
        ],
        "sensitivity_record": TDS_SENSITIVITY_RECORD,
        "sensitivity_document": TDS_SENSITIVITY_DOCUMENT,
    }


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchor_payload(anchor: CatalogueBandAnchor) -> dict[str, Any]:
    """Return the full-catalogue band anchor as a JSON-ready mapping.

    Args:
        anchor: The catalogue-level counts.

    Returns:
        The anchor, with per-class counts and shares.
    """
    return {
        "band_detector_frame_chirp_mass": anchor.band_detector_frame_chirp_mass,
        "redshift_reach": anchor.redshift_reach,
        "class_counts": [
            {
                "name": counts.name,
                "drawn": counts.drawn,
                "in_band": counts.in_band,
                "in_band_fraction": counts.in_band_fraction,
                "share_of_band": counts.share_of_band,
            }
            for counts in anchor.class_counts
        ],
    }


def _black_hole_rate_split(anchor: CatalogueBandAnchor) -> dict[str, float] | None:
    """Return the exact split of the black-hole rate between BBH and IMBH.

    The split is measured over every black-hole catalogue row, so the reported
    BBH and IMBH rates partition the black-hole rate exactly instead of the
    IMBH figure being a noisy per-draw fraction. Returns ``None`` when the
    catalogues hold no black-hole class, leaving the draw's own split in place.

    Args:
        anchor: The full-catalogue counts.

    Returns:
        The fraction of the black-hole rate per black-hole-derived class, or
        ``None`` when no black-hole class is present.
    """
    black_hole_classes = [
        counts for counts in anchor.class_counts if counts.name in (BBH_CLASS, IMBH_CLASS) and counts.drawn > 0
    ]
    total = sum(counts.drawn for counts in black_hole_classes)
    if total == 0:
        return None
    return {counts.name: counts.drawn / total for counts in black_hole_classes}


def _render_composition(composition: BandComposition, anchor: CatalogueBandAnchor | None = None) -> str:
    """Render the composition as a plain-text table.

    Args:
        composition: The measured composition.
        anchor: The full-catalogue anchor, when one was computed. Its in-band
            share is printed next to the draw's so an offset is visible.

    Returns:
        The table, as text.
    """
    header = f"{'class':<6}{'drawn':>10}{'in band':>10}{'in-band frac':>14}{'share of band':>15}"
    lines = [f"Band: detector-frame chirp mass > {composition.band_detector_frame_chirp_mass:g} M_sun"]
    lines.append(
        "Rates [1/yr]: " + ", ".join(f"{name}={rate:.4g}" for name, rate in composition.drawn_rate_per_year.items())
    )
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for counts in composition.class_counts:
        lines.append(
            f"{counts.name:<6}{counts.drawn:>10d}{counts.in_band:>10d}"
            f"{counts.in_band_fraction:>14.4f}{counts.share_of_band:>15.4f}"
        )
    lines.append("")
    lines.append(
        f"BNS share of band: {composition.share_of_band_mean.get('BNS', 0.0):.4f} "
        f"+- {composition.share_of_band_std.get('BNS', 0.0):.4f} over {composition.n_draws} draws "
        f"(min {composition.share_of_band_min.get('BNS', 0.0):.4f}, "
        f"max {composition.share_of_band_max.get('BNS', 0.0):.4f})"
    )
    lines.append(f"BNS share, 90% Wilson lower bound: {composition.bns_share_of_band_lower_90:.4f}")
    lines.append(f"Redshift reach: {composition.redshift_reach}")
    for name, bounds in composition.detector_frame_chirp_mass_range.items():
        lines.append(f"Detector-frame Mc range [{name}]: {bounds}")
    for name, bounds in composition.source_frame_chirp_mass_range.items():
        lines.append(f"Source-frame Mc range [{name}]: {bounds}")
    if composition.in_band_duration_quantiles:
        durations = ", ".join(f"{key}={value:.3g}s" for key, value in composition.in_band_duration_quantiles.items())
        lines.append(f"In-band duration quantiles: {durations}")
    if anchor is not None:
        lines.append("")
        anchor_header = (
            f"{'anchor (full catalogue)':<26}{'rows':>12}{'in band':>12}{'in-band frac':>14}{'share of band':>15}"
        )
        lines.append(anchor_header)
        lines.append("-" * len(anchor_header))
        for counts in anchor.class_counts:
            lines.append(
                f"{counts.name:<26}{counts.drawn:>12d}{counts.in_band:>12d}"
                f"{counts.in_band_fraction:>14.4f}{counts.share_of_band:>15.4f}"
            )
        lines.append(f"Anchor redshift reach: {anchor.redshift_reach}")
    return "\n".join(lines)


def _validate_persistence_options(*, output: Path | None, stamp: Path | None) -> None:
    """Refuse persistence options that cannot describe a persisted draw.

    The destination format and the relationship between the two destinations are
    settled before the population is drawn, so a bad option costs no fetch and
    no draw instead of failing once the work is already done.

    Args:
        output: Destination draw file, or ``None`` when no draw is archived.
        stamp: Destination stamp file, or ``None``.

    Raises:
        ValueError: If ``output`` names a format the writer cannot produce, if
            ``stamp`` is given without ``output``, or if the two name the same
            file.
    """
    if output is not None:
        infer_population_file_format(output)
    if stamp is not None and output is None:
        raise ValueError("--stamp needs --output: the stamp binds the draw's digest to its composition.")
    if stamp is not None and output is not None and output.expanduser().resolve() == stamp.expanduser().resolve():
        raise ValueError("--output and --stamp must name different files: the stamp is not the draw.")


def _write_draw(
    *,
    output_path: Path,
    draw: CatalogueDraw,
    overwrite: bool,
) -> None:
    """Write one draw with its provenance record.

    Args:
        output_path: Destination catalogue file.
        draw: The draw to persist.
        overwrite: Whether an existing destination may be replaced.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output file {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = build_provenance_record(
        origin=catalogue_draw_origin(
            catalogue=_catalogue_block(),
            files=_fetch_block(draw),
            configuration=_draw_configuration(draw),
        ),
        source_type=None,
        parameter_names=list(PARAMETER_NAMES),
        n_samples=draw.n_rows,
        file_format=infer_population_file_format(output_path),
        writer=_WRITER,
        run=run_metadata(name="catalogue-draw", seed=draw.seed, seed_source="cli"),
    )
    write_population_catalogue(
        output_path=output_path,
        population=draw.parameters,
        provenance=record,
        compression="gzip",
    )


def _stamp_payload(
    *,
    draw: CatalogueDraw,
    output_path: Path,
    composition: BandComposition,
    anchor: CatalogueBandAnchor,
) -> dict[str, Any]:
    """Assemble the stamp written beside a draw.

    Args:
        draw: The archived draw.
        output_path: Where the draw was written.
        composition: The measured composition.
        anchor: The full-catalogue band anchor.

    Returns:
        A JSON-ready mapping tying the draw's digest to the composition.
    """
    return {
        "draw": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "n_rows": draw.n_rows,
            "in_band_rows": draw.in_band_rows,
            "seed": draw.seed,
        },
        "inputs": _fetch_block(draw),
        "configuration": _draw_configuration(draw),
        "composition": composition.to_dict(),
        "catalogue_anchor": _anchor_payload(anchor),
    }


def catalogue_command(  # noqa: PLR0913, PLR0917  # the draw's settings, surfaced one option each
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Destination .csv, .h5, or .hdf5 file for the archived draw."),
    ] = None,
    stamp: Annotated[
        Path | None,
        typer.Option("--stamp", help="Optional JSON file for the composition stamp beside the draw."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed of the archived draw.")] = 42,
    draws: Annotated[
        int,
        typer.Option("--draws", min=1, help="Number of draws whose composition spread is reported."),
    ] = _DEFAULT_DRAWS,
    span_seconds: Annotated[
        float,
        typer.Option("--span-seconds", min=0.0, help="Observation span each draw covers, in seconds."),
    ] = _REFERENCE_SPAN_SECONDS,
    band_detector_frame_chirp_mass: Annotated[
        float,
        typer.Option("--band-mc", min=0.0, help="Lower edge of the searched band, detector-frame chirp mass."),
    ] = 5.0,
    imbh_total_mass_threshold: Annotated[
        float,
        typer.Option("--imbh-mass", min=0.0, help="Source-frame total mass at or above which a row is IMBH."),
    ] = 100.0,
    rate_multiplier: Annotated[
        float,
        typer.Option("--rate-multiplier", min=0.0, help="Factor applied to every catalogue's annual rate."),
    ] = 1.0,
    f_low: Annotated[
        float,
        typer.Option("--f-low", min=0.0, help="Observed low-frequency cutoff used for the in-band duration."),
    ] = 5.0,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Cache directory for the on-demand catalogue download."),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-download catalogues even when cached.")] = False,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Use only a cache entry whose digest matches; never download."),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an existing output file.")] = False,
) -> None:
    """Draw the pinned population and report its detector-frame band composition.

    The catalogues are fetched on demand into a local cache and verified against
    their pinned SHA-256, so the package does not ship them and a draw is still
    reproducible. The archived draw is the first seed's; ``--draws`` further
    seeds measure the draw-to-draw spread of the composition.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        _validate_persistence_options(output=output, stamp=stamp)
    except ValueError as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    seeds = [seed + offset for offset in range(draws)]
    try:
        catalogues = tuple(COBA_CATALOGUES)
        all_draws = draw_catalogue_population_many(
            seeds=seeds,
            span_seconds=span_seconds,
            catalogues=catalogues,
            rate_multiplier=rate_multiplier,
            band_detector_frame_chirp_mass=band_detector_frame_chirp_mass,
            imbh_total_mass_threshold=imbh_total_mass_threshold,
            cache_dir=cache_dir,
            refresh=refresh,
            offline=offline,
        )
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    try:
        archived = all_draws[0]
        anchor = catalogue_band_anchor(
            archived.resolved,
            band_detector_frame_chirp_mass=band_detector_frame_chirp_mass,
            imbh_total_mass_threshold=imbh_total_mass_threshold,
        )
        composition = composition_summary(
            all_draws,
            f_low=f_low,
            black_hole_rate_split=_black_hole_rate_split(anchor),
        )
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    try:
        if output is not None:
            _write_draw(output_path=output.expanduser(), draw=archived, overwrite=overwrite)
            logger.info("Wrote %d-row draw to %s", archived.n_rows, output.expanduser())
        if stamp is not None and output is not None:
            payload = _stamp_payload(
                draw=archived,
                output_path=output.expanduser(),
                composition=composition,
                anchor=anchor,
            )
            stamp.expanduser().parent.mkdir(parents=True, exist_ok=True)
            stamp.expanduser().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    typer.echo(_render_composition(composition, anchor))


__all__ = ["catalogue_command"]
