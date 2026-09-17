"""Seeded draws from published merger catalogues.

The catalogues are fetched on demand and verified against a pinned digest rather
than shipped with the package, so an installation stays light while a draw is
still reproducible from the published product. See
:mod:`gwmock_pop.catalogue.manifest` for the pinned files and
:mod:`gwmock_pop.catalogue.population` for the draw.
"""

from __future__ import annotations

from gwmock_pop.catalogue.composition import (
    BandComposition,
    CatalogueBandAnchor,
    ClassBandCounts,
    catalogue_band_anchor,
    composition_summary,
)
from gwmock_pop.catalogue.manifest import (
    CATALOGUE_CITATION,
    COBA_BBH,
    COBA_BNS,
    COBA_CATALOGUES,
    TDS_CATALOGUE_DOCUMENT,
    TDS_CATALOGUE_RECORD,
    CatalogueFile,
)
from gwmock_pop.catalogue.population import (
    PARAMETER_NAMES,
    CatalogueDraw,
    draw_catalogue_population,
    draw_catalogue_population_many,
)
from gwmock_pop.catalogue.source import ResolvedCatalogue, fetch_catalogue, fetch_catalogues

__all__ = [
    "CATALOGUE_CITATION",
    "COBA_BBH",
    "COBA_BNS",
    "COBA_CATALOGUES",
    "PARAMETER_NAMES",
    "TDS_CATALOGUE_DOCUMENT",
    "TDS_CATALOGUE_RECORD",
    "BandComposition",
    "CatalogueBandAnchor",
    "CatalogueDraw",
    "CatalogueFile",
    "ClassBandCounts",
    "ResolvedCatalogue",
    "catalogue_band_anchor",
    "composition_summary",
    "draw_catalogue_population",
    "draw_catalogue_population_many",
    "fetch_catalogue",
    "fetch_catalogues",
]
