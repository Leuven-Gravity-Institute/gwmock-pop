"""The pinned identity of the public catalogues a draw is made from.

A draw is only reproducible if the file it was made from can be fetched again
and shown to be the same file. The catalogues this module names are served from
the Einstein Telescope document server by an unversioned URL, so the URL cannot
be the identity: the SHA-256 of the bytes is. Every entry here records that
digest, the size, and the row count, and the fetch refuses bytes that do not
match -- see :func:`gwmock_pop.loaders.resolve_digest_pinned_path`.

The values are the published data product's own, not a transcript of one
download: the digests below were computed from the files the record serves and
cross-checked against an independent local copy of the same product.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

#: Base URL of the ET document server.
TDS_BASE_URL = "https://apps.et-gw.eu/tds/"

#: Document-server record holding the design-comparison source catalogues.
TDS_CATALOGUE_RECORD = 18321

#: Document code of the note that describes those catalogues.
TDS_CATALOGUE_DOCUMENT = "ET-0197A-23"

#: Document-server record holding the 10 km sensitivity data vector.
TDS_SENSITIVITY_RECORD = 18213

#: Document code of the note that describes the sensitivity data vector.
TDS_SENSITIVITY_DOCUMENT = "ET-0304B-22"

#: Published reference for the population the catalogues encode.
CATALOGUE_CITATION = "https://arxiv.org/abs/2303.15923"


@dataclass(frozen=True, slots=True)
class CatalogueFile:
    """One catalogue file, with the identity needed to fetch and verify it.

    Attributes:
        name: Short name used to address the file in code and configs.
        source_class: Source class the file holds, as recorded in a draw.
        filename: File name the document server serves it under.
        sha256: Expected SHA-256 of the file contents, lowercase hex.
        size_bytes: Expected size of the file in bytes.
        n_rows: Number of catalogue rows, a property of the published product.
    """

    name: str
    source_class: str
    filename: str
    sha256: str
    size_bytes: int
    n_rows: int

    @property
    def url(self) -> str:
        """Return the document-server download URL for this file.

        Returns:
            The URL, with the file name carried in the ``call_file`` query
            parameter the server expects.
        """
        return f"{TDS_BASE_URL}?call_file={quote(self.filename)}"

    @property
    def record_url(self) -> str:
        """Return the human-readable document-server record URL.

        Returns:
            The record page URL.
        """
        return f"{TDS_BASE_URL}?r={TDS_CATALOGUE_RECORD}"


#: Design-comparison binary-neutron-star catalogue (uniform source-frame masses).
COBA_BNS = CatalogueFile(
    name="bns",
    source_class="BNS",
    filename="18321_1yrCatalogBNS.h5",
    sha256="2c984772bb5a618ee66efdc821fe65487b46602516e0131c948a05b43478b3ff",
    size_bytes=93290425,
    n_rows=721341,
)

#: Design-comparison binary-black-hole catalogue.
COBA_BBH = CatalogueFile(
    name="bbh",
    source_class="BBH",
    filename="18321_1yrCatalogBBH.h5",
    sha256="5c421a4bbabc33b3c76c0595298de8663b6dc1b2e65713c346abdcac64e5ca65",
    size_bytes=23449120,
    n_rows=118119,
)

#: Catalogue files a design-comparison draw is made from, in draw order.
#:
#: The order is part of the draw's identity: it fixes the order in which the
#: per-class Poisson counts consume the random stream, so reordering it changes
#: every seeded draw. Binary neutron stars come first because that is the order
#: the reference population draw used.
COBA_CATALOGUES: tuple[CatalogueFile, ...] = (COBA_BNS, COBA_BBH)

#: Catalogue files by short name.
COBA_CATALOGUES_BY_NAME: dict[str, CatalogueFile] = {catalogue.name: catalogue for catalogue in COBA_CATALOGUES}


def catalogue_url(name: str) -> str:
    """Return the pinned download URL for a named catalogue.

    Args:
        name: Short catalogue name, such as ``"bbh"``.

    Returns:
        The document-server download URL.

    Raises:
        KeyError: If no catalogue carries that name.
    """
    return COBA_CATALOGUES_BY_NAME[name].url


__all__ = [
    "CATALOGUE_CITATION",
    "COBA_BBH",
    "COBA_BNS",
    "COBA_CATALOGUES",
    "COBA_CATALOGUES_BY_NAME",
    "TDS_BASE_URL",
    "TDS_CATALOGUE_DOCUMENT",
    "TDS_CATALOGUE_RECORD",
    "TDS_SENSITIVITY_DOCUMENT",
    "TDS_SENSITIVITY_RECORD",
    "CatalogueFile",
    "catalogue_url",
]
