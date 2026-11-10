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

from gwmock_pop.catalogue.layouts import LAYOUTS

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

#: Paper that publishes the catalogue files this module pins.
CATALOGUE_SOURCE_CITATION = "https://arxiv.org/abs/2303.15923"

#: Public science reference for the 10 km detector configuration a draw serves.
#:
#: This is not the files' source paper: it is the reference that documents the
#: configuration the draw is made for, and a record carries it so a reader can
#: tell which citation establishes the data product and which establishes the
#: detector configuration. The catalogue files themselves predate it.
SCIENCE_REFERENCE_CITATION = "https://arxiv.org/abs/2503.12263"

#: What each citation in a draw's provenance establishes.
#:
#: A record lists its citations by role so that neither can be read as standing
#: for the other: the source paper publishes the data product, the science
#: reference documents the configuration.
CITATION_ROLES = {
    "catalogue_source_paper": "publishes the catalogue files this module pins",
    "science_reference": "public reference for the 10 km configuration a draw serves",
}


#: Catalogue columns a file may carry beyond the common set, mapped to the
#: output parameter each one is emitted as.
OPTIONAL_COLUMNS: dict[str, str] = {
    "chi1x": "spin_1x",
    "chi1y": "spin_1y",
    "chi2x": "spin_2x",
    "chi2y": "spin_2y",
    "Lambda1": "lambda_1",
    "Lambda2": "lambda_2",
}


@dataclass(frozen=True, slots=True)
class CatalogueArchive:
    """An archive a record publishes, holding a catalogue file as one of its members.

    Attributes:
        filename: File name of the archive.
        url: Download URL of the archive.
        sha256: Expected SHA-256 of the archive, lowercase hex.
        size_bytes: Expected size of the archive in bytes.
    """

    filename: str
    url: str
    sha256: str
    size_bytes: int


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
        extra_columns: Columns the file carries beyond those every catalogue
            carries, each one of :data:`OPTIONAL_COLUMNS`. A draw reads and
            emits them, and refuses the file if one is absent.
        spin_reference_frequency: Frequency in Hz at which the file's spin
            components are defined, or ``None`` when the file carries only
            aligned components, which do not depend on it.
        layout: How the file names and scales its columns, one of
            :data:`gwmock_pop.catalogue.layouts.LAYOUTS`; ``"coba"`` is read as is.
        download_url: Where the file is fetched from, when it is not on the ET
            document server.
        record: The human-readable record the file is published in, when it is
            not on the ET document server.
        citation: The paper that publishes the file, when it is not the
            design-comparison catalogues' :data:`CATALOGUE_SOURCE_CITATION`.
        archive: The archive the file is a member of, when the record publishes
            it inside one; ``filename`` is then the member's path in the archive,
            and ``sha256`` and ``size_bytes`` describe the member.
    """

    name: str
    source_class: str
    filename: str
    sha256: str
    size_bytes: int
    n_rows: int
    extra_columns: tuple[str, ...] = ()
    spin_reference_frequency: float | None = None
    layout: str = "coba"
    download_url: str | None = None
    record: str | None = None
    citation: str | None = None
    archive: CatalogueArchive | None = None

    def __post_init__(self) -> None:
        """Refuse an extra column the draw would not know how to emit, or an unknown layout.

        Raises:
            ValueError: If an extra column is not one of :data:`OPTIONAL_COLUMNS`, or the layout is
                not one of :data:`gwmock_pop.catalogue.layouts.LAYOUTS`.
        """
        unknown = [name for name in self.extra_columns if name not in OPTIONAL_COLUMNS]
        if unknown:
            raise ValueError(f"Catalogue {self.name} declares unknown extra columns: {', '.join(unknown)}.")
        if self.layout not in LAYOUTS:
            raise ValueError(f"Catalogue {self.name} declares unknown layout {self.layout!r}.")

    @property
    def url(self) -> str:
        """Return the URL this file is downloaded from.

        Returns:
            Its archive's URL, when it is an archive member; else its own
            download URL; else the document-server URL, with the file name
            carried in the ``call_file`` query parameter the server expects.
        """
        if self.archive is not None:
            return self.archive.url
        if self.download_url is not None:
            return self.download_url
        return f"{TDS_BASE_URL}?call_file={quote(self.filename)}"

    @property
    def record_url(self) -> str:
        """Return the human-readable record URL.

        Returns:
            The record page URL: the file's own record when it names one, else
            the document-server record of the design-comparison catalogues.
        """
        if self.record is not None:
            return self.record
        return f"{TDS_BASE_URL}?r={TDS_CATALOGUE_RECORD}"


#: Design-comparison binary-neutron-star catalogue (uniform source-frame masses).
COBA_BNS = CatalogueFile(
    name="bns",
    source_class="BNS",
    filename="18321_1yrCatalogBNS.h5",
    sha256="2c984772bb5a618ee66efdc821fe65487b46602516e0131c948a05b43478b3ff",
    size_bytes=93290425,
    n_rows=721341,
    extra_columns=("Lambda1", "Lambda2"),
)

#: Design-comparison binary-black-hole catalogue.
COBA_BBH = CatalogueFile(
    name="bbh",
    source_class="BBH",
    filename="18321_1yrCatalogBBH.h5",
    sha256="5c421a4bbabc33b3c76c0595298de8663b6dc1b2e65713c346abdcac64e5ca65",
    size_bytes=23449120,
    n_rows=118119,
    extra_columns=("chi1x", "chi1y", "chi2x", "chi2y"),
    # The spin components and `iota` are LAL's transform of the file's own
    # `thetaJN`, `phiJL`, `tilt1`, `tilt2` and `phi12` at 2 Hz with detector-frame
    # masses and zero reference phase: over all 118,119 rows that reproduces all
    # seven to at most 2.9e-8 (lalsuite 7.26.16), while 5 Hz misses by up to 1.08
    # and source-frame masses by up to 1.30. A waveform must use the same frequency.
    spin_reference_frequency=2.0,
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

#: Zenodo record of Dupletsa et al. 2026, publishing the populations of their ET study (CC BY 4.0).
DUPLETSA2026_RECORD = "https://zenodo.org/records/21682194"

#: Paper the record belongs to.
DUPLETSA2026_CITATION = "https://arxiv.org/abs/2607.27311"

#: The record's injection archive, holding the four population files.
DUPLETSA2026_INJECTIONS = CatalogueArchive(
    filename="injections.zip",
    url="https://zenodo.org/records/21682194/files/injections.zip?download=1",
    sha256="ad6c1d0a1b2bf9de42c909ac8125a04d459d16be7eb1b8974a7636a268141dba",
    size_bytes=288540114,
)

# Every file is read as one year of mergers, as the paper's Table 2 counts them ("one year of
# observation"). The coalescence times inside the files are not used -- a draw sets its own -- which
# matters for PBHs.h5: its 1,920,000 rows carry times spread over 2030-2039, while the paper and its
# source (Franciolini et al. 2023, Eq. 22) give 1.92e6 mergers per year.

#: Binary neutron stars (SEVN + cosmoRate rates, N(1.33, 0.09) masses, PCP(BSK24) tides).
DUPLETSA2026_BNS = CatalogueFile(
    name="dupletsa2026_bns",
    source_class="BNS",
    filename="injections/BNSs.hdf5",
    sha256="643c30e213ef1e63df5f685719fcc4c3fcd2c876cedd64ad7c2b59e75e65895b",
    size_bytes=42151840,
    n_rows=351205,
    extra_columns=("Lambda1", "Lambda2"),
    layout="gwfish_pandas",
    record=DUPLETSA2026_RECORD,
    citation=DUPLETSA2026_CITATION,
    archive=DUPLETSA2026_INJECTIONS,
)

#: Pop I/II binary black holes and IMBHs (B-POP: isolated and cluster channels).
DUPLETSA2026_BBH = CatalogueFile(
    name="dupletsa2026_bbh",
    source_class="BBH",
    filename="injections/BBHs.h5",
    sha256="ee830ee8212f3c7109d2a08f52ffca58e6cd66ffd7fd5a06764009ee32d6d5fd",
    size_bytes=27979845,
    n_rows=129491,
    extra_columns=("chi1x", "chi1y", "chi2x", "chi2y"),
    # The stored components and iota are LAL's transform of the stored angles at 2 Hz, with
    # detector-frame masses and Phicoal as reference phase (median residual 5e-11); see
    # gwmock_pop.catalogue.layouts.
    spin_reference_frequency=2.0,
    layout="gwfast",
    record=DUPLETSA2026_RECORD,
    citation=DUPLETSA2026_CITATION,
    archive=DUPLETSA2026_INJECTIONS,
)

#: Binary black holes from Pop III stars (SEVN, fastcluster and cosmoRate).
DUPLETSA2026_POP3_BBH = CatalogueFile(
    name="dupletsa2026_pop3_bbh",
    source_class="BBH",
    filename="injections/BBHs_from_POP3.hdf5",
    sha256="9bc491430b29f0f64512f9117c40c291d36750a14eb2961718a21f8a78038074",
    size_bytes=468688,
    n_rows=3393,
    extra_columns=("chi1x", "chi1y", "chi2x", "chi2y"),
    # Spins are given as angles, at a reference frequency the record does not state. 2 Hz is the one
    # BBHs.h5's stored components were computed at, and the components are produced at it.
    spin_reference_frequency=2.0,
    layout="gwfish_pandas",
    record=DUPLETSA2026_RECORD,
    citation=DUPLETSA2026_CITATION,
    archive=DUPLETSA2026_INJECTIONS,
)

#: Primordial black-hole binaries (Franciolini et al. 2023).
DUPLETSA2026_PBH = CatalogueFile(
    name="dupletsa2026_pbh",
    source_class="PBH",
    filename="injections/PBHs.h5",
    sha256="eb65cdb18d73ce3f013e40133ab7849ad93b30819a87b9e2c77661fd2eb53e4d",
    size_bytes=224767249,
    n_rows=1920000,
    extra_columns=("chi1x", "chi1y", "chi2x", "chi2y"),
    # Spin angles, as for the Pop III file.
    spin_reference_frequency=2.0,
    layout="gwfast",
    record=DUPLETSA2026_RECORD,
    citation=DUPLETSA2026_CITATION,
    archive=DUPLETSA2026_INJECTIONS,
)

#: The record's populations, in draw order: binary neutron stars, Pop I/II binary black holes, Pop III
#: binary black holes, primordial black holes. As for the design-comparison files, the order is part
#: of a seeded draw's identity.
DUPLETSA2026_CATALOGUES: tuple[CatalogueFile, ...] = (
    DUPLETSA2026_BNS,
    DUPLETSA2026_BBH,
    DUPLETSA2026_POP3_BBH,
    DUPLETSA2026_PBH,
)


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
    "CATALOGUE_SOURCE_CITATION",
    "CITATION_ROLES",
    "COBA_BBH",
    "COBA_BNS",
    "COBA_CATALOGUES",
    "COBA_CATALOGUES_BY_NAME",
    "DUPLETSA2026_BBH",
    "DUPLETSA2026_BNS",
    "DUPLETSA2026_CATALOGUES",
    "DUPLETSA2026_CITATION",
    "DUPLETSA2026_INJECTIONS",
    "DUPLETSA2026_PBH",
    "DUPLETSA2026_POP3_BBH",
    "DUPLETSA2026_RECORD",
    "SCIENCE_REFERENCE_CITATION",
    "TDS_BASE_URL",
    "TDS_CATALOGUE_DOCUMENT",
    "TDS_CATALOGUE_RECORD",
    "TDS_SENSITIVITY_DOCUMENT",
    "TDS_SENSITIVITY_RECORD",
    "CatalogueArchive",
    "CatalogueFile",
    "catalogue_url",
]
