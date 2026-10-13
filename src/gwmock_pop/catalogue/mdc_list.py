"""Reproduce the Einstein Telescope Mock Data Challenge CBC injection lists.

The first ET Mock Data Challenge drew its compact-binary injections from three
source-population catalogues (binary neutron stars, neutron-star--black-hole
binaries and binary black holes, one row per merger) and a table of the
neutron-star tidal deformability against mass. This module regenerates those
lists from the same inputs and settings, event for event, so the injected
population carries a machine-readable record of how it was made.

The generation is a sequential process, and every detail below is part of the
result rather than an implementation choice:

**Each event has its own generator.** Event ``k`` (counting from 1) re-seeds a
Mersenne Twister (MT19937, initialised with the reference ``init_genrand``
routine) with ``seed + k`` and draws uniforms on ``[0, 1)`` as one 32-bit word
divided by ``2**32``. A seed that reduces to zero is replaced by 4357, the
convention of the generator the lists were first made with.

**The draws are consumed in a fixed order.** For each event: the waiting time
since the previous merger (exponential with mean ``mean_interval``, as
``-mean_interval * log(u)``); the class, by one uniform against the cumulative
fractions in the order BNS, BBH, NSBH; for a binary black hole, the azimuths of
the two spin vectors; then, only for an event whose signal starts before the
end of the data, right ascension, the cosine of the polar angle, the cosine of
the inclination, the polarisation angle and the coalescence phase.

**Catalogue rows are read in order.** The ``i``-th event of a class takes the
``i``-th row of that class's catalogue, whether or not the event lands in the
data, so the rows an event consumes depend only on the classes drawn before it.

**Tidal deformability is set for binary neutron stars only.** It is the natural
cubic spline of the tabulated deformability, evaluated at each source-frame
mass. Neutron-star--black-hole and binary-black-hole events carry zero.

**The signal bounds decide placement, not the merger time.** The coalescence
time is the accumulated waiting time. The signal start is the coalescence time
less an upper bound on the chirp duration from ``f_min``; the signal end adds
upper bounds on the merger and ringdown durations. Those bounds decide whether
an event is listed (its signal starts before the end of the data) and which
data segments it is listed in. They are the analytic bounds of the LIGO
Algorithm Library (LAL) and are computed here from the same expressions.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_pop.exceptions import PopulationValidationError
from gwmock_pop.provenance import build_provenance_record, catalogue_draw_origin, run_metadata

#: Class code written in the lists for a binary neutron star.
BNS_TYPE = 1

#: Class code written in the lists for a neutron-star--black-hole binary.
NSBH_TYPE = 2

#: Class code written in the lists for a binary black hole.
BBH_TYPE = 3

#: Source-type routing key of each class code.
SOURCE_TYPES = {BNS_TYPE: "bns", NSBH_TYPE: "nsbh", BBH_TYPE: "bbh"}

#: File names of the inputs, as the challenge's generator distributes them.
CATALOGUE_FILENAMES = {
    "bns": "list_BNS.txt",
    "nsbh": "list_BHNS.txt",
    "bbh": "list_BBH.txt",
    "lambda": "lambda.txt",
}

#: SHA-256 of the input files the first Mock Data Challenge was generated from.
MDC1_CATALOGUE_SHA256 = {
    "bns": "39965cd3e06cc1d81f7b300f77500e90c39dca1658873aec5bcf99adb8710b26",
    "nsbh": "15ce05519f0116f05b7b5f83234044c2939827272737e2700f7b88cb453cbc5e",
    "bbh": "51568edf1a83204cbd6dee997bd03fb9cfa4b735afec4673eaa89c747d7a8a56",
    "lambda": "c6fdc97c6fcda441c382b8aaf55ffb43d41ac2e4c7fcc8fd88d38c8ff9562b69",
}

#: Columns of each catalogue, in file order. Masses are source-frame solar
#: masses, the distance is the luminosity distance in Mpc, spins are
#: dimensionless magnitudes and tilts are in radians.
_CATALOGUE_COLUMNS = {
    BNS_TYPE: ("mass_1", "mass_2", "redshift", "distance"),
    NSBH_TYPE: ("mass_1", "mass_2", "redshift", "distance"),
    BBH_TYPE: ("mass_1", "mass_2", "spin_1", "spin_2", "tilt_1", "tilt_2", "redshift", "distance"),
}

#: Tidal-deformability table: one row per mass, on a grid of ``1.0 + 0.01 i``.
#: The grid is computed rather than read, so the file's mass column is ignored.
_LAMBDA_GRID_ROWS = 151
_LAMBDA_GRID_START = 1.0
_LAMBDA_GRID_STEP = 0.01

#: Uniforms an event can consume: waiting time, class, two spin azimuths and
#: five extrinsic angles.
_DRAWS_PER_EVENT = 9

#: Replacement for a seed that reduces to zero, as in the original generator.
_ZERO_SEED_REPLACEMENT = 4357
_UINT32_RANGE = 2**32

#: Extra time after the signal end that still counts toward a data segment, in seconds.
_SEGMENT_END_PADDING = 3.0

# LAL constants (IAU 2015 nominal solar parameter, CODATA 2018 G), so the
# duration bounds agree with the library the lists were made with.
_LAL_MSUN_SI = 1.9884098706980507e30
_LAL_MRSUN_SI = 1476.6250380501247
_LAL_MTSUN_SI = 4.925490947641267e-06
_LAL_G_SI = 6.6743e-11
_LAL_C_SI = 299792458.0
_MAXIMUM_BLACK_HOLE_SPIN = 0.998
_RINGDOWN_EFOLDS = 11.0


@dataclass(frozen=True, slots=True)
class MdcListSettings:
    """Settings of one list generation, named after the original options.

    Attributes:
        start_time: GPS time of the start of the data, in seconds.
        segment_duration: Duration of one data segment, in seconds.
        n_segments: Number of data segments.
        mean_interval: Mean waiting time between mergers, in seconds.
        bns_fraction: Probability that an event is a binary neutron star.
        bbh_fraction: Probability that an event is a binary black hole. The
            neutron-star--black-hole fraction is the remainder, so the three
            cannot disagree.
        seed: Base seed; event ``k`` is generated with ``seed + k``.
        f_min: Lower frequency of the chirp-duration bound, in hertz.
        min_component_mass: Component mass, in solar masses, of the longest
            signal the run allows for; it fixes how long after the end of the
            data mergers are still drawn.
    """

    start_time: float
    segment_duration: float
    n_segments: int
    mean_interval: float
    bns_fraction: float
    bbh_fraction: float
    seed: int = 100
    f_min: float = 5.0
    min_component_mass: float = 1.1

    @property
    def nsbh_fraction(self) -> float:
        """Return the neutron-star--black-hole fraction, the remainder of the other two.

        Returns:
            ``1 - bns_fraction - bbh_fraction``.
        """
        return 1.0 - self.bns_fraction - self.bbh_fraction

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
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise PopulationValidationError(f"seed must be an integer, got {self.seed!r}.")
        if isinstance(self.n_segments, bool) or not isinstance(self.n_segments, (int, np.integer)):
            raise PopulationValidationError(f"n_segments must be an integer, got {self.n_segments!r}.")
        if self.n_segments < 1:
            raise PopulationValidationError(f"n_segments must be at least 1, got {self.n_segments}.")
        for name in ("segment_duration", "mean_interval", "f_min", "min_component_mass"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise PopulationValidationError(f"{name} must be finite and positive, got {value!r}.")
        if not math.isfinite(self.start_time):
            raise PopulationValidationError(f"start_time must be finite, got {self.start_time!r}.")
        for name in ("bns_fraction", "bbh_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PopulationValidationError(f"{name} must lie in [0, 1], got {value!r}.")
        if self.bns_fraction + self.bbh_fraction > 1.0:
            raise PopulationValidationError(
                f"bns_fraction + bbh_fraction must not exceed 1, got {self.bns_fraction + self.bbh_fraction!r}."
            )


#: Settings of the first Mock Data Challenge: 1300 segments of 2048 s from GPS
#: 1e9, a mean interval of 38.2239 s and its class fractions, with the default
#: seed and lower frequency.
MDC1_SETTINGS = MdcListSettings(
    start_time=1_000_000_000.0,
    segment_duration=2048.0,
    n_segments=1300,
    mean_interval=38.2239,
    bns_fraction=0.8743,
    bbh_fraction=0.0978,
)


@dataclass(frozen=True, slots=True)
class MdcInputFile:
    """One input file, with the identity a generation records.

    Attributes:
        name: Role of the file: ``"bns"``, ``"nsbh"``, ``"bbh"`` or ``"lambda"``.
        path: Local path of the file.
        sha256: SHA-256 of the file contents, lowercase hex.
        size_bytes: Size of the file in bytes.
    """

    name: str
    path: Path
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mdc_inputs(
    directory: str | os.PathLike[str],
    *,
    expected_sha256: Mapping[str, str] | None = MDC1_CATALOGUE_SHA256,
) -> dict[str, MdcInputFile]:
    """Locate and fingerprint the four input files in a directory.

    The catalogues are not distributed with this package; point this at a local
    copy of them.

    Args:
        directory: Directory holding the files under :data:`CATALOGUE_FILENAMES`.
        expected_sha256: Digests the files must match, keyed by role. Defaults to
            the inputs of the first Mock Data Challenge; pass ``None`` to accept
            any files and only record their digests.

    Returns:
        The input files keyed by role.

    Raises:
        PopulationValidationError: If a file is missing or does not match its
            expected digest.
    """
    base = Path(directory)
    inputs: dict[str, MdcInputFile] = {}
    for name, filename in CATALOGUE_FILENAMES.items():
        path = base / filename
        if not path.is_file():
            raise PopulationValidationError(f"Input file {path} does not exist.")
        sha256 = _sha256(path)
        if expected_sha256 is not None and name in expected_sha256 and sha256 != expected_sha256[name]:
            raise PopulationValidationError(
                f"Input file {path} has SHA-256 {sha256}, expected {expected_sha256[name]}."
            )
        inputs[name] = MdcInputFile(name=name, path=path, sha256=sha256, size_bytes=path.stat().st_size)
    return inputs


def _chirp_time_bound(f_min: float, mass_1_kg: np.ndarray, mass_2_kg: np.ndarray, spin: np.ndarray) -> np.ndarray:
    """Return LAL's upper bound on the chirp duration from ``f_min`` to merger.

    The bound is the TaylorT2 duration to second post-Newtonian order, keeping
    only positive corrections and over-estimating the 1.5 post-Newtonian spin
    term with the larger of the two spin magnitudes.
    """
    total_mass = mass_1_kg + mass_2_kg
    eta = (mass_1_kg * mass_2_kg / total_mass) / total_mass
    c0 = np.abs(-5.0 * (total_mass * (_LAL_G_SI / _LAL_C_SI**3)) / (256.0 * eta))
    c2 = 7.43 / 2.52 + 11.0 / 3.0 * eta
    c3 = (226.0 / 15.0) * spin
    c4 = 30.58673 / 5.08032 + 54.29 / 5.04 * eta + 6.17 / 0.72 * eta * eta
    v = np.cbrt(np.pi * _LAL_G_SI * total_mass * f_min) / _LAL_C_SI
    return c0 * v**-8 * (1.0 + (c2 + (c3 + c4 * v) * v) * v * v)


def _final_spin_bound(spin_1: np.ndarray, spin_2: np.ndarray) -> np.ndarray:
    """Return LAL's over-estimate of the remnant spin, capped at 0.998."""
    spin = 0.686 + 0.15 * (spin_1 + spin_2)
    spin = np.maximum(spin, np.abs(spin_1))
    spin = np.maximum(spin, np.abs(spin_2))
    return np.minimum(spin, _MAXIMUM_BLACK_HOLE_SPIN)


def _merge_time_bound(mass_1_kg: np.ndarray, mass_2_kg: np.ndarray) -> np.ndarray:
    """Return LAL's bound on the plunge duration: one orbit at nine gravitational radii."""
    total_mass = mass_1_kg + mass_2_kg
    radius = 9.0 * total_mass * _LAL_MRSUN_SI / _LAL_MSUN_SI
    speed = _LAL_C_SI * np.sqrt(total_mass * _LAL_MRSUN_SI / _LAL_MSUN_SI / radius)
    return 2.0 * np.pi * radius / speed


def _ringdown_time_bound(total_mass_kg: np.ndarray, spin: np.ndarray) -> np.ndarray:
    """Return LAL's bound on the ringdown duration: eleven e-folds of the fundamental mode."""
    omega = (1.5251 + -1.1568 * (1.0 - spin) ** 0.1292) / (total_mass_kg * _LAL_MTSUN_SI / _LAL_MSUN_SI)
    quality = 0.7 + 1.4187 * (1.0 - spin) ** -0.4990
    return _RINGDOWN_EFOLDS * (2.0 * quality / omega)


def _natural_cubic_spline(x_grid: np.ndarray, y_grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic spline through ``(x_grid, y_grid)`` at ``x``.

    The second-derivative system is solved by an ``L D L^T`` factorisation of
    the symmetric tridiagonal matrix and each piece is evaluated in Horner form,
    so the values agree with the reference spline to the last bits rather than
    only to interpolation accuracy.

    Raises:
        PopulationValidationError: If any ``x`` lies outside the grid.
    """
    if np.any((x < x_grid[0]) | (x > x_grid[-1])):
        raise PopulationValidationError(
            f"Masses must lie within the tidal-deformability table [{x_grid[0]}, {x_grid[-1]}]."
        )
    n_interior = len(x_grid) - 2
    h = np.diff(x_grid)
    slope_change = 3.0 * (np.diff(y_grid)[1:] * (1.0 / h[1:]) - np.diff(y_grid)[:-1] * (1.0 / h[:-1]))
    diagonal = 2.0 * (h[1:] + h[:-1])
    off_diagonal = h[1:]

    alpha = np.empty(n_interior)
    gamma = np.empty(n_interior)
    alpha[0] = diagonal[0]
    gamma[0] = off_diagonal[0] / alpha[0]
    for i in range(1, n_interior):
        alpha[i] = diagonal[i] - off_diagonal[i - 1] * gamma[i - 1]
        gamma[i] = off_diagonal[i] / alpha[i]
    forward = np.empty(n_interior)
    forward[0] = slope_change[0]
    for i in range(1, n_interior):
        forward[i] = slope_change[i] - gamma[i - 1] * forward[i - 1]
    scaled = forward / alpha
    interior = np.empty(n_interior)
    interior[-1] = scaled[-1]
    for i in range(n_interior - 2, -1, -1):
        interior[i] = scaled[i] - gamma[i] * interior[i + 1]
    half_curvature = np.concatenate(([0.0], interior, [0.0]))

    index = np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, len(x_grid) - 2)
    x_lo = x_grid[index]
    dx = x_grid[index + 1] - x_lo
    dy = y_grid[index + 1] - y_grid[index]
    c_lo = half_curvature[index]
    c_hi = half_curvature[index + 1]
    b = dy / dx - dx * (c_hi + 2.0 * c_lo) / 3.0
    d = (c_hi - c_lo) / (3.0 * dx)
    offset = x - x_lo
    return y_grid[index] + offset * (b + offset * (c_lo + offset * d))


def _read_lambda_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the tidal-deformability table onto its fixed mass grid.

    Raises:
        PopulationValidationError: If the table does not hold the expected rows.
    """
    table = np.loadtxt(path, ndmin=2)
    if table.shape[0] != _LAMBDA_GRID_ROWS or table.shape[1] < 2:  # noqa: PLR2004  # mass and deformability
        raise PopulationValidationError(
            f"Tidal-deformability table {path} must have {_LAMBDA_GRID_ROWS} rows of two columns, "
            f"got shape {table.shape}."
        )
    grid = _LAMBDA_GRID_START + np.arange(_LAMBDA_GRID_ROWS) * _LAMBDA_GRID_STEP
    return grid, table[:, 1]


def _read_catalogue_rows(path: Path, source_type: int, n_rows: int) -> dict[str, np.ndarray]:
    """Read the first ``n_rows`` rows of a class's catalogue.

    Raises:
        PopulationValidationError: If the catalogue holds fewer rows, the wrong
            number of columns or a non-finite value.
    """
    columns = _CATALOGUE_COLUMNS[source_type]
    if n_rows == 0:
        return {name: np.empty(0) for name in columns}
    table = np.loadtxt(path, ndmin=2, max_rows=n_rows)
    if table.shape[0] < n_rows:
        raise PopulationValidationError(
            f"Catalogue {path} holds {table.shape[0]} rows but the run draws {n_rows} {SOURCE_TYPES[source_type]} events."
        )
    if table.shape[1] != len(columns):
        raise PopulationValidationError(f"Catalogue {path} must have {len(columns)} columns, got {table.shape[1]}.")
    if not np.all(np.isfinite(table)):
        raise PopulationValidationError(f"Catalogue {path} holds non-finite values.")
    return {name: table[:, index] for index, name in enumerate(columns)}


def _event_uniforms(generator: np.random.RandomState, seed: int) -> np.ndarray:
    """Return the uniforms an event can consume, from its own seeded generator.

    ``RandomState`` seeded with an integer runs the reference ``init_genrand``,
    and an integer draw over the full 32-bit range consumes exactly one word,
    so ``word / 2**32`` is the original generator's uniform bit for bit.
    """
    seed %= _UINT32_RANGE
    generator.seed(seed or _ZERO_SEED_REPLACEMENT)
    return generator.randint(0, _UINT32_RANGE, size=_DRAWS_PER_EVENT, dtype=np.uint64) / float(_UINT32_RANGE)


@dataclass(frozen=True, slots=True)
class MdcCbcList:
    """One generated injection list.

    Attributes:
        settings: Settings the list was generated with.
        inputs: The input files, keyed by role.
        columns: One 1-D array per column, one entry per listed event, in event
            order. Canonical gwmock-pop parameter names are used where one
            exists; the remaining columns are ``event_id`` (the event counter
            ``k``), ``source_type`` (class code), ``start_time`` and
            ``end_time`` (signal bounds), ``source_frame_mass_1``/``_2``,
            ``spin_1``/``spin_2`` (magnitudes), ``first_segment`` and
            ``stop_segment`` (the event is listed in segments
            ``first_segment <= i < stop_segment``).
        n_events: Events drawn, including those whose signal starts after the
            end of the data and so are not listed.
        rows_consumed: Catalogue rows read per class, which is the number of
            events of that class drawn.
    """

    settings: MdcListSettings
    inputs: Mapping[str, MdcInputFile] = field(repr=False)
    columns: dict[str, np.ndarray] = field(repr=False)
    n_events: int
    rows_consumed: Mapping[str, int]

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

    def population(self, source_type: str) -> dict[str, np.ndarray]:
        """Return one class's events as a canonical CBC population.

        Args:
            source_type: ``"bns"``, ``"nsbh"`` or ``"bbh"``.

        Returns:
            Mapping from canonical parameter name to a 1-D array, in event order.
            Neutron-star--black-hole and binary-black-hole events carry an
            explicit zero tidal deformability.

        Raises:
            ValueError: If ``source_type`` names no class.
        """
        codes = {name: code for code, name in SOURCE_TYPES.items()}
        if source_type not in codes:
            raise ValueError(f"source_type must be one of {sorted(codes)}, got {source_type!r}.")
        selected = self.columns["source_type"] == codes[source_type]
        return {name: self.columns[name][selected] for name in _POPULATION_COLUMNS}

    def provenance(self, source_type: str, *, file_format: str) -> dict[str, Any]:
        """Return the provenance record of one class's population file.

        Args:
            source_type: ``"bns"``, ``"nsbh"`` or ``"bbh"``.
            file_format: Format the population is written in, ``"csv"`` or ``"hdf5"``.

        Returns:
            A record for :func:`gwmock_pop.loaders.write_population_catalogue`,
            naming the input digests, the seed, the class fractions, the mean
            interval, the start time and span, and the package versions.
        """
        population = self.population(source_type)
        configuration = {
            "generator": "gwmock_pop.catalogue.mdc_list.generate_mdc_cbc_list",
            **asdict(self.settings),
            "nsbh_fraction": self.settings.nsbh_fraction,
            "span_seconds": self.settings.n_segments * self.settings.segment_duration,
        }
        origin = catalogue_draw_origin(
            catalogue={
                "name": "ET Mock Data Challenge CBC source catalogues",
                "n_events_drawn": self.n_events,
                "n_events_listed": self.n_listed,
                "rows_consumed": dict(self.rows_consumed),
            },
            files=[
                {"name": item.name, "filename": item.path.name, "sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in self.inputs.values()
            ],
            configuration=configuration,
        )
        origin["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
        return build_provenance_record(
            origin=origin,
            source_type=source_type,
            parameter_names=list(population),
            n_samples=len(population["coa_time"]),
            file_format=file_format,
            writer="gwmock_pop.loaders.file_loader.write_population_catalogue",
            run=run_metadata(name=None, seed=self.settings.seed, seed_source="library"),
        )


#: Canonical columns of a class population, in output order.
_POPULATION_COLUMNS = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
    "redshift",
    "distance",
    "right_ascension",
    "declination",
    "polarization_angle",
    "inclination",
    "coa_phase",
    "coa_time",
)


def _draw_schedule(settings: MdcListSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw every event's coalescence time, class and uniforms.

    Mergers are drawn while the previous coalescence time is no later than the
    end of the data plus the longest chirp the run allows for; the event that
    crosses that time is still drawn.

    Returns:
        Coalescence times, class codes and the per-event uniforms.
    """
    min_mass_kg = np.asarray(_LAL_MSUN_SI * settings.min_component_mass)
    longest_chirp = float(_chirp_time_bound(settings.f_min, min_mass_kg, min_mass_kg, np.asarray(0.0)))
    horizon = settings.stop_time + longest_chirp
    bbh_threshold = settings.bns_fraction + settings.bbh_fraction

    generator = np.random.RandomState()
    coa_times: list[float] = []
    types: list[int] = []
    uniforms: list[np.ndarray] = []
    coa_time = float(settings.start_time)
    k = 0
    while coa_time <= horizon:
        k += 1
        draws = _event_uniforms(generator, int(settings.seed) + k)
        # A zero draw is an infinite wait, which ends the run as it would in the original.
        coa_time += -settings.mean_interval * math.log(draws[0]) if draws[0] > 0.0 else math.inf
        if draws[1] < settings.bns_fraction:
            types.append(BNS_TYPE)
        elif draws[1] < bbh_threshold:
            types.append(BBH_TYPE)
        else:
            types.append(NSBH_TYPE)
        coa_times.append(coa_time)
        uniforms.append(draws)
    return np.asarray(coa_times), np.asarray(types, dtype=np.int64), np.asarray(uniforms).reshape(-1, _DRAWS_PER_EVENT)


_INTRINSIC_COLUMNS = (
    "source_frame_mass_1",
    "source_frame_mass_2",
    "redshift",
    "distance",
    "spin_1",
    "spin_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
)


def _intrinsic_columns(
    source_type: np.ndarray, uniforms: np.ndarray, inputs: Mapping[str, MdcInputFile]
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Fill each event's catalogue parameters, spins and tidal deformability.

    Returns:
        The columns, one entry per drawn event, and the rows read per class.
    """
    columns = {name: np.zeros(len(source_type)) for name in _INTRINSIC_COLUMNS}
    rows_consumed: dict[str, int] = {}
    for code, name in SOURCE_TYPES.items():
        selected = source_type == code
        rows_consumed[name] = int(np.count_nonzero(selected))
        rows = _read_catalogue_rows(inputs[name].path, code, rows_consumed[name])
        for column in ("redshift", "distance"):
            columns[column][selected] = rows[column]
        columns["source_frame_mass_1"][selected] = rows["mass_1"]
        columns["source_frame_mass_2"][selected] = rows["mass_2"]
        if code == BBH_TYPE:
            # Uniforms 2 and 3 are the spin azimuths.
            for index in (1, 2):
                magnitude, tilt = rows[f"spin_{index}"], rows[f"tilt_{index}"]
                azimuth = 2.0 * np.pi * uniforms[selected, index + 1]
                columns[f"spin_{index}"][selected] = magnitude
                columns[f"spin_{index}x"][selected] = magnitude * np.sin(tilt) * np.cos(azimuth)
                columns[f"spin_{index}y"][selected] = magnitude * np.sin(tilt) * np.sin(azimuth)
                columns[f"spin_{index}z"][selected] = magnitude * np.cos(tilt)
        if code == BNS_TYPE:
            grid, deformability = _read_lambda_table(inputs["lambda"].path)
            columns["lambda_1"][selected] = _natural_cubic_spline(grid, deformability, rows["mass_1"])
            columns["lambda_2"][selected] = _natural_cubic_spline(grid, deformability, rows["mass_2"])
    return columns, rows_consumed


def _signal_bounds(columns: Mapping[str, np.ndarray], coa_time: np.ndarray, f_min: float) -> dict[str, np.ndarray]:
    """Return detector-frame masses, the coalescence time and the signal bounds.

    Returns:
        Columns ``detector_frame_mass_1``/``_2``, ``coa_time``, ``start_time`` and ``end_time``.
    """
    one_plus_z = 1.0 + columns["redshift"]
    mass_1 = columns["source_frame_mass_1"] * one_plus_z
    mass_2 = columns["source_frame_mass_2"] * one_plus_z
    mass_1_kg = _LAL_MSUN_SI * mass_1
    mass_2_kg = _LAL_MSUN_SI * mass_2
    spin_1, spin_2 = columns["spin_1"], columns["spin_2"]
    larger_spin = np.abs(np.where(np.abs(spin_1) > np.abs(spin_2), spin_1, spin_2))
    chirp = _chirp_time_bound(f_min, mass_1_kg, mass_2_kg, larger_spin)
    final_spin = _final_spin_bound(spin_1, spin_2)
    tail = _merge_time_bound(mass_1_kg, mass_2_kg) + _ringdown_time_bound(mass_1_kg + mass_2_kg, final_spin)
    return {
        "detector_frame_mass_1": mass_1,
        "detector_frame_mass_2": mass_2,
        "coa_time": coa_time,
        "start_time": coa_time - chirp,
        "end_time": coa_time + tail,
    }


def _extrinsic_angles(source_type: np.ndarray, uniforms: np.ndarray) -> dict[str, np.ndarray]:
    """Return the sky position, orientation and phase of every event.

    Uniforms 2-6 are the five angles, shifted past the two spin azimuths for a
    binary black hole.

    Returns:
        Columns ``right_ascension``, ``declination``, ``inclination``,
        ``polarization_angle`` and ``coa_phase``.
    """
    offset = np.where(source_type == BBH_TYPE, 4, 2)
    events = np.arange(len(source_type))
    draws = [uniforms[events, offset + i] for i in range(5)]
    return {
        "right_ascension": 2.0 * np.pi * draws[0] - np.pi,
        "declination": np.pi / 2.0 - np.arccos(2.0 * draws[1] - 1.0),
        "inclination": np.arccos(draws[2] * 2.0 - 1.0),
        "polarization_angle": 2.0 * np.pi * draws[3],
        "coa_phase": 2.0 * np.pi * draws[4],
    }


def _segment_range(start_time: np.ndarray, end_time: np.ndarray, settings: MdcListSettings) -> dict[str, np.ndarray]:
    """Return the data segments each signal overlaps, padded after its end.

    Returns:
        Columns ``first_segment`` and ``stop_segment``; an event belongs to
        segments ``first_segment <= i < stop_segment``.
    """
    origin, duration = settings.start_time, settings.segment_duration
    first = np.where(start_time < origin, 0.0, np.floor((start_time - origin) / duration))
    stop = np.minimum(np.ceil((end_time + _SEGMENT_END_PADDING - origin) / duration), settings.n_segments)
    return {"first_segment": first.astype(np.int64), "stop_segment": stop.astype(np.int64)}


def generate_mdc_cbc_list(
    settings: MdcListSettings,
    inputs: Mapping[str, MdcInputFile],
) -> MdcCbcList:
    """Generate the CBC injection list for a set of settings and input files.

    Args:
        settings: Run settings; :data:`MDC1_SETTINGS` reproduces the first Mock
            Data Challenge.
        inputs: Input files keyed by role, from :func:`resolve_mdc_inputs`.

    Returns:
        The listed events and the record of how they were generated.

    Raises:
        PopulationValidationError: If the settings are out of range, an input is
            missing or malformed, a catalogue runs out of rows, or a neutron-star
            mass lies outside the tidal-deformability table.
    """
    settings.validate()
    missing = sorted(set(CATALOGUE_FILENAMES) - set(inputs))
    if missing:
        raise PopulationValidationError(f"Missing input files: {', '.join(missing)}.")

    coa_time, source_type, uniforms = _draw_schedule(settings)
    n_events = len(coa_time)

    columns, rows_consumed = _intrinsic_columns(source_type, uniforms, inputs)
    columns.update(_signal_bounds(columns, coa_time, settings.f_min))
    columns.update(_extrinsic_angles(source_type, uniforms))
    columns.update(_segment_range(columns["start_time"], columns["end_time"], settings))
    columns["event_id"] = np.arange(1, n_events + 1, dtype=np.int64)
    columns["source_type"] = source_type

    listed = columns["start_time"] < settings.stop_time
    return MdcCbcList(
        settings=settings,
        inputs=dict(inputs),
        columns={name: values[listed] for name, values in columns.items()},
        n_events=n_events,
        rows_consumed=rows_consumed,
    )


__all__ = [
    "BBH_TYPE",
    "BNS_TYPE",
    "CATALOGUE_FILENAMES",
    "MDC1_CATALOGUE_SHA256",
    "MDC1_SETTINGS",
    "NSBH_TYPE",
    "SOURCE_TYPES",
    "MdcCbcList",
    "MdcInputFile",
    "MdcListSettings",
    "generate_mdc_cbc_list",
    "resolve_mdc_inputs",
]
