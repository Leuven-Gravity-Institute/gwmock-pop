"""Population-catalogue validation and CBC canonicalization."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gwmock_pop.constants import CBC_PARAMETER_NAMES
from gwmock_pop.exceptions import PopulationValidationError

_CBC_SOURCE_TYPES = frozenset({"bbh", "bns", "nsbh"})
_CBC_REQUIRED_COLUMNS = frozenset({"detector_frame_mass_1", "detector_frame_mass_2"})
_CBC_INTERNAL_COLUMNS = frozenset({"source_frame_mass_1", "source_frame_mass_2"})
_CBC_CANONICAL_NAME_MAP = {
    "m1": "detector_frame_mass_1",
    "mass1": "detector_frame_mass_1",
    "mass_1": "detector_frame_mass_1",
    "detector_frame_mass_1": "detector_frame_mass_1",
    "m2": "detector_frame_mass_2",
    "mass2": "detector_frame_mass_2",
    "mass_2": "detector_frame_mass_2",
    "detector_frame_mass_2": "detector_frame_mass_2",
    "m1_source": "source_frame_mass_1",
    "mass1_source": "source_frame_mass_1",
    "mass_1_source": "source_frame_mass_1",
    "srcmass1": "source_frame_mass_1",
    "source_frame_mass_1": "source_frame_mass_1",
    "m2_source": "source_frame_mass_2",
    "mass2_source": "source_frame_mass_2",
    "mass_2_source": "source_frame_mass_2",
    "srcmass2": "source_frame_mass_2",
    "source_frame_mass_2": "source_frame_mass_2",
    "chi1x": "spin_1x",
    "chi1y": "spin_1y",
    "chi1z": "spin_1z",
    "chi2x": "spin_2x",
    "chi2y": "spin_2y",
    "chi2z": "spin_2z",
    "spin1x": "spin_1x",
    "spin1y": "spin_1y",
    "spin1z": "spin_1z",
    "spin2x": "spin_2x",
    "spin2y": "spin_2y",
    "spin2z": "spin_2z",
    "dL": "distance",
    "luminosity_distance": "distance",
    "Phicoal": "coa_phase",
    "iota": "inclination",
    "tGPS": "coa_time",
    "tc": "coa_time",
    "ra": "right_ascension",
    "dec": "declination",
    "polarization": "polarization_angle",
    "psi": "polarization_angle",
    "z": "redshift",
}


def validate_population_catalogue(source_type: str, catalogue: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate a loaded population catalogue for the configured source type."""
    normalized_catalogue = {name: np.atleast_1d(np.asarray(values)) for name, values in catalogue.items()}
    _validate_array_shapes_and_lengths(normalized_catalogue)

    if source_type.lower() not in _CBC_SOURCE_TYPES:
        _validate_numeric_values(normalized_catalogue)
        return normalized_catalogue

    canonical_catalogue = _canonicalize_cbc_catalogue(normalized_catalogue, source_type=source_type)
    canonical_catalogue = _derive_detector_frame_masses(canonical_catalogue)
    _validate_array_shapes_and_lengths(canonical_catalogue)
    _validate_numeric_values(canonical_catalogue)

    public_catalogue = {name: values for name, values in canonical_catalogue.items() if name in CBC_PARAMETER_NAMES}
    missing_columns = sorted(_CBC_REQUIRED_COLUMNS - set(public_catalogue))
    if missing_columns:
        missing_repr = ", ".join(missing_columns)
        raise PopulationValidationError(
            f"CBC population for source_type {source_type!r} is missing required canonical columns: {missing_repr}."
        )
    return public_catalogue


def _canonicalize_cbc_catalogue(catalogue: Mapping[str, np.ndarray], *, source_type: str) -> dict[str, np.ndarray]:
    """Canonicalize CBC aliases to the gwmock-pop public parameter names."""
    canonical_catalogue: dict[str, np.ndarray] = {}
    provenance_by_target: dict[str, str] = {}
    valid_targets = CBC_PARAMETER_NAMES | _CBC_INTERNAL_COLUMNS

    for original_name, values in catalogue.items():
        target_name = _CBC_CANONICAL_NAME_MAP.get(original_name, original_name)
        if target_name not in valid_targets:
            raise PopulationValidationError(
                f"Column {original_name!r} is not recognized for CBC source_type {source_type!r}."
            )
        if target_name in canonical_catalogue:
            previous_name = provenance_by_target[target_name]
            raise PopulationValidationError(
                f"Columns {previous_name!r} and {original_name!r} both map to canonical name {target_name!r}."
            )
        provenance_by_target[target_name] = original_name
        canonical_catalogue[target_name] = values

    return canonical_catalogue


def _derive_detector_frame_masses(catalogue: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derive detector-frame masses from source-frame masses and redshift when needed."""
    derived_catalogue = dict(catalogue)
    redshift = derived_catalogue.get("redshift")

    if "detector_frame_mass_1" not in derived_catalogue:
        source_mass_1 = derived_catalogue.get("source_frame_mass_1")
        if source_mass_1 is None or redshift is None:
            raise PopulationValidationError(
                "Missing detector_frame_mass_1. Provide 'detector_frame_mass_1' directly or provide "
                "'source_frame_mass_1' together with 'redshift'."
            )
        _require_matching_lengths("source_frame_mass_1", source_mass_1, "redshift", redshift)
        derived_catalogue["detector_frame_mass_1"] = source_mass_1 * (1.0 + redshift)

    if "detector_frame_mass_2" not in derived_catalogue:
        source_mass_2 = derived_catalogue.get("source_frame_mass_2")
        if source_mass_2 is None or redshift is None:
            raise PopulationValidationError(
                "Missing detector_frame_mass_2. Provide 'detector_frame_mass_2' directly or provide "
                "'source_frame_mass_2' together with 'redshift'."
            )
        _require_matching_lengths("source_frame_mass_2", source_mass_2, "redshift", redshift)
        derived_catalogue["detector_frame_mass_2"] = source_mass_2 * (1.0 + redshift)

    return derived_catalogue


def _validate_array_shapes_and_lengths(catalogue: Mapping[str, np.ndarray]) -> None:
    """Validate that every catalogue column is a non-empty 1-D array of matching length."""
    if not catalogue:
        raise PopulationValidationError("Loaded catalogue is empty.")

    expected_length: int | None = None
    for name, values in catalogue.items():
        if values.ndim != 1:
            raise PopulationValidationError(f"Column {name!r} must be a 1-D array, got shape {values.shape}.")
        if expected_length is None:
            expected_length = len(values)
            if expected_length == 0:
                raise PopulationValidationError(f"Column {name!r} is empty.")
            continue
        if len(values) != expected_length:
            raise PopulationValidationError(
                f"Column {name!r} has length {len(values)} but expected {expected_length} to match the catalogue."
            )


def _validate_numeric_values(catalogue: Mapping[str, np.ndarray]) -> None:
    """Validate that every catalogue column is numeric and finite."""
    for name, values in catalogue.items():
        if not np.issubdtype(values.dtype, np.number):
            raise PopulationValidationError(f"Column {name!r} must contain numeric values, got dtype {values.dtype}.")
        if not np.all(np.isfinite(values)):
            invalid_index = int(np.flatnonzero(~np.isfinite(values))[0])
            invalid_value = values[invalid_index]
            raise PopulationValidationError(
                f"Column {name!r} contains a non-finite value at index {invalid_index}: {invalid_value!r}."
            )


def _require_matching_lengths(
    left_name: str,
    left_values: np.ndarray,
    right_name: str,
    right_values: np.ndarray,
) -> None:
    """Validate that two arrays can participate in element-wise derivations."""
    if len(left_values) != len(right_values):
        raise PopulationValidationError(
            f"Cannot derive values because {left_name!r} has length {len(left_values)} but "
            f"{right_name!r} has length {len(right_values)}."
        )


__all__ = ["validate_population_catalogue"]
