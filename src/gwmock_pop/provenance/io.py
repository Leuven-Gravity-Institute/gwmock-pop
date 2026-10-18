"""Store and retrieve provenance records alongside population catalogues.

HDF5 files carry their record internally, in a ``metadata`` group beside the
``data`` dataset. CSV files have nowhere to put one, so they get a JSON sidecar
named after the catalogue.

That difference is the reason to prefer HDF5 for anything published: a sidecar
is a separate file, and a catalogue that arrives without it arrives without its
provenance. The sidecar exists so a CSV catalogue is not silently undocumented,
not so CSV can be called publication-grade.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gwmock_pop._optional import require

if TYPE_CHECKING:
    import h5py

# Name of the HDF5 group holding the record.
HDF5_METADATA_GROUP = "metadata"

# Name of the dataset holding the encoded record.
HDF5_PROVENANCE_DATASET = "provenance"

# Path of the encoded record within an HDF5 catalogue.
HDF5_PROVENANCE_PATH = f"{HDF5_METADATA_GROUP}/{HDF5_PROVENANCE_DATASET}"

# Suffix appended to a catalogue's file name to name its sidecar.
PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"

# The eight bytes every HDF5 file starts its superblock with.
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _is_hdf5(path: Path) -> bool:
    """Whether ``path`` is an HDF5 file, read from its signature rather than through h5py.

    The superblock, and so the signature, sits at byte 0 or, after a user block, at 512, 1024, 2048 and
    so on (HDF5 File Format Specification, section II.A). Checking it here means telling a CSV catalogue
    from an HDF5 one needs no h5py, which is an optional dependency.
    """
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        offset = 0
        while offset + len(_HDF5_SIGNATURE) <= size:
            stream.seek(offset)
            if stream.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE:
                return True
            offset = 512 if offset == 0 else offset * 2
    return False


def provenance_sidecar_path(catalogue_path: str | os.PathLike[str]) -> Path:
    """Return the sidecar path for a catalogue.

    The suffix is appended to the whole file name rather than replacing the
    existing one, so a CSV and an HDF5 catalogue with the same stem cannot end
    up sharing a sidecar.

    Args:
        catalogue_path: Path of the catalogue file.

    Returns:
        Path of the sidecar that belongs to it.
    """
    path = Path(catalogue_path)
    return path.with_name(path.name + PROVENANCE_SIDECAR_SUFFIX)


def encode_provenance(record: dict[str, Any]) -> str:
    """Encode a record for storage.

    Keys are never sorted. Graph node order fixes the output column order and
    the order in which samplers consume the random stream, so reordering the
    record would change the run it describes.

    Args:
        record: The record to encode.

    Returns:
        The record as indented JSON text.
    """
    return json.dumps(record, indent=2, ensure_ascii=False, default=str)


def _decode_provenance(payload: str | bytes, *, origin: str) -> dict[str, Any]:
    """Decode a stored record.

    Args:
        payload: Encoded record.
        origin: Description of where it was read from, used in error messages.

    Returns:
        The decoded record.

    Raises:
        ValueError: If the payload is not a JSON object.
    """
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        record = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not decode the provenance record in {origin}: {error}") from error
    if not isinstance(record, dict):
        # A malformed stored record is bad data, not a bad call, so it reads as a
        # value error alongside the decode failure above.
        raise ValueError(f"The provenance record in {origin} is not a JSON object.")  # noqa: TRY004
    return record


def write_hdf5_provenance(handle: h5py.File, record: dict[str, Any]) -> None:
    """Embed a record in an open HDF5 catalogue.

    The record goes into a dataset rather than an attribute: HDF5 attributes are
    size-limited and a resolved configuration is not.

    Args:
        handle: Open, writable HDF5 file.
        record: The record to embed.
    """
    group = handle.require_group(HDF5_METADATA_GROUP)
    group.create_dataset(HDF5_PROVENANCE_DATASET, data=encode_provenance(record))


def write_provenance_sidecar(catalogue_path: str | os.PathLike[str], record: dict[str, Any]) -> Path:
    """Write a record beside a catalogue that cannot hold one.

    Args:
        catalogue_path: Path of the catalogue file.
        record: The record to write.

    Returns:
        Path of the sidecar that was written.
    """
    sidecar = provenance_sidecar_path(catalogue_path)
    sidecar.write_text(encode_provenance(record) + "\n", encoding="utf-8")
    return sidecar


def read_provenance(catalogue_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read the provenance record of a catalogue.

    An embedded record is preferred over a sidecar: it travels with the file it
    describes, so it cannot be the stale one.

    Args:
        catalogue_path: Path of the catalogue file.

    Returns:
        The record, or ``None`` when the catalogue carries none.

    Raises:
        ValueError: If a stored record cannot be decoded.
    """
    path = Path(catalogue_path)
    if path.is_file() and _is_hdf5(path):
        h5py = require("h5py", "hdf5", "Reading the provenance of an HDF5 catalogue")
        with h5py.File(path, "r") as handle:
            stored = handle.get(HDF5_PROVENANCE_PATH)
            if isinstance(stored, h5py.Dataset):
                return _decode_provenance(stored[()], origin=f"{path}/{HDF5_PROVENANCE_PATH}")

    sidecar = provenance_sidecar_path(path)
    if sidecar.is_file():
        return _decode_provenance(sidecar.read_text(encoding="utf-8"), origin=str(sidecar))
    return None
