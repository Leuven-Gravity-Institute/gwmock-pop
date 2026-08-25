"""Machine-readable provenance for population catalogues.

A catalogue is only publishable if the run behind it can be reconstructed from
the file. :func:`build_provenance_record` assembles that record in one place,
:mod:`gwmock_pop.provenance.io` stores and retrieves it, and
:func:`replay_catalogue` redraws the samples from a record alone.
"""

from __future__ import annotations

from gwmock_pop.provenance.configuration import resolved_configuration_payload
from gwmock_pop.provenance.io import (
    HDF5_METADATA_GROUP,
    HDF5_PROVENANCE_DATASET,
    HDF5_PROVENANCE_PATH,
    PROVENANCE_SIDECAR_SUFFIX,
    encode_provenance,
    provenance_sidecar_path,
    read_provenance,
    write_hdf5_provenance,
    write_provenance_sidecar,
)
from gwmock_pop.provenance.record import (
    CONVERTED_CATALOGUE,
    EXTERNAL_ENGINE,
    GRAPH_SIMULATION,
    PROVENANCE_SCHEMA_VERSION,
    SEED_SOURCES,
    SIMULATOR,
    TOOL_NAME,
    EngineDescription,
    GraphConfigResolution,
    build_provenance_record,
    configuration_hash,
    converted_catalogue_origin,
    external_engine_origin,
    git_source_state,
    graph_simulation_origin,
    resolve_graph_config_defaults,
    run_metadata,
    simulator_origin,
    source_code_provenance,
)
from gwmock_pop.provenance.replay import reconstruct_run, replay_catalogue

__all__ = [
    "CONVERTED_CATALOGUE",
    "EXTERNAL_ENGINE",
    "GRAPH_SIMULATION",
    "HDF5_METADATA_GROUP",
    "HDF5_PROVENANCE_DATASET",
    "HDF5_PROVENANCE_PATH",
    "PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_SIDECAR_SUFFIX",
    "SEED_SOURCES",
    "SIMULATOR",
    "TOOL_NAME",
    "EngineDescription",
    "GraphConfigResolution",
    "build_provenance_record",
    "configuration_hash",
    "converted_catalogue_origin",
    "encode_provenance",
    "external_engine_origin",
    "git_source_state",
    "graph_simulation_origin",
    "provenance_sidecar_path",
    "read_provenance",
    "reconstruct_run",
    "replay_catalogue",
    "resolve_graph_config_defaults",
    "resolved_configuration_payload",
    "run_metadata",
    "simulator_origin",
    "source_code_provenance",
    "write_hdf5_provenance",
    "write_provenance_sidecar",
]
