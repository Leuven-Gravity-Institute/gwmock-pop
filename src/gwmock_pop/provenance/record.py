"""Build the machine-readable provenance record of a population catalogue.

A catalogue without a record of how it was produced cannot be published: the
seed, the configuration and the code that made it are not recoverable from the
samples. Every record this package writes is assembled here, so the two
persistence paths cannot describe the same run in two different ways.

Record layout, version :data:`PROVENANCE_SCHEMA_VERSION`::

    schema_version  the version of this layout
    created_utc     when the file was written, ISO 8601 in UTC
    tool            the package name, version, checkout state and writer
    catalogue       source type, row count, column names in output order, format
    run             what the run actually did: name, seed, how the seed was chosen
    origin          where the samples came from, keyed by ``kind``

``run`` is the authority on the seed and ``catalogue`` on the row count;
``origin.configuration`` is the configuration those values were resolved from,
with defaults filled in. Keeping the two roles apart is deliberate: a declared
configuration and an effective run are different facts, and a reader that
cannot tell them apart cannot tell an override from a default.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from gwmock_pop.utils.import_utils import import_from_string
from gwmock_pop.version import __version__

# Version of the record layout documented in this module.
PROVENANCE_SCHEMA_VERSION = "1.0"

# Name recorded as the producer of a catalogue.
TOOL_NAME = "gwmock-pop"

# Origin kind for a catalogue sampled from a parameter graph by this package.
GRAPH_SIMULATION = "graph_simulation"

# Origin kind for a catalogue from a simulator with no recoverable graph config.
SIMULATOR = "simulator"

# Origin kind for a catalogue produced by another population engine.
EXTERNAL_ENGINE = "external_engine"

# Origin kind for a catalogue derived from another catalogue file.
CONVERTED_CATALOGUE = "converted_catalogue"

# How the seed of a run came to be chosen.
SEED_SOURCES = ("cli", "config", "drawn", "library")

_GIT_TIMEOUT_SECONDS = 10
_GIT_COMMIT_LENGTH = 40
_INJECTED_ARGUMENTS = frozenset({"key", "n_samples"})
_VARIADIC_KINDS = frozenset({inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD})
_JSON_SCALARS = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class EngineDescription:
    """Identity of the engine that produced a catalogue.

    Attributes:
        name: Name of the engine.
        version: Version of the engine, when it reports one.
        config_hash: Digest of the engine's own configuration.
        run_record: Pointer to the engine's own run record, such as a URL or path.
    """

    name: str
    version: str | None = None
    config_hash: str | None = None
    run_record: str | None = None

    def payload(self) -> dict[str, Any]:
        """Return the engine block as it appears in a record.

        Returns:
            Mapping with the engine name, version, config hash and run-record pointer.
        """
        return {
            "name": self.name,
            "version": self.version,
            "config_hash": self.config_hash,
            "run_record": self.run_record,
        }


@dataclass(frozen=True)
class GraphConfigResolution:
    """A parameter graph with its unstated defaults made explicit.

    Attributes:
        config: The graph config with callable defaults filled into ``arguments``.
        unresolved_nodes: Names of nodes whose defaults could not be resolved.
    """

    config: dict[str, Any]
    unresolved_nodes: tuple[str, ...]


def _package_directory() -> Path:
    """Return the directory holding the installed package."""
    return Path(__file__).resolve().parents[1]


def _run_git(*arguments: str) -> str | None:
    """Run a read-only git command against the package directory.

    Args:
        *arguments: Arguments to pass to git.

    Returns:
        Standard output of the command, or ``None`` when git is unavailable or
        the command failed.
    """
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603  # fixed argument list, resolved executable, no shell
            [git_executable, "--no-optional-locks", "-C", str(_package_directory()), *arguments],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


@lru_cache(maxsize=1)
def git_source_state() -> dict[str, Any] | None:
    """Return the commit and dirty flag of the checkout the package runs from.

    The state is read next to the installed package rather than from the working
    directory, so an unrelated repository the process happens to be started in
    is never reported as the source of the code.

    Returns:
        Mapping with the ``commit`` hash and a ``dirty`` flag, or ``None`` when
        the package is not running from a git checkout.
    """
    inside_work_tree = _run_git("rev-parse", "--is-inside-work-tree")
    if inside_work_tree is None or inside_work_tree.strip() != "true":
        return None
    commit = _run_git("rev-parse", "HEAD")
    if commit is None or len(commit.strip()) != _GIT_COMMIT_LENGTH:
        return None
    status = _run_git("status", "--porcelain")
    return {"commit": commit.strip(), "dirty": bool(status.strip())} if status is not None else None


def source_code_provenance() -> dict[str, Any]:
    """Return the block identifying the code that produced a catalogue.

    Returns:
        Mapping with the tool name, the package version and the checkout state.
    """
    return {"name": TOOL_NAME, "version": __version__, "git": git_source_state()}


def configuration_hash(payload: Any) -> str:
    """Return a digest of a configuration payload.

    The digest is order-sensitive. Graph node order fixes both the output column
    order and the order in which the samplers consume the random stream, so two
    configurations differing only in node order are different runs.

    Args:
        payload: JSON-ready configuration payload.

    Returns:
        The digest, prefixed with the name of the algorithm.
    """
    canonical = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _is_json_value(value: Any) -> bool:
    """Return whether a value can be recorded verbatim in a JSON record."""
    if isinstance(value, _JSON_SCALARS):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    return False


def _fill_callable_defaults(block: Mapping[str, Any], *, default_module: str) -> tuple[dict[str, Any], bool]:
    """Make a sampler or transform block's unstated argument defaults explicit.

    Args:
        block: Mapping-style sampler or transform block.
        default_module: Module searched when the block names a bare function.

    Returns:
        The block with defaults filled into ``arguments``, and whether every
        default could be resolved.
    """
    filled_block = dict(block)
    function_name = block.get("function")
    arguments = block.get("arguments")
    if not isinstance(function_name, str) or not function_name:
        return filled_block, False
    if arguments is not None and not isinstance(arguments, Mapping):
        # Positional argument styles are recorded exactly as configured.
        return filled_block, False

    try:
        function = import_from_string(object_path=function_name, default_module=default_module)
        signature = inspect.signature(function)
    except (ImportError, TypeError, ValueError):
        return filled_block, False

    resolved_arguments = dict(arguments or {})
    resolved_every_default = True
    for name, parameter in signature.parameters.items():
        if name in _INJECTED_ARGUMENTS or name in resolved_arguments:
            continue
        if parameter.kind in _VARIADIC_KINDS or parameter.default is inspect.Parameter.empty:
            continue
        if not _is_json_value(parameter.default):
            resolved_every_default = False
            continue
        resolved_arguments[name] = parameter.default

    filled_block["arguments"] = resolved_arguments
    return filled_block, resolved_every_default


def resolve_graph_config_defaults(config: Mapping[str, Any]) -> GraphConfigResolution:
    """Return a parameter graph with every unstated argument default filled in.

    A record of only the values the caller wrote cannot reconstruct the run: the
    rest of the sampler's arguments come from its signature, and a later version
    of the package may choose them differently.

    Args:
        config: Parameter graph mapping node names to their specifications.

    Returns:
        The resolved graph and the names of any nodes whose defaults could not
        be filled in. The input is not modified.
    """
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for node_name, spec in config.items():
        if not isinstance(spec, Mapping):
            resolved[node_name] = spec
            unresolved.append(node_name)
            continue

        node = dict(spec)
        resolved_every_default = False
        for block_name, default_module in (
            ("sampler", "gwmock_pop.samplers"),
            ("transform", "gwmock_pop.transforms"),
        ):
            block = node.get(block_name)
            if not isinstance(block, Mapping):
                continue
            node[block_name], resolved_every_default = _fill_callable_defaults(block, default_module=default_module)

        if not resolved_every_default:
            unresolved.append(node_name)
        resolved[node_name] = node

    return GraphConfigResolution(config=resolved, unresolved_nodes=tuple(unresolved))


def run_metadata(*, name: str | None, seed: int, seed_source: str) -> dict[str, Any]:
    """Return the block describing what a sampling run actually did.

    Args:
        name: Run identifier, or ``None`` when the caller has no name for it.
        seed: The seed the run's generator was actually initialized with, which
            is the drawn value when the caller asked for no particular seed.
        seed_source: One of :data:`SEED_SOURCES`.

    Returns:
        Mapping with the run name, seed and the origin of the seed.

    Raises:
        ValueError: If ``seed_source`` is not one of :data:`SEED_SOURCES`.
    """
    if seed_source not in SEED_SOURCES:
        supported = ", ".join(SEED_SOURCES)
        raise ValueError(f"seed_source must be one of {supported}, got {seed_source!r}.")
    return {"name": name, "seed": int(seed), "seed_source": seed_source}


def graph_simulation_origin(
    *,
    graph_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
    preset: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Return the origin block for a catalogue sampled from a parameter graph.

    Args:
        graph_config: Parameter graph the simulator was built from. Its unstated
            defaults are filled in and the result becomes the ``parameters``
            block of the recorded configuration, so the graph is stored once.
        configuration: Resolved simulation configuration, JSON-ready, with every
            default filled in.
        preset: Name of the packaged preset the run came from, when it did.
        config_path: Path of the configuration file the run came from, when it did.

    Returns:
        The origin block.
    """
    resolution = resolve_graph_config_defaults(graph_config)
    resolved_configuration = dict(configuration)
    resolved_configuration["parameters"] = resolution.config
    return {
        "kind": GRAPH_SIMULATION,
        "engine": {
            "name": TOOL_NAME,
            "version": __version__,
            "component": "gwmock_pop.simulators.graph.GraphSimulator",
        },
        "preset": preset,
        "config_path": config_path,
        "configuration": resolved_configuration,
        "configuration_hash": configuration_hash(resolved_configuration),
        "unresolved_default_nodes": list(resolution.unresolved_nodes),
    }


def simulator_origin(*, component: str, configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the origin block for a simulator with no recoverable graph config.

    Args:
        component: Import path of the simulator class that produced the samples.
        configuration: Whatever configuration the simulator can report, if any.

    Returns:
        The origin block. A record with this kind names the code that produced
        the catalogue but does not describe a replayable run.
    """
    resolved_configuration = dict(configuration) if configuration is not None else None
    return {
        "kind": SIMULATOR,
        "engine": {"name": TOOL_NAME, "version": __version__, "component": component},
        "configuration": resolved_configuration,
        "configuration_hash": None if resolved_configuration is None else configuration_hash(resolved_configuration),
    }


def external_engine_origin(
    *,
    engine: EngineDescription,
    input_path: str | None = None,
    fetch: Mapping[str, Any] | None = None,
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the origin block for a catalogue produced by another engine.

    Args:
        engine: Identity of the producing engine, including a pointer to its own
            run record.
        input_path: Path or URL the catalogue was read from.
        fetch: Fetch details for a remotely retrieved catalogue, as reported by
            :class:`~gwmock_pop.loaders.FilePopulationLoader`.
        upstream: Provenance record found alongside the input catalogue.

    Returns:
        The origin block.
    """
    return {
        "kind": EXTERNAL_ENGINE,
        "engine": engine.payload(),
        "input": _input_block(input_path=input_path, fetch=fetch, upstream=upstream),
    }


def converted_catalogue_origin(
    *,
    input_path: str,
    column_map: Mapping[str, str] | None = None,
    fetch: Mapping[str, Any] | None = None,
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the origin block for a catalogue derived from another catalogue.

    Args:
        input_path: Path or URL of the input catalogue.
        column_map: Column renaming applied on the way through.
        fetch: Fetch details for a remotely retrieved input catalogue.
        upstream: Provenance record found alongside the input catalogue, so a
            conversion extends the chain rather than truncating it.

    Returns:
        The origin block.
    """
    return {
        "kind": CONVERTED_CATALOGUE,
        "input": _input_block(input_path=input_path, fetch=fetch, upstream=upstream),
        "column_map": dict(column_map or {}),
    }


def _input_block(
    *,
    input_path: str | None,
    fetch: Mapping[str, Any] | None,
    upstream: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the block describing the file a catalogue was derived from."""
    return {
        "path": input_path,
        "fetch": dict(fetch) if fetch is not None else None,
        "provenance": dict(upstream) if upstream is not None else None,
    }


def build_provenance_record(  # noqa: PLR0913  # the record's fields, named at its single build site
    *,
    origin: Mapping[str, Any],
    source_type: str | None,
    parameter_names: Sequence[str],
    n_samples: int,
    file_format: str,
    writer: str,
    run: Mapping[str, Any] | None = None,
    created: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record of one catalogue.

    Args:
        origin: Origin block from one of the ``*_origin`` builders in this module.
        source_type: Routing key of the catalogue, such as ``"bbh"``.
        parameter_names: Column names in the order they are written.
        n_samples: Number of rows written.
        file_format: Format the catalogue was written in.
        writer: Import path of the function that wrote the file.
        run: Block from :func:`run_metadata`, for catalogues this package sampled.
        created: Creation time. Defaults to now, in UTC.

    Returns:
        The complete record, ready to be encoded and stored.
    """
    timestamp = (created or datetime.now(tz=UTC)).astimezone(UTC)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": timestamp.isoformat(),
        "tool": {**source_code_provenance(), "writer": writer},
        "catalogue": {
            "source_type": source_type,
            "n_samples": int(n_samples),
            "parameter_names": list(parameter_names),
            "file_format": file_format,
        },
        "run": dict(run) if run is not None else None,
        "origin": dict(origin),
    }
