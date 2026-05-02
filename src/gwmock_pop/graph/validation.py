"""Validation helpers for graph-based simulator configs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from gwmock_pop.graph.build import build_dependency_graph
from gwmock_pop.graph.sampler import extract_sampler_dependencies
from gwmock_pop.graph.transform import extract_transform_dependencies
from gwmock_pop.utils.import_utils import import_from_string
from gwmock_pop.utils.yaml import read_data_file

_TRANSFORM_STRING_SUMMARY_MAX_CHARS = 120


@dataclass(frozen=True)
class NodeSummary:
    """Human-readable summary of a validated config node."""

    name: str
    function: str
    parameters: tuple[str, ...]
    output_keys: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation issue."""

    message: str
    node_name: str | None = None

    def render(self) -> str:
        """Render a human-readable validation message."""
        if self.node_name is None:
            return self.message
        return f"Node '{self.node_name}': {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of validating a graph config."""

    summaries: tuple[NodeSummary, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the config passed validation."""
        return not self.issues


def load_parameters_config(config_path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Load and normalize a graph config file."""
    config_path = Path(config_path)
    try:
        config = read_data_file(config_path, encoding=encoding)
    except ValueError as error:
        message = str(error)
        if "Suffix of filename=" in message:
            raise ValueError(
                f"Suffix of config_path={config_path} is not supported. Only '.yaml', '.yml', and '.toml' are supported."
            ) from error
        if "mapping at the top level" in message:
            raise ValueError("Config file must contain a mapping at the root.") from error
        raise

    if "parameters" in config:
        config = config["parameters"]

    if not isinstance(config, dict):
        raise ValueError("Graph config must define a mapping of parameters.")

    return config


def validate_graph_config_file(config_path: str | Path, encoding: str = "utf-8") -> ValidationReport:
    """Validate a graph config file without executing any sampling."""
    return validate_graph_config(load_parameters_config(config_path=config_path, encoding=encoding))


def validate_graph_config(config: dict[str, Any]) -> ValidationReport:
    """Validate a graph config mapping without executing any nodes."""
    issues: list[ValidationIssue] = []
    summaries: list[NodeSummary] = []
    parameter_names = set(config)

    for node_name, spec in config.items():
        node_issues, summary = _validate_node(node_name=node_name, spec=spec, parameter_names=parameter_names)
        issues.extend(node_issues)
        if summary is not None:
            summaries.append(summary)

    graph_issues, ordered_names = _validate_graph_structure(config=config)
    issues.extend(graph_issues)

    if ordered_names is not None:
        summary_by_name = {summary.name: summary for summary in summaries}
        summaries = [summary_by_name[name] for name in ordered_names if name in summary_by_name]

    return ValidationReport(summaries=tuple(summaries), issues=tuple(issues))


def render_validation_summary(summaries: tuple[NodeSummary, ...] | list[NodeSummary]) -> str:
    """Render a plain-text table summarizing validated nodes."""
    headers = ("Node", "Transform", "Parameters", "Output key(s)")
    rows = [
        (
            summary.name,
            summary.function,
            ", ".join(summary.parameters) if summary.parameters else "—",
            ", ".join(summary.output_keys) if summary.output_keys else "—",
        )
        for summary in summaries
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(headers[index])
        for index in range(len(headers))
    ]

    def _format_row(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row, start=0))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def _validate_graph_structure(config: dict[str, Any]) -> tuple[list[ValidationIssue], list[str] | None]:
    """Validate cross-node graph structure."""
    issues: list[ValidationIssue] = []

    try:
        graph = build_dependency_graph(config)
        ordered_names = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        graph = build_dependency_graph(config)
        cycle_edges = nx.find_cycle(graph, orientation="original")
        cycle_nodes = [cycle_edges[0][0], *(edge[1] for edge in cycle_edges)]
        cycle_text = " -> ".join(cycle_nodes)
        issues.append(ValidationIssue(message=f"Dependency cycle detected: {cycle_text}."))
        return issues, None

    undefined = [name for name in ordered_names if name not in config]
    if undefined:
        issues.append(ValidationIssue(message=f"Undefined parameter dependencies: {undefined}"))

    return issues, ordered_names


def _validate_node(
    node_name: str,
    spec: Any,
    parameter_names: set[str],
) -> tuple[list[ValidationIssue], NodeSummary | None]:
    """Validate a single config node."""
    issues: list[ValidationIssue] = []
    resolved = _resolve_node_block(node_name=node_name, spec=spec)
    if isinstance(resolved, ValidationIssue):
        issues.append(resolved)
        return issues, None

    block_name, block, function_name = resolved
    arguments_or_issue = _resolve_node_arguments(node_name=node_name, block_name=block_name, block=block)
    if isinstance(arguments_or_issue, ValidationIssue):
        issues.append(arguments_or_issue)
        return issues, None
    arguments = arguments_or_issue

    dependencies = (
        extract_sampler_dependencies(block) if block_name == "sampler" else extract_transform_dependencies(block)
    )
    undefined_dependencies = sorted(dependency for dependency in dependencies if dependency not in parameter_names)
    if undefined_dependencies:
        issues.append(
            ValidationIssue(
                node_name=node_name,
                message=f"References undefined parameter(s): {', '.join(undefined_dependencies)}.",
            )
        )

    if not (block_name == "transform" and isinstance(block, str)):
        callable_issues = _validate_callable(
            node_name=node_name,
            block_name=block_name,
            function_name=function_name,
            arguments=arguments,
        )
        issues.extend(callable_issues)

    summary = NodeSummary(
        name=node_name,
        function=function_name,
        parameters=tuple(arguments),
        output_keys=(node_name,),
    )
    return issues, summary


def _resolve_node_block(  # noqa: PLR0912
    node_name: str,
    spec: Any,
) -> tuple[str, dict[str, Any] | str, str] | ValidationIssue:
    """Resolve the active node block and function name."""
    issue: ValidationIssue | None = None
    resolved: tuple[str, dict[str, Any] | str, str] | None = None

    if not isinstance(spec, dict):
        issue = ValidationIssue(node_name=node_name, message="Node spec must be a mapping.")
    else:
        has_sampler = "sampler" in spec
        has_transform = "transform" in spec
        if has_sampler and has_transform:
            issue = ValidationIssue(
                node_name=node_name, message="Node cannot define both 'sampler' and 'transform' blocks."
            )
        elif not has_sampler and not has_transform:
            issue = ValidationIssue(
                node_name=node_name, message="Node must define either a 'sampler' or 'transform' block."
            )
        else:
            block_name = "sampler" if has_sampler else "transform"
            block = spec[block_name]
            if isinstance(block, dict):
                function_name = block.get("function")
                if not function_name:
                    issue = ValidationIssue(
                        node_name=node_name,
                        message=f"'{block_name}' block is missing required field 'function'.",
                    )
                elif not isinstance(function_name, str):
                    issue = ValidationIssue(node_name=node_name, message=f"'{block_name}.function' must be a string.")
                else:
                    resolved = (block_name, block, function_name)
            elif block_name == "transform" and isinstance(block, str):
                if not block.strip():
                    issue = ValidationIssue(
                        node_name=node_name,
                        message="'transform' expression must be a non-empty string.",
                    )
                else:
                    max_len = _TRANSFORM_STRING_SUMMARY_MAX_CHARS
                    ellipsis = "..."
                    display = block if len(block) <= max_len else f"{block[: max_len - len(ellipsis)]}{ellipsis}"
                    resolved = (block_name, block, display)
            else:
                issue = ValidationIssue(node_name=node_name, message=f"'{block_name}' block must be a mapping.")

    if issue is not None:
        return issue
    if resolved is None:
        raise RuntimeError("Internal error while resolving node block.")
    return resolved


def _resolve_node_arguments(
    node_name: str,
    block_name: str,
    block: dict[str, Any] | str,
) -> dict[str, Any] | ValidationIssue:
    """Resolve configured node arguments into a mapping."""
    if isinstance(block, str):
        if block_name != "transform":
            return ValidationIssue(node_name=node_name, message=f"'{block_name}' block must be a mapping.")
        return {}

    arguments = block.get("arguments")
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    return ValidationIssue(node_name=node_name, message=f"'{block_name}.arguments' must be a mapping or null.")


def _validate_callable(
    node_name: str,
    block_name: str,
    function_name: str,
    arguments: dict[str, Any],
) -> list[ValidationIssue]:
    """Validate function import and configured arguments for a node."""
    issues: list[ValidationIssue] = []
    default_module = "gwmock_pop.samplers" if block_name == "sampler" else "gwmock_pop.transforms"

    try:
        callable_object = import_from_string(function_name, default_module=default_module)
    except ImportError as error:
        issues.append(ValidationIssue(node_name=node_name, message=f"Unknown {block_name} '{function_name}': {error}"))
        return issues

    signature = inspect.signature(callable_object)
    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    auto_arguments = _auto_arguments_for_signature(block_name=block_name, signature=signature)

    unexpected = sorted(argument for argument in arguments if argument not in parameters and not accepts_kwargs)
    if unexpected:
        issues.append(
            ValidationIssue(
                node_name=node_name,
                message=f"Unexpected {block_name} argument(s) for '{function_name}': {', '.join(unexpected)}.",
            )
        )

    missing: list[str] = []
    for parameter_name, parameter in parameters.items():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            issues.append(
                ValidationIssue(
                    node_name=node_name,
                    message=f"{block_name.capitalize()} '{function_name}' has unsupported positional-only parameter '{parameter_name}'.",
                )
            )
            continue
        if parameter_name in arguments or parameter_name in auto_arguments:
            continue
        missing.append(parameter_name)

    if missing:
        issues.append(
            ValidationIssue(
                node_name=node_name,
                message=f"Missing required {block_name} argument(s) for '{function_name}': {', '.join(missing)}.",
            )
        )

    return issues


def _auto_arguments_for_signature(block_name: str, signature: inspect.Signature) -> set[str]:
    """Return arguments injected automatically by GraphSimulator at runtime."""
    auto_arguments: set[str] = set()
    if block_name == "sampler":
        auto_arguments.update({"key", "n_samples"})
        return auto_arguments

    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if "key" in signature.parameters or accepts_kwargs:
        auto_arguments.add("key")
    return auto_arguments
