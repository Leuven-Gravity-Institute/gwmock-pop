"""Functions to build a dependency graph."""

from __future__ import annotations

from typing import Any

from networkx import DiGraph

from gwmock_pop.graph.generic import extract_dependencies_from_spec, extract_references
from gwmock_pop.graph.sampler import extract_sampler_dependencies
from gwmock_pop.graph.transform import extract_transform_dependencies


def build_dependency_graph(parameters_config: dict[str, Any]) -> DiGraph:
    """Build a dependency graph.

    Build a directed graph here:

    - nodes = parameter names
    - edge A → B means: "to compute/sample B, you need A first"

    Args:
        parameters_config: A dictionary of parameters.

    Returns:
        A DiGraph for topological sort.
    """
    graph = DiGraph()

    for parameter_name, spec in parameters_config.items():
        # Add all parameters to the graph
        graph.add_node(node_for_adding=parameter_name)

        # Case 1: Sampler block
        if "sampler" in spec:
            dependencies = extract_sampler_dependencies(sampler_spec=spec["sampler"])
            add_dependencies_to_graph(graph=graph, dependencies=dependencies, parameter_name=parameter_name)

        # Case 1: simple transform expression like "@mass_1 * @mass_ratio"
        if "transform" in spec:
            dependencies = extract_transform_dependencies(transform=spec["transform"])
            add_dependencies_to_graph(graph=graph, dependencies=dependencies, parameter_name=parameter_name)

        # Case 2: condition Expression
        if "condition" in spec:
            dependencies = extract_references(expr=spec["condition"])
            add_dependencies_to_graph(graph=graph, dependencies=dependencies, parameter_name=parameter_name)

        # Case 3: condition_else block (recursive)
        if "condition_else" in spec:
            dependencies = extract_dependencies_from_spec(spec=spec["condition_else"])
            add_dependencies_to_graph(graph=graph, dependencies=dependencies, parameter_name=parameter_name)

        # 4. branches style
        if "branches" in spec:
            for branch in spec["branches"]:
                # condition in branch
                if "condition" in branch:
                    dependencies = extract_references(expr=branch["condition"])
                    add_dependencies_to_graph(graph=graph, dependencies=dependencies, parameter_name=parameter_name)
                # recursive into branch content
                branch_dependencies = extract_dependencies_from_spec(spec=branch)
                add_dependencies_to_graph(graph=graph, dependencies=branch_dependencies, parameter_name=parameter_name)

    return graph


def add_dependencies_to_graph(graph: DiGraph, dependencies: set[str], parameter_name: str) -> None:
    """Add dependencies of a parameter to the graph.

    Args:
        graph: Dependency graph.
        dependencies: A set of dependencies.
        parameter_name: Parameter name.
    """
    for dependency in dependencies:
        graph.add_edge(u_of_edge=dependency, v_of_edge=parameter_name)
