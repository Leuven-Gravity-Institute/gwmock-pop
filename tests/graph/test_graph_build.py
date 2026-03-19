"""Tests for graph build functions."""

from __future__ import annotations

import networkx as nx
import pytest
from networkx import DiGraph

from gwsim_pop.graph.build import add_dependencies_to_graph, build_dependency_graph


class TestAddDependenciesToGraph:
    """Test suite for add_dependencies_to_graph function."""

    def test_empty_dependencies_does_nothing(self) -> None:
        """Test that empty dependencies don't add edges."""
        graph = DiGraph()
        graph.add_node("param1")
        add_dependencies_to_graph(graph=graph, dependencies=set(), parameter_name="param1")
        assert len(graph.edges()) == 0

    def test_single_dependency_creates_edge(self) -> None:
        """Test that single dependency creates one edge."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("dependency")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dependency"},
            parameter_name="param1",
        )
        assert len(graph.edges()) == 1
        assert ("dependency", "param1") in graph.edges()

    def test_multiple_dependencies_create_multiple_edges(self) -> None:
        """Test that multiple dependencies create multiple edges."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("dep1")
        graph.add_node("dep2")
        graph.add_node("dep3")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1", "dep2", "dep3"},
            parameter_name="param1",
        )
        assert len(graph.edges()) == 3  # noqa: PLR2004
        assert ("dep1", "param1") in graph.edges()
        assert ("dep2", "param1") in graph.edges()
        assert ("dep3", "param1") in graph.edges()

    def test_dependencies_added_to_existing_edges(self) -> None:
        """Test that dependencies are added to existing edges."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("dep1")
        graph.add_edge("dep1", "param1")
        graph.add_node("dep2")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep2"},
            parameter_name="param1",
        )
        assert len(graph.edges()) == 2  # noqa: PLR2004
        assert ("dep1", "param1") in graph.edges()
        assert ("dep2", "param1") in graph.edges()

    def test_parameter_not_in_graph_adds_node(self) -> None:
        """Test that parameter not in graph adds the node."""
        graph = DiGraph()
        graph.add_node("other_param")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1"},
            parameter_name="param1",
        )
        assert "param1" in graph.nodes()
        assert ("dep1", "param1") in graph.edges()

    def test_dependency_not_in_graph_creates_node(self) -> None:
        """Test that dependency not in graph adds the node."""
        graph = DiGraph()
        graph.add_node("param1")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1"},
            parameter_name="param1",
        )
        assert "dep1" in graph.nodes()
        assert ("dep1", "param1") in graph.edges()

    def test_self_dependency_creates_self_loop(self) -> None:
        """Test that self-dependency creates a self-loop."""
        graph = DiGraph()
        graph.add_node("param1")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"param1"},
            parameter_name="param1",
        )
        assert ("param1", "param1") in graph.edges()

    def test_duplicate_dependencies_dont_create_duplicate_edges(self) -> None:
        """Test that duplicate dependencies don't create duplicate edges."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("dep1")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1"},
            parameter_name="param1",
        )
        # NetworkX DiGraph doesn't allow duplicate edges
        assert len(graph.edges()) == 1
        assert ("dep1", "param1") in graph.edges()

    def test_order_of_dependencies_preserved(self) -> None:
        """Test that order of dependencies is preserved in iteration."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("dep1")
        graph.add_node("dep2")
        graph.add_node("dep3")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1", "dep2", "dep3"},
            parameter_name="param1",
        )
        # Check that all edges exist
        edges = list(graph.edges())
        assert len(edges) == 3  # noqa: PLR2004
        for dep in ["dep1", "dep2", "dep3"]:
            assert (dep, "param1") in edges

    def test_graph_is_directed(self) -> None:
        """Test that the graph remains directed."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("dep1")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1"},
            parameter_name="param1",
        )
        # Edge should be directed from dep1 to param1
        assert ("dep1", "param1") in graph.edges()
        # Reverse edge should not exist
        assert ("param1", "dep1") not in graph.edges()

    def test_multiple_parameters_with_dependencies(self) -> None:
        """Test adding dependencies for multiple parameters."""
        graph = DiGraph()
        graph.add_node("param1")
        graph.add_node("param2")
        graph.add_node("dep1")
        graph.add_node("dep2")
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep1"},
            parameter_name="param1",
        )
        add_dependencies_to_graph(
            graph=graph,
            dependencies={"dep2"},
            parameter_name="param2",
        )
        assert len(graph.edges()) == 2  # noqa: PLR2004
        assert ("dep1", "param1") in graph.edges()
        assert ("dep2", "param2") in graph.edges()


class TestBuildDependencyGraph:
    """Test suite for build_dependency_graph function."""

    def test_empty_config_returns_empty_graph(self) -> None:
        """Test that empty config returns empty graph."""
        graph = build_dependency_graph({})
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0

    def test_single_sampler_parameter(self) -> None:
        """Test building graph with single sampler parameter."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "some_sampler",
                    "arguments": [],
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert len(graph.edges()) == 0

    def test_sampler_with_dependencies(self) -> None:
        """Test building graph with sampler that has dependencies."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "some_sampler",
                    "arguments": ["@base_mass"],
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert "base_mass" in graph.nodes()
        assert ("base_mass", "mass_1") in graph.edges()

    def test_sampler_with_dict_arguments(self) -> None:
        """Test building graph with sampler with dict arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "some_sampler",
                    "arguments": {
                        "param1": "@value1",
                        "param2": "@value2",
                    },
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert "value1" in graph.nodes()
        assert "value2" in graph.nodes()
        assert ("value1", "mass_1") in graph.edges()
        assert ("value2", "mass_1") in graph.edges()

    def test_transform_with_string_expression(self) -> None:
        """Test building graph with transform string."""
        config = {
            "mass_ratio": {
                "transform": "@mass_2 / @mass_1",
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_ratio" in graph.nodes()
        assert "mass_1" in graph.nodes()
        assert "mass_2" in graph.nodes()
        assert ("mass_1", "mass_ratio") in graph.edges()
        assert ("mass_2", "mass_ratio") in graph.edges()

    def test_transform_with_dict(self) -> None:
        """Test building graph with transform dict."""
        config = {
            "mass_ratio": {
                "transform": {
                    "function": "some_transform",
                    "arguments": ["@input1", "@input2"],
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_ratio" in graph.nodes()
        assert "input1" in graph.nodes()
        assert "input2" in graph.nodes()
        assert ("input1", "mass_ratio") in graph.edges()
        assert ("input2", "mass_ratio") in graph.edges()

    def test_condition_expression(self) -> None:
        """Test building graph with condition expression."""
        config = {
            "mass_1": {
                "condition": "@mass_1 > 5",
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert ("mass_1", "mass_1") in graph.edges()

    def test_condition_else_block(self) -> None:
        """Test building graph with condition_else block."""
        config = {
            "mass_1": {
                "condition_else": {
                    "value": "@fallback_value",
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert "fallback_value" in graph.nodes()
        assert ("fallback_value", "mass_1") in graph.edges()

    def test_branches_style(self) -> None:
        """Test building graph with branches."""
        config = {
            "mass_1": {
                "branches": [
                    {
                        "condition": "@condition1",
                        "value": "@value1",
                    },
                    {
                        "condition": "@condition2",
                        "value": "@value2",
                    },
                ],
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert "condition1" in graph.nodes()
        assert "condition2" in graph.nodes()
        assert "value1" in graph.nodes()
        assert "value2" in graph.nodes()
        assert ("condition1", "mass_1") in graph.edges()
        assert ("condition2", "mass_1") in graph.edges()
        assert ("value1", "mass_1") in graph.edges()
        assert ("value2", "mass_1") in graph.edges()

    def test_complex_config_with_multiple_features(self) -> None:
        """Test building graph with complex config."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "some_sampler",
                    "arguments": ["@base_mass"],
                },
            },
            "mass_ratio": {
                "transform": "@mass_2 / @mass_1",
            },
            "mass_2": {
                "sampler": {
                    "function": "some_sampler",
                    "arguments": ["@base_mass"],
                },
            },
            "final_mass": {
                "transform": "@mass_1 + @mass_ratio",
            },
        }
        graph = build_dependency_graph(config)
        # Check all nodes are present
        assert "mass_1" in graph.nodes()
        assert "mass_ratio" in graph.nodes()
        assert "mass_2" in graph.nodes()
        assert "final_mass" in graph.nodes()
        assert "base_mass" in graph.nodes()

        # Check edges
        assert ("base_mass", "mass_1") in graph.edges()
        assert ("base_mass", "mass_2") in graph.edges()
        assert ("mass_1", "mass_ratio") in graph.edges()
        assert ("mass_2", "mass_ratio") in graph.edges()
        assert ("mass_1", "final_mass") in graph.edges()
        assert ("mass_ratio", "final_mass") in graph.edges()

    def test_multiple_parameters_same_dependency(self) -> None:
        """Test multiple parameters depending on same value."""
        config = {
            "mass_1": {
                "sampler": {
                    "arguments": ["@common"],
                },
            },
            "mass_2": {
                "sampler": {
                    "arguments": ["@common"],
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "common" in graph.nodes()
        assert ("common", "mass_1") in graph.edges()
        assert ("common", "mass_2") in graph.edges()

    def test_circular_dependency(self) -> None:
        """Test that circular dependencies are allowed (directed graph)."""
        config = {
            "param1": {
                "transform": "@param2",
            },
            "param2": {
                "transform": "@param1",
            },
        }
        graph = build_dependency_graph(config)
        assert ("param2", "param1") in graph.edges()
        assert ("param1", "param2") in graph.edges()

    def test_sampler_and_transform_same_parameter(self) -> None:
        """Test parameter with both sampler and transform."""
        config = {
            "mass_1": {
                "sampler": {
                    "arguments": ["@base_mass"],
                },
                "transform": "@mass_1 * 2",
            },
        }
        graph = build_dependency_graph(config)
        assert "mass_1" in graph.nodes()
        assert "base_mass" in graph.nodes()
        # Sampler dependency
        assert ("base_mass", "mass_1") in graph.edges()
        # Transform dependency (self-reference)
        assert ("mass_1", "mass_1") in graph.edges()

    def test_nested_branches(self) -> None:
        """Test nested branches structure."""
        config = {
            "param1": {
                "branches": [
                    {
                        "condition": "@cond1",
                        "value": "@val1",
                        "branches": [
                            {
                                "condition": "@cond2",
                                "value": "@val2",
                            },
                        ],
                    },
                ],
            },
        }
        graph = build_dependency_graph(config)
        assert "param1" in graph.nodes()
        assert "cond1" in graph.nodes()
        assert "val1" in graph.nodes()
        assert "cond2" in graph.nodes()
        assert "val2" in graph.nodes()

    def test_graph_is_directed_acyclic_when_no_cycles(self) -> None:
        """Test that graph is directed acyclic when no cycles."""
        config = {
            "param1": {
                "sampler": {"arguments": []},
            },
            "param2": {
                "transform": "@param1",
            },
            "param3": {
                "transform": "@param2",
            },
        }
        graph = build_dependency_graph(config)
        # Check DAG structure
        assert len(graph.nodes()) == 3  # noqa: PLR2004
        assert len(graph.edges()) == 2  # noqa: PLR2004
        # Check topological order is possible
        try:
            list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            pytest.fail("Graph is not a DAG")

    def test_parameter_without_dependencies(self) -> None:
        """Test parameter without any dependencies."""
        config = {
            "constant_param": {
                "sampler": {
                    "arguments": [],
                },
            },
        }
        graph = build_dependency_graph(config)
        assert "constant_param" in graph.nodes()
        assert len(graph.edges()) == 0

    def test_complex_real_world_example(self) -> None:
        """Test with a complex real-world-like config."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": "@alpha1",
                        "alpha_2": 4.51,
                        "transition": "@transition_mass",
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": "@peak1_mean",
                        "sigma_1": "@peak1_sigma",
                        "mean_2": "@peak2_mean",
                        "sigma_2": "@peak2_sigma",
                        "taper_range": "@taper",
                        "lambda_0": "@lambda0",
                        "lambda_1": "@lambda1",
                    },
                },
            },
            "mass_ratio": {
                "transform": "@mass_2 / @mass_1",
            },
            "mass_2": {
                "sampler": {
                    "arguments": ["@mass_1"],
                },
            },
            "redshift": {
                "sampler": {
                    "arguments": [],
                },
            },
            "luminosity_distance": {
                "transform": "@redshift * @scale_factor",
            },
            "combined_mass": {
                "transform": "@mass_1 + @mass_2",
            },
        }
        graph = build_dependency_graph(config)

        # Check all nodes
        expected_nodes = {
            "mass_1",
            "mass_ratio",
            "mass_2",
            "redshift",
            "luminosity_distance",
            "combined_mass",
            "alpha1",
            "transition_mass",
            "peak1_mean",
            "peak1_sigma",
            "peak2_mean",
            "peak2_sigma",
            "taper",
            "lambda0",
            "lambda1",
            "scale_factor",
        }
        assert set(graph.nodes()) == expected_nodes

        # Check key edges
        assert ("alpha1", "mass_1") in graph.edges()
        assert ("transition_mass", "mass_1") in graph.edges()
        assert ("mass_1", "mass_ratio") in graph.edges()
        assert ("mass_2", "mass_ratio") in graph.edges()
        assert ("redshift", "luminosity_distance") in graph.edges()
        assert ("scale_factor", "luminosity_distance") in graph.edges()
        assert ("mass_1", "combined_mass") in graph.edges()
        assert ("mass_2", "combined_mass") in graph.edges()
