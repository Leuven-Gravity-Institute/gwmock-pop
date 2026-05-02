"""Tests for ``gwmock_pop.graph.validation``."""

from __future__ import annotations

import pytest

from gwmock_pop.graph.validation import validate_graph_config


def _minimal_two_distance_samplers_plus_string_transform(transform: str) -> dict:
    return {
        "d1": {
            "sampler": {
                "function": "uniform_comoving_volume_distance",
                "arguments": {"d_max": 1000.0},
            },
        },
        "d2": {
            "sampler": {
                "function": "uniform_comoving_volume_distance",
                "arguments": {"d_max": 1000.0},
            },
        },
        "d_sum": {"transform": transform},
    }


def test_string_transform_config_passes_validation() -> None:
    """String ``transform`` entries match the graph config model (see ``extract_transform_dependencies``)."""
    report = validate_graph_config(_minimal_two_distance_samplers_plus_string_transform("@d1 + @d2"))
    assert report.is_valid
    d_sum = next(s for s in report.summaries if s.name == "d_sum")
    assert d_sum.function == "@d1 + @d2"
    assert d_sum.parameters == ()


def test_whitespace_only_string_transform_is_invalid() -> None:
    """Empty or whitespace-only expressions are rejected as malformed."""
    report = validate_graph_config(_minimal_two_distance_samplers_plus_string_transform("   \n\t  "))
    assert not report.is_valid
    assert any("non-empty string" in issue.message for issue in report.issues)


def test_string_transform_with_undefined_reference_is_invalid() -> None:
    """Dependency extraction still applies; undefined ``@`` names are flagged."""
    report = validate_graph_config(_minimal_two_distance_samplers_plus_string_transform("@d1 + @missing"))
    assert not report.is_valid
    assert any("missing" in issue.message for issue in report.issues)


@pytest.mark.parametrize(
    ("bad_block", "fragment"),
    [
        (1.0, "mapping"),
        ([], "mapping"),
    ],
)
def test_transform_block_must_be_mapping_or_string(bad_block: object, fragment: str) -> None:
    """Non-string, non-mapping ``transform`` values are rejected."""
    report = validate_graph_config(
        {
            "d1": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {"d_max": 1000.0},
                },
            },
            "bad": {"transform": bad_block},
        }
    )
    assert not report.is_valid
    assert any(fragment in issue.message for issue in report.issues)


def test_list_transform_arguments_pass_validation() -> None:
    """List ``arguments`` match dependency extraction in ``transform.py`` (dict-style vs list-style)."""
    report = validate_graph_config(
        {
            "d1": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {"d_max": 1000.0},
                },
            },
            "d2": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {"d_max": 1000.0},
                },
            },
            "product": {
                "transform": {
                    "function": "multiply",
                    "arguments": ["@d1", "@d2"],
                },
            },
        }
    )
    assert report.is_valid
    product = next(s for s in report.summaries if s.name == "product")
    assert product.parameters == ("@d1", "@d2")


def test_tuple_transform_arguments_pass_validation() -> None:
    """Tuple ``arguments`` (e.g. from TOML) are treated like list-style positional configs."""
    report = validate_graph_config(
        {
            "d1": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {"d_max": 1000.0},
                },
            },
            "d2": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {"d_max": 1000.0},
                },
            },
            "product": {
                "transform": {
                    "function": "multiply",
                    "arguments": ("@d1", "@d2"),
                },
            },
        }
    )
    assert report.is_valid


def test_non_mapping_non_sequence_arguments_rejected() -> None:
    """Scalar ``arguments`` values are rejected with an updated error message."""
    report = validate_graph_config(
        {
            "d1": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {"d_max": 1000.0},
                },
            },
            "bad": {
                "transform": {
                    "function": "multiply",
                    "arguments": 3.14,
                },
            },
        }
    )
    assert not report.is_valid
    assert any("mapping, sequence, or null" in issue.message for issue in report.issues)
