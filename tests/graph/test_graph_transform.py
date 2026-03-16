"""Tests for graph transform helper functions."""

from __future__ import annotations

from gwsim_pop.graph.transform import extract_transform_dependencies


class TestExtractTransformDependencies:
    """Test suite for extract_transform_dependencies function."""

    def test_empty_string_returns_empty_set(self) -> None:
        """Test that empty string returns empty set."""
        result = extract_transform_dependencies("")
        assert result == set()

    def test_simple_transform_string(self) -> None:
        """Test extracting dependencies from simple transform string."""
        transform = "@mass_1 * @mass_ratio"
        result = extract_transform_dependencies(transform)
        assert result == {"mass_1", "mass_ratio"}

    def test_transform_with_math_operations(self) -> None:
        """Test extracting from transform with math operations."""
        transform = "(@a + @b) * @c - @d"
        result = extract_transform_dependencies(transform)
        assert result == {"a", "b", "c", "d"}

    def test_transform_dict_with_function(self) -> None:
        """Test extracting from transform dict with function."""
        transform = {
            "function": "some_transform",
            "arguments": ["@input", "@scale"],
        }
        result = extract_transform_dependencies(transform)
        assert result == {"input", "scale"}

    def test_transform_dict_with_dict_arguments(self) -> None:
        """Test extracting from transform dict with dict arguments."""
        transform = {
            "function": "some_transform",
            "arguments": {
                "data": "@input_data",
                "factor": "@scale_factor",
            },
        }
        result = extract_transform_dependencies(transform)
        assert result == {"input_data", "scale_factor"}

    def test_transform_dict_with_list_arguments(self) -> None:
        """Test extracting from transform dict with list arguments."""
        transform = {
            "function": "some_transform",
            "arguments": ["@param1", "@param2", "@param3"],
        }
        result = extract_transform_dependencies(transform)
        assert result == {"param1", "param2", "param3"}

    def test_transform_dict_without_function(self) -> None:
        """Test transform dict without function key."""
        transform = {
            "key1": "value1",
            "key2": "@value2",
        }
        result = extract_transform_dependencies(transform)
        assert result == set()

    def test_transform_string_with_no_references(self) -> None:
        """Test transform string without references."""
        transform = "123 + 456"
        result = extract_transform_dependencies(transform)
        assert result == set()

    def test_transform_dict_with_empty_arguments_list(self) -> None:
        """Test transform dict with empty arguments list."""
        transform = {
            "function": "some_transform",
            "arguments": [],
        }
        result = extract_transform_dependencies(transform)
        assert result == set()

    def test_transform_dict_with_empty_arguments_dict(self) -> None:
        """Test transform dict with empty arguments dict."""
        transform = {
            "function": "some_transform",
            "arguments": {},
        }
        result = extract_transform_dependencies(transform)
        assert result == set()

    def test_transform_dict_with_non_string_arguments(self) -> None:
        """Test transform dict with non-string arguments."""
        transform = {
            "function": "some_transform",
            "arguments": [123, 456, None],
        }
        result = extract_transform_dependencies(transform)
        assert result == set()

    def test_transform_dict_with_mixed_arguments(self) -> None:
        """Test transform dict with mixed string and non-string arguments."""
        transform = {
            "function": "some_transform",
            "arguments": ["@valid", 42, "@another", None],
        }
        result = extract_transform_dependencies(transform)
        assert result == {"valid", "another"}

    def test_transform_dict_with_dict_arguments_mixed(self) -> None:
        """Test transform dict with dict arguments containing mixed types."""
        transform = {
            "function": "some_transform",
            "arguments": {
                "valid": "@param",
                "number": 42,
                "another": "@second",
            },
        }
        result = extract_transform_dependencies(transform)
        assert result == {"param", "second"}

    def test_complex_transform_example(self) -> None:
        """Test complex transform example."""
        transform = {
            "function": "gwsim_pop.transforms.offset_values",
            "arguments": {
                "values": "@mass_1",
                "offset": "@offset_value",
            },
        }
        result = extract_transform_dependencies(transform)
        assert result == {"mass_1", "offset_value"}

    def test_transform_string_with_special_chars(self) -> None:
        """Test transform string with special characters."""
        transform = "@param1 + @param2! @param3;"
        result = extract_transform_dependencies(transform)
        assert result == {"param1", "param2", "param3"}

    def test_transform_dict_with_nested_function(self) -> None:
        """Test that nested functions are not processed recursively."""
        transform = {
            "function": "outer_transform",
            "arguments": {
                "inner": {
                    "function": "inner_transform",
                    "arguments": ["@inner_dep"],
                },
                "outer_dep": "@outer_dep",
            },
        }
        result = extract_transform_dependencies(transform)
        # Only top-level arguments are processed
        assert result == {"outer_dep"}

    def test_transform_dict_with_depends_on_style(self) -> None:
        """Test transform dict with depends_on style arguments."""
        transform = {
            "function": "some_transform",
            "arguments": ["@input"],
        }
        result = extract_transform_dependencies(transform)
        assert result == {"input"}

    def test_non_dict_non_string_transform_returns_empty(self) -> None:
        """Test that non-dict, non-string transform returns empty set."""
        # Type annotations say str | dict[str, Any], but function handles other types
        result = extract_transform_dependencies(42)  # type: ignore[arg-type]
        assert result == set()

    def test_transform_dict_with_none_arguments(self) -> None:
        """Test transform dict with None arguments."""
        transform = {
            "function": "some_transform",
            "arguments": None,
        }
        result = extract_transform_dependencies(transform)
        assert result == set()

    def test_transform_string_with_references_at_boundaries(self) -> None:
        """Test transform with references at string boundaries."""
        transform = "@start middle @end"
        result = extract_transform_dependencies(transform)
        assert result == {"start", "end"}

    def test_transform_dict_with_duplicate_references(self) -> None:
        """Test that duplicate references are deduplicated."""
        transform = {
            "function": "some_transform",
            "arguments": ["@dup", "@dup", "@dup"],
        }
        result = extract_transform_dependencies(transform)
        assert result == {"dup"}
        assert len(result) == 1

    def test_transform_string_with_complex_expression(self) -> None:
        """Test transform with complex mathematical expression."""
        transform = "(@mass_1 * @mass_ratio + @redshift) / @lum_dist"
        result = extract_transform_dependencies(transform)
        assert result == {"mass_1", "mass_ratio", "redshift", "lum_dist"}
