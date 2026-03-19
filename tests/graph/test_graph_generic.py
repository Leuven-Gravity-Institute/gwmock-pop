"""Tests for graph generic helper functions."""

from __future__ import annotations

from gwsim_pop.graph.generic import extract_dependencies_from_spec, extract_references


class TestExtractReferences:
    """Test suite for extract_references function."""

    def test_empty_string_returns_empty_set(self) -> None:
        """Test that empty string returns empty set."""
        result = extract_references("")
        assert result == set()

    def test_single_reference(self) -> None:
        """Test extracting a single @reference."""
        result = extract_references("@mass_1")
        assert result == {"mass_1"}

    def test_multiple_references(self) -> None:
        """Test extracting multiple @references."""
        result = extract_references("@m1 * @q + @z")
        assert result == {"m1", "q", "z"}

    def test_references_with_underscore(self) -> None:
        """Test that references with underscores are correctly extracted."""
        result = extract_references("@param_name_1")
        assert result == {"param_name_1"}

    def test_references_with_numbers(self) -> None:
        """Test that references with numbers are correctly extracted."""
        result = extract_references("@mass1 + @mass2")
        assert result == {"mass1", "mass2"}

    def test_expression_with_math_operations(self) -> None:
        """Test extracting references from mathematical expression."""
        result = extract_references("@mass_1 * @mass_ratio + 5")
        assert result == {"mass_1", "mass_ratio"}

    def test_expression_with_parentheses(self) -> None:
        """Test extracting references from parenthesized expression."""
        result = extract_references("(@a + @b) * @c")
        assert result == {"a", "b", "c"}

    def test_no_references_returns_empty_set(self) -> None:
        """Test that string without references returns empty set."""
        result = extract_references("123 + 456")
        assert result == set()

    def test_references_with_special_chars_not_in_name(self) -> None:
        """Test that special chars are not included in reference names."""
        result = extract_references("@param1 + @param2! @param3;")
        assert result == {"param1", "param2", "param3"}

    def test_duplicate_references_are_removed(self) -> None:
        """Test that duplicate references are deduplicated."""
        result = extract_references("@a + @a + @a")
        assert result == {"a"}
        assert len(result) == 1

    def test_reference_at_start_of_string(self) -> None:
        """Test reference at start of string."""
        result = extract_references("@start middle end")
        assert result == {"start"}

    def test_reference_at_end_of_string(self) -> None:
        """Test reference at end of string."""
        result = extract_references("start middle @end")
        assert result == {"end"}

    def test_consecutive_references(self) -> None:
        """Test consecutive references without spaces."""
        result = extract_references("@a@b")
        assert result == {"a", "b"}


class TestExtractDependenciesFromSpec:
    """Test suite for extract_dependencies_from_spec function."""

    def test_empty_dict_returns_empty_set(self) -> None:
        """Test that empty dict returns empty set."""
        result = extract_dependencies_from_spec({})
        assert result == set()

    def test_simple_dict_with_reference(self) -> None:
        """Test extracting from simple dict with reference."""
        spec = {"key": "@value"}
        result = extract_dependencies_from_spec(spec)
        assert result == {"value"}

    def test_nested_dict_with_references(self) -> None:
        """Test extracting from nested dict."""
        spec = {"outer": {"inner": "@dep"}}
        result = extract_dependencies_from_spec(spec)
        assert result == {"dep"}

    def test_list_with_references(self) -> None:
        """Test extracting from list."""
        spec = ["@a", "@b", "@c"]
        result = extract_dependencies_from_spec(spec)
        assert result == {"a", "b", "c"}

    def test_mixed_dict_and_list(self) -> None:
        """Test extracting from mixed dict and list."""
        spec = {"list": ["@x", "@y"], "value": "@z"}
        result = extract_dependencies_from_spec(spec)
        assert result == {"x", "y", "z"}

    def test_complex_nested_structure(self) -> None:
        """Test extracting from complex nested structure."""
        spec = {
            "level1": {
                "level2": {
                    "expression": "@a * @b",
                    "list": ["@c", "@d"],
                },
            },
            "top": "@e",
        }
        result = extract_dependencies_from_spec(spec)
        assert result == {"a", "b", "c", "d", "e"}

    def test_non_string_values_ignored(self) -> None:
        """Test that non-string values are ignored."""
        spec = {"num": 42, "bool": True, "none": None, "list": [1, 2, 3]}
        result = extract_dependencies_from_spec(spec)
        assert result == set()

    def test_deeply_nested_structure(self) -> None:
        """Test extracting from deeply nested structure."""
        spec = {
            "a": {
                "b": {
                    "c": {
                        "d": {
                            "expression": "@deep",
                        },
                    },
                },
            },
        }
        result = extract_dependencies_from_spec(spec)
        assert result == {"deep"}

    def test_empty_string_in_list(self) -> None:
        """Test that empty strings don't cause issues."""
        spec = ["", "@valid", ""]
        result = extract_dependencies_from_spec(spec)
        assert result == {"valid"}

    def test_multiple_levels_with_duplicates(self) -> None:
        """Test that duplicates across levels are removed."""
        spec = {
            "level1": "@shared",
            "level2": {"level3": "@shared"},
        }
        result = extract_dependencies_from_spec(spec)
        assert result == {"shared"}
        assert len(result) == 1

    def test_nested_list_of_dicts(self) -> None:
        """Test extracting from nested list of dicts."""
        spec = [
            {"key": "@a"},
            {"key": "@b"},
        ]
        result = extract_dependencies_from_spec(spec)
        assert result == {"a", "b"}

    def test_dict_in_list(self) -> None:
        """Test extracting from dict inside list."""
        spec = [{"key": "@value"}]
        result = extract_dependencies_from_spec(spec)
        assert result == {"value"}

    def test_list_in_dict(self) -> None:
        """Test extracting from list inside dict."""
        spec = {"list": ["@a", "@b"]}
        result = extract_dependencies_from_spec(spec)
        assert result == {"a", "b"}

    def test_complex_real_world_example(self) -> None:
        """Test with a complex real-world-like spec."""
        spec = {
            "sampler": {
                "function": "some_sampler",
                "arguments": {
                    "param1": "@mass_1",
                    "param2": "@mass_2",
                    "nested": {
                        "expression": "@redshift * @lum_dist",
                    },
                },
            },
            "condition": "@mass_1 > 5",
            "branches": [
                {
                    "condition": "@q > 0.5",
                    "value": "@high_q_value",
                },
                {
                    "condition": "@q <= 0.5",
                    "value": "@low_q_value",
                },
            ],
        }
        result = extract_dependencies_from_spec(spec)
        assert result == {
            "mass_1",
            "mass_2",
            "redshift",
            "lum_dist",
            "q",
            "high_q_value",
            "low_q_value",
        }

    def test_none_values_ignored(self) -> None:
        """Test that None values don't cause issues."""
        spec = {"key1": None, "key2": "@valid", "key3": None}
        result = extract_dependencies_from_spec(spec)
        assert result == {"valid"}

    def test_boolean_values_ignored(self) -> None:
        """Test that boolean values don't cause issues."""
        spec = {"key1": True, "key2": False, "key3": "@valid"}
        result = extract_dependencies_from_spec(spec)
        assert result == {"valid"}

    def test_numeric_values_ignored(self) -> None:
        """Test that numeric values don't cause issues."""
        spec = {"int": 42, "float": 3.14, "scientific": 1e-10, "key": "@valid"}
        result = extract_dependencies_from_spec(spec)
        assert result == {"valid"}
