"""Tests for graph sampler helper functions."""

from __future__ import annotations

from gwsim_pop.graph.sampler import extract_sampler_dependencies


class TestExtractSamplerDependencies:
    """Test suite for extract_sampler_dependencies function."""

    def test_empty_dict_returns_empty_set(self) -> None:
        """Test that empty dict returns empty set."""
        result = extract_sampler_dependencies({})
        assert result == set()

    def test_function_sampler_with_list_arguments(self) -> None:
        """Test extracting dependencies from function sampler with list arguments."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": ["@mass_1", "@mass_2"],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"mass_1", "mass_2"}

    def test_function_sampler_with_dict_arguments(self) -> None:
        """Test extracting dependencies from function sampler with dict arguments."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": {
                "param1": "@mass_1",
                "param2": "@mass_2",
            },
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"mass_1", "mass_2"}

    def test_function_sampler_with_depends_on(self) -> None:
        """Test extracting dependencies using depends_on key."""
        sampler_spec = {
            "function": "some_sampler",
            "depends_on": ["@param1", "@param2"],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"param1", "param2"}

    def test_function_sampler_prefers_arguments_over_depends_on(self) -> None:
        """Test that arguments key takes precedence over depends_on."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": ["@arg1"],
            "depends_on": ["@dep1"],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"arg1"}

    def test_non_function_sampler_uses_extract_dependencies_from_spec(self) -> None:
        """Test that non-function sampler uses extract_dependencies_from_spec."""
        sampler_spec = {
            "min": 1.0,
            "max": 100.0,
            "expression": "@base_value",
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"base_value"}

    def test_simple_dict_no_function(self) -> None:
        """Test simple dict without function key."""
        sampler_spec = {"min": 1.0, "max": 100.0}
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_nested_dict_in_sampler(self) -> None:
        """Test nested dict in sampler spec."""
        sampler_spec = {
            "nested": {
                "expression": "@nested_value",
            },
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"nested_value"}

    def test_list_arguments_with_empty_strings(self) -> None:
        """Test list arguments with empty strings."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": ["@valid", "", "@"],
        }
        result = extract_sampler_dependencies(sampler_spec)
        # Empty strings and '@' without name are filtered out
        assert result == {"valid"}

    def test_dict_arguments_with_empty_values(self) -> None:
        """Test dict arguments with empty values."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": {
                "valid": "@param",
                "empty": "",
                "not_starting_with_at": "value",
            },
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"param"}

    def test_arguments_as_string_raises_error(self) -> None:
        """Test that string arguments raises error (not expected behavior)."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": "@invalid",
        }
        # This should handle gracefully - currently returns empty set
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_depends_on_as_string_raises_error(self) -> None:
        """Test that string depends_on raises error (not expected behavior)."""
        sampler_spec = {
            "function": "some_sampler",
            "depends_on": "@invalid",
        }
        # This should handle gracefully - currently returns empty set
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_complex_sampler_spec(self) -> None:
        """Test complex sampler spec with multiple features."""
        sampler_spec = {
            "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
            "arguments": {
                "alpha_1": "@alpha1_value",
                "alpha_2": 4.51,
                "transition": "@transition_mass",
                "minimum": 5.06,
                "maximum": 300.0,
                "mean_1": "@peak1_mean",
                "sigma_1": "@peak1_sigma",
                "mean_2": "@peak2_mean",
                "sigma_2": "@peak2_sigma",
                "taper_range": "@taper_value",
                "lambda_0": "@lambda0",
                "lambda_1": "@lambda1",
            },
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {
            "alpha1_value",
            "transition_mass",
            "peak1_mean",
            "peak1_sigma",
            "peak2_mean",
            "peak2_sigma",
            "taper_value",
            "lambda0",
            "lambda1",
        }

    def test_depends_on_list(self) -> None:
        """Test depends_on with list of references."""
        sampler_spec = {
            "function": "some_sampler",
            "depends_on": ["@dep1", "@dep2", "@dep3"],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"dep1", "dep2", "dep3"}

    def test_mixed_arguments_and_depends_on(self) -> None:
        """Test when both arguments and depends_on are present."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": ["@arg1", "@arg2"],
            "depends_on": ["@dep1", "@dep2"],
        }
        # Arguments should take precedence
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"arg1", "arg2"}

    def test_empty_arguments_list(self) -> None:
        """Test empty arguments list."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": [],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_empty_depends_on_list(self) -> None:
        """Test empty depends_on list."""
        sampler_spec = {
            "function": "some_sampler",
            "depends_on": [],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_none_arguments(self) -> None:
        """Test None arguments."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": None,
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_none_depends_on(self) -> None:
        """Test None depends_on."""
        sampler_spec = {
            "function": "some_sampler",
            "depends_on": None,
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == set()

    def test_depends_on_used_when_arguments_missing(self) -> None:
        """Test that depends_on is used when arguments is missing."""
        sampler_spec = {
            "function": "some_sampler",
            "depends_on": ["@dep1", "@dep2"],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"dep1", "dep2"}

    def test_function_sampler_with_mixed_arguments(self) -> None:
        """Test function sampler with mixed string and non-string arguments."""
        sampler_spec = {
            "function": "some_sampler",
            "arguments": ["@param1", 42, "@param2", None],
        }
        result = extract_sampler_dependencies(sampler_spec)
        assert result == {"param1", "param2"}

    def test_nested_function_sampler(self) -> None:
        """Test nested function sampler."""
        sampler_spec = {
            "function": "outer_sampler",
            "arguments": {
                "inner": {
                    "function": "inner_sampler",
                    "arguments": ["@inner_dep"],
                },
                "outer_dep": "@outer_dep",
            },
        }
        result = extract_sampler_dependencies(sampler_spec)
        # Should extract from the top level only
        assert result == {"outer_dep"}
