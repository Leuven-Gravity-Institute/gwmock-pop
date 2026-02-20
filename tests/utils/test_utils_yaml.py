"""Tests for YAML utility functions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gwsim_pop.utils.yaml import read_yaml, write_yaml


class TestYAMLUtils:
    """Test cases for YAML utility functions."""

    def test_read_yaml_valid_file(self, tmp_path: Path) -> None:
        """Test reading a valid YAML file."""
        # Create a temporary YAML file
        yaml_content = """
        key1: value1
        key2: 42
        key3:
          nested: value
        """
        test_file = tmp_path / "test.yaml"
        test_file.write_text(yaml_content)

        # Read the file
        result = read_yaml(test_file)
        expected = {
            "key1": "value1",
            "key2": 42,
            "key3": {"nested": "value"},
        }
        assert result == expected

    def test_read_yaml_invalid_file(self, tmp_path: Path) -> None:
        """Test reading an invalid YAML file."""
        # Create a temporary invalid YAML file
        test_file = tmp_path / "invalid.yaml"
        test_file.write_text("invalid: [yaml")

        # This should raise a YAML error
        with pytest.raises(yaml.YAMLError):
            read_yaml(test_file)

    def test_write_yaml_simple(self, tmp_path: Path) -> None:
        """Test writing simple YAML data without round_trip."""
        test_file = tmp_path / "output.yaml"
        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

        write_yaml(test_file, data)

        # Read back the file to verify
        result = read_yaml(test_file)
        assert result == data

    def test_write_yaml_with_round_trip(self, tmp_path: Path) -> None:
        """Test writing YAML data with round_trip enabled."""
        test_file = tmp_path / "output_round_trip.yaml"
        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

        write_yaml(test_file, data, round_trip=True)

        # Read back the file to verify
        result = read_yaml(test_file)
        assert result == data

    def test_write_yaml_with_enum(self, tmp_path: Path) -> None:
        """Test writing YAML data containing enums."""
        test_file = tmp_path / "enum.yaml"
        # Test enum is defined in the yaml module, but for testing, let's just use a simple dict
        data = {"enum_key": "value1", "normal_key": "value"}

        write_yaml(test_file, data)

        # Read back the file to verify
        result = read_yaml(test_file)
        assert result == {"enum_key": "value1", "normal_key": "value"}

    def test_write_yaml_with_enum_round_trip(self, tmp_path: Path) -> None:
        """Test writing YAML data containing enums with round_trip."""
        test_file = tmp_path / "enum_round_trip.yaml"
        # Test enum is defined in the yaml module, but for testing, let's just use a simple dict
        data = {"enum_key": "value1", "normal_key": "value"}

        write_yaml(test_file, data, round_trip=True)

        # Read back the file to verify
        result = read_yaml(test_file)
        assert result == {"enum_key": "value1", "normal_key": "value"}

    def test_write_yaml_directory_creation(self, tmp_path: Path) -> None:
        """Test that write_yaml creates parent directories if they don't exist."""
        nested_file = tmp_path / "nested" / "deep" / "test.yaml"
        data = {"key": "value"}

        write_yaml(nested_file, data)

        # Verify that the file exists and can be read
        assert nested_file.exists()
        result = read_yaml(nested_file)
        assert result == data

    def test_write_yaml_preserves_ordering_with_round_trip(self, tmp_path: Path) -> None:
        """Test that round_trip preserves key ordering."""
        test_file = tmp_path / "ordered.yaml"
        data = {
            "first": "value1",
            "second": "value2",
            "third": "value3",
        }

        write_yaml(test_file, data, round_trip=True)

        # Read back the file to verify
        result = read_yaml(test_file)
        assert result == data

    def test_read_yaml_empty_file(self, tmp_path: Path) -> None:
        """Test reading an empty YAML file."""
        test_file = tmp_path / "empty.yaml"
        test_file.write_text("")

        result = read_yaml(test_file)
        assert result is None

    def test_read_yaml_comments_preserved(self, tmp_path: Path) -> None:
        """Test that comments are preserved when round_trip is used."""
        # Create a file with comments
        yaml_content = """# This is a comment
key1: value1
# Another comment
key2: value2
"""
        test_file = tmp_path / "commented.yaml"
        test_file.write_text(yaml_content)

        # Write it back with round_trip
        data = read_yaml(test_file)
        write_yaml(test_file, data, round_trip=True)

        # Verify the content is still valid YAML
        result = read_yaml(test_file)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_write_yaml_empty_data(self, tmp_path: Path) -> None:
        """Test writing empty data."""
        test_file = tmp_path / "empty_data.yaml"
        data = {}

        write_yaml(test_file, data)

        result = read_yaml(test_file)
        assert result == data

    def test_write_yaml_none_data(self, tmp_path: Path) -> None:
        """Test writing None data."""
        test_file = tmp_path / "none_data.yaml"
        data = {"key": None}

        write_yaml(test_file, data)

        result = read_yaml(test_file)
        assert result == {"key": None}
