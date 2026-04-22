"""Protocol conformance tests for ExternalPopulationLoader.

Two test classes verify complementary properties of
``ExternalPopulationLoader``:

``TestExternalPopulationLoaderStructure``
    Plain pytest tests that verify the structural contract: conformant
    external loaders satisfy both ``GWPopSimulator`` and
    ``ExternalPopulationLoader``, while classes missing required members fail
    the runtime ``isinstance`` check.

``TestExternalPopulationLoaderContracts``
    Hypothesis-driven tests parametrized over both minimal loader fixtures.
    They exercise the same invariants as the base simulator protocol and pin
    the expected ``ValueError`` behavior for unsupported file formats.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import Array

from gwmock_pop.protocols import ExternalPopulationLoader, GWPopSimulator


class _MinimalHDF5Loader:
    """Minimal HDF5-backed loader satisfying both protocols."""

    parameter_names: ClassVar[list[str]] = ["mass_1", "mass_2", "redshift"]
    source_type = "bbh"
    _supported_formats: ClassVar[set[str]] = {"hdf5", "h5"}

    def __init__(self, file_format: str = "hdf5") -> None:
        """Validate the requested file format."""
        if file_format not in self._supported_formats:
            raise ValueError(f"Unsupported file format for external population loader: {file_format}")

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
        """Return constant 1-D arrays of length ``n_samples``."""
        return {key: jnp.full((n_samples,), index + 1.0) for index, key in enumerate(self.parameter_names)}


class _MinimalCSVLoader:
    """Minimal CSV-backed loader satisfying both protocols."""

    parameter_names: ClassVar[list[str]] = ["chirp_mass", "mass_ratio"]
    source_type = "bns"
    _supported_formats: ClassVar[set[str]] = {"csv"}

    def __init__(self, file_format: str = "csv") -> None:
        """Validate the requested file format."""
        if file_format not in self._supported_formats:
            raise ValueError(f"Unsupported file format for external population loader: {file_format}")

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
        """Return constant 1-D arrays of length ``n_samples``."""
        return {key: jnp.full((n_samples,), index + 1.0) for index, key in enumerate(self.parameter_names)}


_LOADERS = [_MinimalHDF5Loader(), _MinimalCSVLoader()]


class TestExternalPopulationLoaderStructure:
    """Verify that loader protocol membership checks are correct."""

    def test_hdf5_loader_satisfies_both_protocols(self) -> None:
        """Test that a conformant HDF5 loader satisfies both protocols."""
        loader = _MinimalHDF5Loader()
        assert isinstance(loader, GWPopSimulator)
        assert isinstance(loader, ExternalPopulationLoader)

    def test_csv_loader_satisfies_both_protocols(self) -> None:
        """Test that a conformant CSV loader satisfies both protocols."""
        loader = _MinimalCSVLoader()
        assert isinstance(loader, GWPopSimulator)
        assert isinstance(loader, ExternalPopulationLoader)

    def test_missing_parameter_names_fails(self) -> None:
        """Test that a class without parameter_names fails the isinstance check."""

        class _NoParameterNames:
            source_type = "bbh"

            def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
                """Return empty mapping."""
                return {}

        assert not isinstance(_NoParameterNames(), ExternalPopulationLoader)

    def test_missing_source_type_fails(self) -> None:
        """Test that a class without source_type fails the isinstance check."""

        class _NoSourceType:
            parameter_names: ClassVar[list[str]] = ["mass_1"]

            def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
                """Return empty mapping."""
                return {}

        assert not isinstance(_NoSourceType(), ExternalPopulationLoader)

    def test_missing_simulate_fails(self) -> None:
        """Test that a class without simulate fails the isinstance check."""

        class _NoSimulate:
            parameter_names: ClassVar[list[str]] = ["mass_1"]
            source_type = "bbh"

        assert not isinstance(_NoSimulate(), ExternalPopulationLoader)

    @pytest.mark.parametrize(
        ("loader_class", "file_format"),
        [(_MinimalHDF5Loader, "fits"), (_MinimalCSVLoader, "json")],
        ids=["hdf5", "csv"],
    )
    def test_unsupported_file_format_raises_value_error(
        self,
        loader_class: type[_MinimalHDF5Loader] | type[_MinimalCSVLoader],
        file_format: str,
    ) -> None:
        """Test that unsupported formats raise a clear ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader_class(file_format=file_format)


class TestExternalPopulationLoaderContracts:
    """Hypothesis-driven contract invariants, parametrized over both loaders."""

    @pytest.mark.parametrize("loader", _LOADERS, ids=["hdf5", "csv"])
    @settings(max_examples=25)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_simulate_keys_match_parameter_names(self, loader: ExternalPopulationLoader, n_samples: int) -> None:
        """Test that simulate() returns exactly the keys in parameter_names."""
        result = loader.simulate(n_samples)
        assert list(result.keys()) == list(loader.parameter_names)

    @pytest.mark.parametrize("loader", _LOADERS, ids=["hdf5", "csv"])
    @settings(max_examples=25)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_simulate_arrays_have_n_samples_leading_dim(
        self,
        loader: ExternalPopulationLoader,
        n_samples: int,
    ) -> None:
        """Test that every output array has leading dimension equal to n_samples."""
        result = loader.simulate(n_samples)
        for array in result.values():
            assert array.shape[0] == n_samples

    @pytest.mark.parametrize("loader", _LOADERS, ids=["hdf5", "csv"])
    @settings(max_examples=20)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_source_type_is_non_empty_string(self, loader: ExternalPopulationLoader, n_samples: int) -> None:
        """Test that source_type is a non-empty string."""
        assert isinstance(loader.source_type, str)
        assert len(loader.source_type) > 0

    @pytest.mark.parametrize("loader", _LOADERS, ids=["hdf5", "csv"])
    def test_parameter_names_is_stable(self, loader: ExternalPopulationLoader) -> None:
        """Test that parameter_names returns the same sequence on repeated access."""
        first = list(loader.parameter_names)
        loader.simulate(1)
        second = list(loader.parameter_names)
        third = list(loader.parameter_names)
        assert first == second == third

    @pytest.mark.parametrize("loader", _LOADERS, ids=["hdf5", "csv"])
    @settings(max_examples=20)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_no_extra_keys_beyond_parameter_names(self, loader: ExternalPopulationLoader, n_samples: int) -> None:
        """Test that simulate() produces no keys beyond parameter_names."""
        result = loader.simulate(n_samples)
        assert set(result.keys()) <= set(loader.parameter_names)

    @pytest.mark.parametrize("loader", _LOADERS, ids=["hdf5", "csv"])
    @settings(max_examples=20)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_simulate_result_values_are_jax_arrays(self, loader: ExternalPopulationLoader, n_samples: int) -> None:
        """Test that all values in the simulate() result are jax.Array instances."""
        result = loader.simulate(n_samples)
        for value in result.values():
            assert isinstance(value, Array)
