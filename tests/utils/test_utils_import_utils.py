"""Tests for import_utils."""

from __future__ import annotations

import pytest

from gwsim_pop.utils.import_utils import import_from_string


class TestImportFromString:
    """Test suite for import_from_string function."""

    def test_import_class_from_module(self) -> None:
        """Test importing a class from a module."""
        from gwsim_pop.simulators.simulator import Simulator  # noqa: PLC0415

        result = import_from_string("gwsim_pop.simulators.simulator.Simulator")
        assert result is Simulator

    def test_import_function_from_module(self) -> None:
        """Test importing a function from a module."""
        from gwsim_pop.utils.log import setup_logger  # noqa: PLC0415

        result = import_from_string("gwsim_pop.utils.log.setup_logger")
        assert result is setup_logger

    def test_import_submodule_object(self) -> None:
        """Test importing from a submodule."""
        from gwsim_pop.simulators.bbh.base import BBHSimulator  # noqa: PLC0415

        result = import_from_string("gwsim_pop.simulators.bbh.base.BBHSimulator")
        assert result is BBHSimulator

    def test_import_with_default_module(self) -> None:
        """Test importing with default module path."""
        from gwsim_pop.simulators.simulator import Simulator  # noqa: PLC0415

        result = import_from_string("Simulator", default_module="gwsim_pop.simulators.simulator")
        assert result is Simulator

    def test_import_with_default_module_when_full_path_provided(self) -> None:
        """Test that full path takes precedence over default module."""
        from gwsim_pop.simulators.simulator import Simulator  # noqa: PLC0415

        result = import_from_string(
            "gwsim_pop.simulators.simulator.Simulator",
            default_module="gwsim_pop.simulators.bbh.base",
        )
        assert result is Simulator

    def test_import_nonexistent_object(self) -> None:
        """Test importing a non-existent object raises ImportError."""
        with pytest.raises(ImportError, match="Could not import"):
            import_from_string("gwsim_pop.simulators.simulator.NonExistentClass")

    def test_import_nonexistent_module(self) -> None:
        """Test importing from a non-existent module raises ImportError."""
        with pytest.raises(ImportError, match="Could not import"):
            import_from_string("gwsim_pop.nonexistent.module.Class")

    def test_import_without_default_module(self) -> None:
        """Test that importing without default module and full path fails."""
        with pytest.raises(ImportError, match="does not contain a module path"):
            import_from_string("Simulator")

    def test_import_from_top_level_package(self) -> None:
        """Test importing from top-level package."""
        from gwsim_pop.version import __version__  # noqa: PLC0415

        result = import_from_string("gwsim_pop.version.__version__")
        assert result == __version__

    def test_import_returns_correct_type(self) -> None:
        """Test that imported object has correct type."""
        result = import_from_string("gwsim_pop.simulators.simulator.Simulator")
        assert callable(result)

    def test_import_function_with_args(self) -> None:
        """Test importing a function that can be called."""
        import jax.numpy as jnp  # noqa: PLC0415

        from gwsim_pop.conversion.cbc import compute_chirp_mass_from_mass_1_mass_2  # noqa: PLC0415

        result = import_from_string("gwsim_pop.conversion.cbc.compute_chirp_mass_from_mass_1_mass_2")
        assert result is compute_chirp_mass_from_mass_1_mass_2
        # Test that it's callable
        chirp_mass = result(mass_1=jnp.array(10.0), mass_2=jnp.array(5.0))
        assert chirp_mass.shape == ()
