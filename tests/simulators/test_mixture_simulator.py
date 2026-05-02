"""Tests for the mixture simulator wrapper."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop import MixtureSimulator
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import MixtureSimulator as SimulatorsMixtureSimulator


class _ConstantComponentSimulator:
    """Minimal component simulator that emits a fixed component label."""

    parameter_names: ClassVar[list[str]] = ["mass_1", "component_label"]
    source_type = "bbh"

    def __init__(
        self, component_label: float, *, parameter_names: list[str] | None = None, source_type: str = "bbh"
    ) -> None:
        """Initialize the component stub."""
        self._component_label = float(component_label)
        self.parameter_names = self.parameter_names if parameter_names is None else parameter_names
        self.source_type = source_type
        self.calls: list[tuple[int, int | None]] = []

    def simulate(self, n_samples: int, **kwargs: Any) -> dict[str, jnp.ndarray]:
        """Return constant component labels with a deterministic ramp."""
        self.calls.append((n_samples, kwargs.get("seed")))
        return {
            "mass_1": jnp.arange(n_samples, dtype=jnp.float32) + self._component_label,
            "component_label": jnp.full((n_samples,), self._component_label),
        }


def test_simulator_satisfies_protocol() -> None:
    """MixtureSimulator structurally satisfies ``GWPopSimulator``."""
    assert isinstance(
        MixtureSimulator(
            [_ConstantComponentSimulator(0.0), _ConstantComponentSimulator(1.0)],
            weights=[1.0, 3.0],
        ),
        GWPopSimulator,
    )


def test_component_fractions_match_requested_weights_within_three_sigma() -> None:
    """Large draws recover the requested mixture weights."""
    simulator = MixtureSimulator(
        [_ConstantComponentSimulator(0.0), _ConstantComponentSimulator(1.0)],
        weights=[1.0, 3.0],
    )

    result = simulator.simulate(10_000, seed=123)
    labels = np.asarray(result["component_label"])

    for label, probability in ((0.0, 0.25), (1.0, 0.75)):
        observed = int(np.count_nonzero(labels == label))
        expected = 10_000 * probability
        sigma = math.sqrt(10_000 * probability * (1.0 - probability))
        assert abs(observed - expected) <= 3.0 * sigma


def test_mismatched_parameter_names_raise_value_error_at_construction() -> None:
    """Component simulators must agree on the parameter-name set."""
    with pytest.raises(ValueError, match="same parameter_names set"):
        MixtureSimulator(
            [
                _ConstantComponentSimulator(0.0),
                _ConstantComponentSimulator(1.0, parameter_names=["mass_1", "distance"]),
            ],
            weights=[0.5, 0.5],
        )


def test_single_component_case_delegates_all_samples() -> None:
    """A single-component mixture behaves like the wrapped simulator."""
    component = _ConstantComponentSimulator(7.0)
    simulator = MixtureSimulator([component], weights=[10.0])

    result = simulator.simulate(32, seed=99)

    assert component.calls
    assert component.calls[0][0] == 32
    np.testing.assert_array_equal(np.asarray(result["component_label"]), np.full(32, 7.0))


def test_source_type_mismatch_raises_value_error_at_construction() -> None:
    """Component simulators must agree on the source type."""
    with pytest.raises(ValueError, match=r"share the same source_type"):
        MixtureSimulator(
            [
                _ConstantComponentSimulator(0.0, source_type="bbh"),
                _ConstantComponentSimulator(1.0, source_type="nsbh"),
            ],
            weights=[0.5, 0.5],
        )


def test_public_import_surfaces_export_mixture_simulator() -> None:
    """The wrapper is importable from the package and simulators namespace."""
    assert SimulatorsMixtureSimulator is MixtureSimulator
