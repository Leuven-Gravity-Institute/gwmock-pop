"""Physical-consistency checks shared by every packaged preset and example config.

A non-precessing binary has its total angular momentum ``J`` parallel to its
orbital angular momentum ``L``, so ``theta_jn`` and ``inclination`` are the same
angle. A preset that draws them independently describes no physical binary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_pop import list_presets
from gwmock_pop.simulators.graph import GraphSimulator

_SEEDS = (0, 1, 7, 42, 2026)
_N_SAMPLES = 256
_IN_PLANE_SPINS = ("spin_1x", "spin_1y", "spin_2x", "spin_2y")
_EXAMPLE_CONFIGS = sorted((Path(__file__).resolve().parents[2] / "examples").rglob("*.yaml"))


def _simulators() -> list[tuple[str, GraphSimulator]]:
    """Return one simulator factory per packaged preset and example config."""
    return [*((name, name) for name in list_presets()), *((path.name, path) for path in _EXAMPLE_CONFIGS)]


def _build(source: str | Path, seed: int) -> GraphSimulator:
    """Build a seeded simulator from a preset name or a config path."""
    if isinstance(source, Path):
        return GraphSimulator.from_config_file(source, source_type="bbh", seed=seed)
    return GraphSimulator.from_preset(source, seed=seed)


_CASES = _simulators()


def test_example_configs_are_discovered() -> None:
    """The example configs exist, so the parametrisation below covers them."""
    assert _EXAMPLE_CONFIGS


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize(("label", "source"), _CASES, ids=[label for label, _ in _CASES])
def test_theta_jn_equals_inclination_for_non_precessing_draws(label: str, source: str | Path, seed: int) -> None:
    """Every non-precessing draw writes ``theta_jn`` equal to ``inclination``."""
    del label
    result = _build(source, seed).simulate(_N_SAMPLES)
    if "theta_jn" not in result:
        pytest.skip("this configuration writes no theta_jn")

    assert "inclination" in result
    precessing = any(name in result and bool(np.any(np.asarray(result[name]) != 0.0)) for name in _IN_PLANE_SPINS)
    assert not precessing, "no packaged configuration draws in-plane spin; extend this test if one does"
    np.testing.assert_array_equal(np.asarray(result["theta_jn"]), np.asarray(result["inclination"]))
