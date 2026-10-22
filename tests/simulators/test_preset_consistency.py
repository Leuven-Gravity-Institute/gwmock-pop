"""Orientation and reference-frequency columns of every packaged preset and example config.

The orientation of a binary is written as ``inclination`` alone. ``theta_jn`` is
not an independent angle: for a spinning binary it follows from the inclination,
the spins and the reference frequency, and with anti-aligned spins it need not
equal the inclination. A second orientation column could only disagree with the
one the waveform is generated from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock_pop import list_presets
from gwmock_pop.simulators.graph import GraphSimulator

_N_SAMPLES = 256
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


@pytest.mark.parametrize(("label", "source"), _CASES, ids=[label for label, _ in _CASES])
def test_configurations_write_no_theta_jn(label: str, source: str | Path) -> None:
    """No packaged configuration writes ``theta_jn``; ``inclination`` is the orientation."""
    del label
    simulator = _build(source, 0)
    result = simulator.simulate(_N_SAMPLES)

    assert "theta_jn" not in simulator.parameter_names
    assert "theta_jn" not in result


@pytest.mark.parametrize(("label", "source"), _CASES, ids=[label for label, _ in _CASES])
def test_configurations_write_no_reference_frequency(label: str, source: str | Path) -> None:
    """No packaged configuration writes a per-event ``f_ref``.

    None of them draws in-plane spin, so nothing they write depends on the
    frequency the spins are defined at. A per-event value would only have to
    agree with the reference frequency the waveform generator is configured
    with, and could silently disagree with it instead.
    """
    del label
    assert "f_ref" not in _build(source, 0).parameter_names
