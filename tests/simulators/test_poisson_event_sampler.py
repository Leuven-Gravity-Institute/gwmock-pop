"""Tests for the Poisson event sampler wrapper."""

from __future__ import annotations

import math
from typing import ClassVar

import jax.numpy as jnp
import numpy as np

from gwmock_pop import PoissonEventSampler
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators import PoissonEventSampler as SimulatorsPoissonEventSampler


class _RecordingSimulator:
    """Minimal wrapped simulator used to exercise the Poisson wrapper."""

    parameter_names: ClassVar[list[str]] = ["mass_1", "distance"]
    source_type = "bbh"

    def __init__(self, *, fail_on_zero: bool = False) -> None:
        """Initialize the stub simulator."""
        self.fail_on_zero = fail_on_zero
        self.calls: list[tuple[int, int | None]] = []

    def simulate(self, n_samples: int, **kwargs: int) -> dict[str, jnp.ndarray]:
        """Return deterministic arrays of the requested size."""
        if self.fail_on_zero and n_samples == 0:
            raise AssertionError("Wrapped simulator should not be called for zero events.")

        self.calls.append((n_samples, kwargs.get("seed")))
        samples = jnp.arange(n_samples, dtype=jnp.float32)
        return {
            "mass_1": samples,
            "distance": samples + 100.0,
        }


def _poisson_pmf(k: int, mean: float) -> float:
    """Return the Poisson probability mass for ``k``."""
    return math.exp(-mean + (k * math.log(mean)) - math.lgamma(k + 1))


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Approximate the upper-tail probability of a chi-squared variate."""
    z_score = (
        (statistic / degrees_of_freedom) ** (1.0 / 3.0) - (1.0 - (2.0 / (9.0 * degrees_of_freedom)))
    ) / math.sqrt(2.0 / (9.0 * degrees_of_freedom))
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def test_simulator_satisfies_protocol() -> None:
    """PoissonEventSampler structurally satisfies ``GWPopSimulator``."""
    assert isinstance(PoissonEventSampler(_RecordingSimulator(), rate_gpc3_yr=2.0, t_obs_yr=1.5), GWPopSimulator)


def test_parameter_names_and_source_type_delegate_to_wrapped_simulator() -> None:
    """The wrapper re-exposes the wrapped simulator metadata."""
    wrapped = _RecordingSimulator()
    sampler = PoissonEventSampler(wrapped, rate_gpc3_yr=1.0, t_obs_yr=1.0)

    assert sampler.parameter_names == wrapped.parameter_names
    assert sampler.source_type == wrapped.source_type


def test_rate_zero_returns_empty_mapping_with_correct_keys() -> None:
    """Zero expected rate returns empty arrays without calling the wrapped simulator."""
    wrapped = _RecordingSimulator(fail_on_zero=True)
    sampler = PoissonEventSampler(wrapped, rate_gpc3_yr=0.0, t_obs_yr=10.0)

    result = sampler.simulate(seed=123)

    assert list(result.keys()) == wrapped.parameter_names
    assert all(array.shape == (0,) for array in result.values())
    assert wrapped.calls == []


def test_event_counts_pass_poisson_chi_squared_check_over_many_seeds() -> None:
    """Event-count realizations match the configured Poisson distribution."""
    wrapped = _RecordingSimulator()
    mean = 3.0
    sampler = PoissonEventSampler(wrapped, rate_gpc3_yr=2.0, t_obs_yr=1.5)

    counts = np.asarray([int(sampler.simulate(seed=seed)["mass_1"].shape[0]) for seed in range(1000)])

    cutoff = 7
    observed = np.asarray(
        [(counts == k).sum() for k in range(cutoff + 1)] + [(counts >= cutoff + 1).sum()], dtype=float
    )
    expected = np.asarray(
        [1000.0 * _poisson_pmf(k, mean) for k in range(cutoff + 1)]
        + [1000.0 * (1.0 - sum(_poisson_pmf(k, mean) for k in range(cutoff + 1)))],
        dtype=float,
    )

    statistic = float(np.sum(((observed - expected) ** 2) / expected))
    p_value = _chi_square_survival(statistic, degrees_of_freedom=len(observed) - 1)

    assert p_value > 0.01, (statistic, p_value, observed, expected)


def test_public_import_surfaces_export_poisson_event_sampler() -> None:
    """The wrapper is importable from the package and simulators namespace."""
    assert SimulatorsPoissonEventSampler is PoissonEventSampler
