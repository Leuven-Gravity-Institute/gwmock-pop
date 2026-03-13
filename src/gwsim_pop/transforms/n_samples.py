"""A helper function to get the number of samples."""

from __future__ import annotations

from jax import Array


def get_n_samples(samples: Array) -> int:
    """Get number of samples.

    Args:
        samples: An array of samples.

    Returns:
        Number of samples.
    """
    return len(samples)
