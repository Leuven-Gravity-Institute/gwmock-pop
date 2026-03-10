"""Mixin class to handle random number generation."""

from __future__ import annotations

from jax import Array

from gwsim_pop.rng.rng import RNGManager


class RandomMixin:
    """Mixin class to handle random number generation."""

    def __init__(self, seed: int | None = None, *args, **kwargs) -> None:
        """Initialize the instance.

        Args:
            seed: Seed to initialize a random number generator.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self._rng_manager = RNGManager(seed=seed)
        super().__init__(*args, **kwargs)

    @property
    def rng_manager(self) -> RNGManager:
        """Get the RNG manager.

        Returns:
            RNG manager.
        """
        return self._rng_manager

    @property
    def rng_key_data(self) -> Array:
        """Get the key data of the random number generator.

        Returns:
            Key data of the random number generator.
        """
        return self._rng_manager.key_data
