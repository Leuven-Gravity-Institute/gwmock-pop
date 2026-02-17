"""Random number generator manager."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.prng import PRNGKeyArray


class RNGManager:
    """Manager for random number generation with state persistence."""

    _key: PRNGKeyArray = cast(PRNGKeyArray, jax.random.key(int.from_bytes(os.urandom(4), "big")))

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the RNG manager.

        Args:
            seed: Random seed. If None, uses system entropy.

        """
        self._seed = seed
        if seed is not None:
            self._key = cast(PRNGKeyArray, jax.random.key(seed))

    @property
    def key(self) -> PRNGKeyArray:
        """Get the random number generator key.

        Returns:
            JAX random key.

        """
        return self._key

    @key.setter
    def key(self, value: PRNGKeyArray | Array) -> None:
        """Set the random number generator key.

        Args:
            value: The new key value, which can be either a jax._src.prng.PRNGKeyArray
                or a jax.Array. If a jax.Array is provided, it will be wrapped into
                a PRNGKeyArray.

        Raises:
            ValueError: If the value is neither a PRNGKeyArray nor a jax.Array.

        """
        if isinstance(value, PRNGKeyArray):
            self._key = value
        elif isinstance(value, jax.Array):
            self._key = cast(PRNGKeyArray, jax.random.wrap_key_data(value))
        else:
            raise ValueError(f"value = {value} has to be either a PRNGKeyArray or a jax Array.")

    @property
    def key_data(self) -> Array:
        """Get the key data.

        The function can be used to retrieve the key data for storing in disk.

        Returns:
            Key data.

        """
        return jax.random.key_data(self._key)

    @key_data.setter
    def key_data(self, value: Array) -> None:
        """Set the key by key data.

        Args:
            value: Key data.

        """
        self._key = jax.random.wrap_key_data(value)

    def save_key(self, path: str | Path) -> None:
        """Save RNG key to file.

        Args:
            path: Path to save the key file.

        """
        path = Path(path)
        jnp.save(file=path, arr=self.key_data)

    def load_key(self, path: str | Path) -> None:
        """Load RNG key from file.

        Args:
            path: Path to load the key file.

        """
        path = Path(path)
        key_data = jnp.load(path)
        self.key_data = key_data
