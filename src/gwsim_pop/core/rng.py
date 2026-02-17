"""Random number generator manager."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array


class RNGManager:
    """Manager for random number generation with state persistence."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the RNG manager.

        Args:
            seed: Random seed. If None, draw a random integer between [0, 2**63).

        """
        self._seed = seed
        self._key = jax.random.key(secrets.randbelow(2**63) if seed is None else seed)

    def __repr__(self) -> str:
        """Return the string representation.

        Returns:
            String representation.

        """
        return f"RNGManager(seed={self._seed})"

    def __enter__(self) -> RNGManager:
        """Enter the context manager.

        Use the context manager when keeping the initial state of the RNG is preferred.

        Returns:
            RNGManager.

        """
        self._saved_key = self._key
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        self._key = self._saved_key

    @property
    def key(self) -> Array:
        """Get the random number generator key.

        Returns:
            JAX random key.

        """
        return self._key

    @key.setter
    def key(self, value: Array) -> None:
        """Set the random number generator key.

        Args:
            value: The new key value, which can be either a jax._src.prng.PRNGKeyArray
                or a jax.Array. If a jax.Array is provided, it will be wrapped into
                a PRNGKeyArray.

        Raises:
            ValueError: If the value is neither a PRNGKeyArray nor a jax.Array.

        """
        if not isinstance(value, Array):
            raise ValueError("value has to be jax.Array.")
        value_dtype_str = str(value.dtype)
        if value_dtype_str == "key<fry>":
            self._key = value
        elif value_dtype_str == "uint32":
            self._key = jax.random.wrap_key_data(value)
        else:
            raise ValueError(f"Expected dtype of value={value_dtype_str}.")

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

    @property
    def new_key(self) -> Array:
        """Get a new key.

        The internal key is updated to the new key.

        Returns:
            A new key.

        """
        _, sub_key = jax.random.split(key=self._key)
        self._key = cast(Array, sub_key)
        return sub_key

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
