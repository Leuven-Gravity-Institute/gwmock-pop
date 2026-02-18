"""Random number generator manager."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array

logger = logging.getLogger("gwsim_pop")


class RNGManager:
    """Manager for random number generation with state persistence."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the RNG manager.

        Args:
            seed: Random seed. If None, draw a random integer between [0, 2**63).

        """
        self._seed = seed
        self._key = jax.random.key(secrets.randbelow(2**63) if seed is None else seed)
        self._saved_keys: list[Array] = []

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
        self._saved_keys.append(self._key)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        self._key = self._saved_keys.pop()

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
            value: The new key value as a jax.Array. If the dtype starts with
                ``key<``, it is used as-is. If the dtype is ``uint32``, it is
                wrapped into a PRNG key via ``jax.random.wrap_key_data``.

        Raises:
            TypeError: If the value is not a jax.Array or has an unsupported dtype.

        """
        if not isinstance(value, Array):
            raise ValueError("value has to be jax.Array.")
        value_dtype_str = str(value.dtype)
        if value_dtype_str.startswith("key<"):
            self._key = value
        elif value_dtype_str == "uint32":
            self._key = jax.random.wrap_key_data(value)
        else:
            raise TypeError(f"Unexpected dtype of value={value_dtype_str}.")

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

        Returns:
            A new key.

        """
        self._key, sub_key = jax.random.split(key=self._key)
        return sub_key

    def save_key(self, path: str | Path) -> None:
        """Save RNG key to file.

        Args:
            path: Path to save the key file.

        """
        path = Path(path)
        if path.suffix != ".npy":
            logger.warning(f"The suffix of path={path} is not '.npy'.")
            path = path.with_suffix(".npy")
            logger.warning(f"The path is updated to {path}.")
        jnp.save(file=path, arr=self.key_data)

    def load_key(self, path: str | Path) -> None:
        """Load RNG key from file.

        Args:
            path: Path to load the key file.

        """
        path = Path(path)
        if path.suffix != ".npy":
            logger.warning(f"The suffix of path={path} is not '.npy'.")
            path = path.with_suffix(".npy")
            logger.warning(f"The path is updated to {path}.")
        key_data = jnp.load(path)
        self.key_data = key_data
