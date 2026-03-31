"""Tests for RNGManager."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from gwmock_pop.rng import RNGManager


class TestRNGManager:
    """Test suite for RNGManager."""

    def test_init_with_seed(self) -> None:
        """Test initialization with a fixed seed."""
        rng = RNGManager(seed=42)
        assert rng._seed == 42
        assert isinstance(rng.key, jax.Array)

    def test_init_without_seed(self) -> None:
        """Test initialization without a seed."""
        rng = RNGManager(seed=None)
        assert rng._seed is None
        assert isinstance(rng.key, jax.Array)

    def test_key_property(self) -> None:
        """Test that key property returns a jax Array."""
        rng = RNGManager(seed=123)
        assert isinstance(rng.key, jax.Array)

    def test_key_property_setter(self) -> None:
        """Test setting RNG key."""
        rng1 = RNGManager(seed=456)
        key = rng1.key

        rng2 = RNGManager(seed=789)
        rng2.key = key

        assert jnp.array_equal(rng1.key, rng2.key)

    def test_save_key(self, tmp_path: Path) -> None:
        """Test saving RNG key to file."""
        rng = RNGManager(seed=111)

        output_path = tmp_path / "rng_key.npy"
        rng.save_key(output_path)

        assert Path(output_path).exists()

    def test_load_key(self, tmp_path: Path) -> None:
        """Test loading RNG key from file."""
        rng1 = RNGManager(seed=222)

        output_path = tmp_path / "rng_key_load.npy"
        rng1.save_key(output_path)

        rng2 = RNGManager(seed=333)
        rng2.load_key(output_path)

        assert jnp.array_equal(rng1.key, rng2.key)

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same random numbers."""
        rng1 = RNGManager(seed=444)
        rng2 = RNGManager(seed=444)

        _, subkey1 = jax.random.split(rng1.key)
        _, subkey2 = jax.random.split(rng2.key)

        val1 = jax.random.uniform(subkey1)
        val2 = jax.random.uniform(subkey2)

        assert val1 == val2

    def test_different_seeds_produce_different_values(self) -> None:
        """Test that different seeds produce different random numbers."""
        rng1 = RNGManager(seed=555)
        rng2 = RNGManager(seed=666)

        _, subkey1 = jax.random.split(rng1.key)
        _, subkey2 = jax.random.split(rng2.key)

        val1 = jax.random.uniform(subkey1)
        val2 = jax.random.uniform(subkey2)

        assert val1 != val2

    def test_key_persistence_across_calls(self) -> None:
        """Test that RNG key persists between calls."""
        rng = RNGManager(seed=777)

        key1, subkey1 = jax.random.split(rng.key)
        _, subkey2 = jax.random.split(key1)

        val1 = jax.random.uniform(subkey1)
        val2 = jax.random.uniform(subkey2)

        assert val1 != val2

    def test_save_key_with_string_path(self, tmp_path: Path) -> None:
        """Test saving key with string path."""
        rng = RNGManager(seed=888)
        output_path = str(tmp_path / "rng_key_str.npy")

        rng.save_key(output_path)

        assert Path(output_path).exists()

    def test_load_key_with_string_path(self, tmp_path: Path) -> None:
        """Test loading key with string path."""
        rng1 = RNGManager(seed=999)
        output_path = str(tmp_path / "rng_key_load_str.npy")

        rng1.save_key(output_path)

        rng2 = RNGManager(seed=101)
        rng2.load_key(output_path)

        assert jnp.array_equal(rng1.key, rng2.key)

    def test_multiple_save_load_cycles(self, tmp_path: Path) -> None:
        """Test multiple save and load cycles."""
        rng = RNGManager(seed=121)

        for i in range(3):
            key_before = rng.key
            rng.save_key(tmp_path / f"key_cycle_{i}.npy")
            rng.load_key(tmp_path / f"key_cycle_{i}.npy")
            key_after = rng.key

            assert jnp.array_equal(key_before, key_after)

    def test_key_generates_different_values_on_each_call(self) -> None:
        """Test that RNG generates different values on successive calls."""
        rng = RNGManager(seed=131)

        key = rng.key
        values = []
        for _ in range(5):
            key, subkey = jax.random.split(key)
            values.append(jax.random.uniform(subkey))

        assert len(values) == 5
        assert all(0 <= v < 1 for v in values)

    def test_key_can_produce_integers(self) -> None:
        """Test that RNG can generate integers."""
        rng = RNGManager(seed=141)

        _, subkey = jax.random.split(rng.key)
        val = jax.random.randint(subkey, shape=(), minval=0, maxval=100)

        assert isinstance(val, jax.Array)
        assert 0 <= val < 100

    def test_key_can_produce_normal_distribution(self) -> None:
        """Test that RNG can generate normal distribution."""
        rng = RNGManager(seed=151)

        _, subkey = jax.random.split(rng.key)
        values = jax.random.normal(subkey, shape=(100,))

        assert len(values) == 100
        assert isinstance(values, jax.Array)

    def test_repr_with_seed(self) -> None:
        """Test __repr__ with a seed."""
        rng = RNGManager(seed=42)

        repr_str = repr(rng)

        assert repr_str == "RNGManager(seed=42)"

    def test_repr_without_seed(self) -> None:
        """Test __repr__ without a seed."""
        rng = RNGManager(seed=None)

        repr_str = repr(rng)

        assert repr_str == "RNGManager(seed=None)"

    def test_context_manager_saves_and_restores_key(self) -> None:
        """Test that context manager saves and restores RNG key."""
        rng = RNGManager(seed=123)
        key_before = rng.key

        with rng:
            # Generate a new key inside context
            new_key1 = rng.new_key
            new_key2 = rng.new_key

            # Keys should be different from original
            assert not jnp.array_equal(key_before, new_key1)
            assert not jnp.array_equal(key_before, new_key2)
            assert not jnp.array_equal(new_key1, new_key2)

        # After context, key should be restored to original
        assert jnp.array_equal(key_before, rng.key)

    def test_context_manager_with_exception(self) -> None:
        """Test that context manager restores key even when exception occurs."""
        rng = RNGManager(seed=456)
        key_before = rng.key

        def func(rng):
            # Generate a new key
            _ = rng.new_key

            # Raise an exception
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"), rng:
            func(rng)

        # After exception, key should still be restored
        assert jnp.array_equal(key_before, rng.key)

    def test_key_setter_with_invalid_type(self) -> None:
        """Test key setter raises ValueError with invalid type."""
        rng = RNGManager(seed=789)

        with pytest.raises(ValueError, match=r"value has to be jax.Array."):
            rng.key = "invalid"

    def test_key_data_property_getter(self) -> None:
        """Test key_data property getter."""
        rng = RNGManager(seed=101)
        key_data = rng.key_data

        assert isinstance(key_data, jax.Array)
        # Key data should be a 2-element array for PRNG keys
        assert key_data.shape == (2,)

    def test_key_data_property_setter(self) -> None:
        """Test key_data property setter."""
        rng1 = RNGManager(seed=202)
        key_data = rng1.key_data

        rng2 = RNGManager(seed=303)
        rng2.key_data = key_data

        assert jnp.array_equal(rng1.key, rng2.key)

    def test_new_key_property(self) -> None:
        """Test new_key property updates internal key and produces different keys."""
        rng = RNGManager(seed=404)

        key1 = rng.new_key
        key2 = rng.new_key
        key3 = rng.new_key

        # All keys should be different
        assert not jnp.array_equal(key1, key2)
        assert not jnp.array_equal(key2, key3)
        assert not jnp.array_equal(key1, key3)

    def test_save_key_with_invalid_path(self, tmp_path: Path) -> None:
        """Test saving key with invalid path raises error."""
        rng = RNGManager(seed=505)
        invalid_path = tmp_path / "nonexistent_dir" / "rng_key.npy"

        with pytest.raises(OSError, match="No such file or directory"):
            rng.save_key(invalid_path)

    def test_load_key_with_invalid_path(self) -> None:
        """Test loading key with invalid path raises error."""
        rng = RNGManager(seed=606)
        invalid_path = Path("/nonexistent/path/rng_key.npy")

        with pytest.raises(FileNotFoundError):
            rng.load_key(invalid_path)

    def test_load_key_with_corrupted_file(self, tmp_path: Path) -> None:
        """Test loading key with corrupted file raises error."""
        rng = RNGManager(seed=707)
        corrupted_path = tmp_path / "corrupted_key.npy"

        # Write invalid data
        with open(corrupted_path, "w") as f:
            f.write("not a valid numpy file")

        with pytest.raises(Exception, match=r"This file contains pickled"):
            rng.load_key(corrupted_path)
