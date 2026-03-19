"""Tests for RandomMixin."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gwmock_pop.mixins.random import RandomMixin
from gwmock_pop.rng import RNGManager


class TestRandomMixin:
    """Test suite for RandomMixin."""

    def test_init_with_seed(self) -> None:
        """Test initialization with a fixed seed."""
        mixin = RandomMixin(seed=42)

        assert mixin._rng_manager._seed == 42  # noqa: PLR2004
        assert isinstance(mixin.rng_manager, RNGManager)
        assert isinstance(mixin.rng_manager.key, jax.Array)

    def test_init_without_seed(self) -> None:
        """Test initialization without a seed."""
        mixin = RandomMixin(seed=None)

        assert mixin._rng_manager._seed is None
        assert isinstance(mixin.rng_manager, RNGManager)
        assert isinstance(mixin.rng_manager.key, jax.Array)

    def test_rng_manager_property(self) -> None:
        """Test that rng_manager property returns an RNGManager instance."""
        mixin = RandomMixin(seed=123)

        rng_manager = mixin.rng_manager

        assert isinstance(rng_manager, RNGManager)

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same RNG state."""
        mixin1 = RandomMixin(seed=444)
        mixin2 = RandomMixin(seed=444)

        # Keys should be equal for same seeds
        assert jnp.array_equal(mixin1.rng_manager.key, mixin2.rng_manager.key)

    def test_different_seeds_produce_different_states(self) -> None:
        """Test that different seeds produce different RNG states."""
        mixin1 = RandomMixin(seed=555)
        mixin2 = RandomMixin(seed=666)

        # Keys should be different for different seeds
        assert not jnp.array_equal(mixin1.rng_manager.key, mixin2.rng_manager.key)

    def test_rng_manager_generates_different_values(self) -> None:
        """Test that RNG manager generates different values on successive calls."""
        mixin = RandomMixin(seed=777)

        key1 = mixin.rng_manager.new_key
        key2 = mixin.rng_manager.new_key

        # Generate values from different keys
        val1 = jax.random.uniform(key1)
        val2 = jax.random.uniform(key2)

        assert val1 != val2

    def test_rng_manager_with_context_manager(self) -> None:
        """Test that RNG manager works correctly with context manager."""
        mixin = RandomMixin(seed=123)
        key_before = mixin.rng_manager.key

        with mixin.rng_manager:
            # Generate a new key inside context
            new_key1 = mixin.rng_manager.new_key
            new_key2 = mixin.rng_manager.new_key

            # Keys should be different from original
            assert not jnp.array_equal(key_before, new_key1)
            assert not jnp.array_equal(key_before, new_key2)

        # After context, key should be restored to original
        assert jnp.array_equal(key_before, mixin.rng_manager.key)

    def test_instantiation_with_additional_args(self) -> None:
        """Test that RandomMixin can be instantiated with additional args."""

        # Create a class that uses RandomMixin with additional __init__ args
        class TestClass(RandomMixin):
            def __init__(self, value: int, seed: int | None = None) -> None:
                self.value = value
                super().__init__(seed=seed)

        obj = TestClass(value=42, seed=100)

        assert obj.value == 42  # noqa: PLR2004
        assert obj._rng_manager._seed == 100  # noqa: PLR2004

    def test_instantiation_with_kwargs(self) -> None:
        """Test that RandomMixin can be instantiated with kwargs."""

        class TestClass(RandomMixin):
            def __init__(self, **kwargs) -> None:
                self.config = kwargs.get("config", {})
                super().__init__(seed=kwargs.get("seed"))

        obj = TestClass(config={"key": "value"}, seed=200)

        assert obj.config == {"key": "value"}
        assert obj._rng_manager._seed == 200  # noqa: PLR2004

    def test_rng_key_can_generate_various_distributions(self) -> None:
        """Test that RNG from mixin can generate various distributions."""
        mixin = RandomMixin(seed=300)

        key = mixin.rng_manager.key

        # Uniform distribution
        _, subkey = jax.random.split(key)
        uniform = jax.random.uniform(subkey, shape=(10,))

        # Normal distribution
        _, subkey = jax.random.split(subkey)
        normal = jax.random.normal(subkey, shape=(10,))

        # Integer distribution
        _, subkey = jax.random.split(subkey)
        integer = jax.random.randint(subkey, shape=(10,), minval=0, maxval=100)

        assert uniform.shape == (10,)
        assert normal.shape == (10,)
        assert integer.shape == (10,)

    def test_rng_manager_repr(self) -> None:
        """Test that RNG manager has correct string representation."""
        mixin_with_seed = RandomMixin(seed=42)
        mixin_without_seed = RandomMixin(seed=None)

        repr_with_seed = repr(mixin_with_seed.rng_manager)
        repr_without_seed = repr(mixin_without_seed.rng_manager)

        assert repr_with_seed == "RNGManager(seed=42)"
        assert repr_without_seed == "RNGManager(seed=None)"

    def test_multiple_mixins_with_same_seed(self) -> None:
        """Test that multiple instances with same seed have identical RNG states."""
        mixins = [RandomMixin(seed=999) for _ in range(3)]

        # All should have the same key
        keys = [m.rng_manager.key for m in mixins]
        for i in range(1, len(keys)):
            assert jnp.array_equal(keys[0], keys[i])

    def test_rng_can_save_and_load_key(self, tmp_path) -> None:
        """Test that RNG manager can save and load keys."""
        mixin1 = RandomMixin(seed=500)

        output_path = tmp_path / "mixin_rng.npy"
        mixin1.rng_manager.save_key(output_path)

        mixin2 = RandomMixin(seed=600)
        mixin2.rng_manager.load_key(output_path)

        assert jnp.array_equal(mixin1.rng_manager.key, mixin2.rng_manager.key)

    def test_rng_manager_key_data_property(self) -> None:
        """Test that RNG manager can access key data."""
        mixin = RandomMixin(seed=700)

        key_data = mixin.rng_manager.key_data

        assert isinstance(key_data, jax.Array)
        assert key_data.shape == (2,)

    def test_rng_manager_new_key_property(self) -> None:
        """Test that new_key property generates unique keys."""
        mixin = RandomMixin(seed=800)

        key1 = mixin.rng_manager.new_key
        key2 = mixin.rng_manager.new_key
        key3 = mixin.rng_manager.new_key

        # All keys should be different
        assert not jnp.array_equal(key1, key2)
        assert not jnp.array_equal(key2, key3)
        assert not jnp.array_equal(key1, key3)

    def test_rng_key_data_property(self) -> None:
        """Test that rng_key_data property returns key data."""
        mixin = RandomMixin(seed=700)

        key_data = mixin.rng_key_data

        assert isinstance(key_data, jax.Array)
        assert key_data.shape == (2,)

    def test_rng_key_data_matches_rng_manager(self) -> None:
        """Test that rng_key_data matches rng_manager.key_data."""
        mixin = RandomMixin(seed=800)

        assert jnp.array_equal(mixin.rng_key_data, mixin.rng_manager.key_data)

    def test_rng_key_data_reproducibility(self) -> None:
        """Test that same seed produces same key data."""
        mixin1 = RandomMixin(seed=900)
        mixin2 = RandomMixin(seed=900)

        assert jnp.array_equal(mixin1.rng_key_data, mixin2.rng_key_data)

    def test_rng_key_data_with_different_seeds(self) -> None:
        """Test that different seeds produce different key data."""
        mixin1 = RandomMixin(seed=910)
        mixin2 = RandomMixin(seed=920)

        assert not jnp.array_equal(mixin1.rng_key_data, mixin2.rng_key_data)
