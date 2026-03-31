"""Tests for run configuration."""

from gwmock_pop.config.run import RunConfiguration


class TestRunConfiguration:
    """Tests for RunConfiguration class."""

    def test_initialization(self):
        """Test that RunConfiguration can be initialized."""
        config = RunConfiguration()
        assert isinstance(config, RunConfiguration)

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RunConfiguration()
        assert config.name == "simulation"
        assert config.seed == 42
        assert config.mode == "fixed_n_samples"
        assert config.n_samples == 1_000_000
        assert config.duration == 1.0

    def test_initialization_with_values(self):
        """Test that RunConfiguration can be initialized with specific values."""
        config = RunConfiguration(name="test_run", seed=123, mode="fixed_n_samples", n_samples=500000, duration=2.0)
        assert config.name == "test_run"
        assert config.seed == 123
        assert config.mode == "fixed_n_samples"
        assert config.n_samples == 500000
        assert config.duration == 2.0

    def test_mode_validation(self):
        """Test that mode validation works correctly."""
        # Test valid modes
        config1 = RunConfiguration(mode="fixed_n_samples")
        config2 = RunConfiguration(mode="duration")
        assert config1.mode == "fixed_n_samples"
        assert config2.mode == "duration"

    def test_output_and_logging_defaults(self):
        """Test that output and logging configurations are properly initialized."""
        config = RunConfiguration()
        assert hasattr(config, "output")
        assert hasattr(config, "logging")
