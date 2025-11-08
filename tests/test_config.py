import torch
import pytest

from porch.config import PorchConfig


class TestPorchConfig:
    """Test suite for PorchConfig."""

    def test_config_initialization_defaults(self):
        """Test that config initializes with default values."""
        config = PorchConfig()

        # Test default values
        assert config.epochs > 0
        assert config.lr > 0
        assert config.n_interior > 0
        assert config.n_boundary > 0
        assert isinstance(config.device, torch.device)
        assert config.optimizer_type in ["adam", "lbfgs"]

    def test_config_custom_values(self):
        """Test setting custom config values."""
        config = PorchConfig()

        config.epochs = 5000
        config.lr = 1e-3
        config.n_interior = 1000
        config.n_boundary = 100
        config.n_layers = 4
        config.n_neurons = 50

        assert config.epochs == 5000
        assert config.lr == 1e-3
        assert config.n_interior == 1000
        assert config.n_boundary == 100
        assert config.n_layers == 4
        assert config.n_neurons == 50

    def test_config_device_setting(self):
        """Test setting device."""
        config = PorchConfig()

        # Test CPU device
        config.device = torch.device("cpu")
        assert config.device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            config.device = torch.device("cuda")
            assert config.device.type == "cuda"

    def test_config_optimizer_types(self):
        """Test different optimizer types."""
        config = PorchConfig()

        config.optimizer_type = "adam"
        assert config.optimizer_type == "adam"

        config.optimizer_type = "lbfgs"
        assert config.optimizer_type == "lbfgs"

    def test_config_batch_settings(self):
        """Test batch-related settings."""
        config = PorchConfig()

        config.batch_size = 32
        config.batch_shuffle = True
        config.batch_cycle = True

        assert config.batch_size == 32
        assert config.batch_shuffle is True
        assert config.batch_cycle is True

    def test_config_print_frequencies(self):
        """Test print and summary frequencies."""
        config = PorchConfig()

        config.print_freq = 100
        config.summary_freq = 500

        assert config.print_freq == 100
        assert config.summary_freq == 500
