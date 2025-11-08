import torch
import pytest

from porch.network import FullyConnected, xavier_trunc_normal_, init_weights_trunc_normal_


class TestFullyConnected:
    """Test suite for FullyConnected neural network."""

    def test_network_initialization(self):
        """Test that network is properly initialized."""
        network = FullyConnected(d_in=2, d_out=1, n_hidden_layers=3, n_neurons=20)

        assert network.d_in == 2
        assert network.d_out == 1
        assert network.mean is None
        assert network.std is None

        # Check that network has correct structure
        # Input layer + hidden layers + output layer + activations
        # (1 input Linear + 1 Tanh) + (3 hidden Linear + 3 Tanh) + (1 output Linear)
        expected_layers = 1 + 1 + 3 + 3 + 1  # 9 layers total
        assert len(network.net) == expected_layers

    def test_network_forward_pass(self):
        """Test forward pass with random input."""
        network = FullyConnected(d_in=2, d_out=1, n_hidden_layers=2, n_neurons=10)

        batch_size = 5
        input_data = torch.randn(batch_size, 2)

        output = network.forward(input_data)

        assert output.shape == torch.Size([batch_size, 1])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_network_with_normalization(self):
        """Test network with input normalization."""
        network = FullyConnected(d_in=2, d_out=1, n_hidden_layers=2, n_neurons=10)

        mean = torch.tensor([0.5, 0.5])
        std = torch.tensor([0.1, 0.1])
        network.set_normalization(mean, std)

        assert torch.equal(network.mean, mean)
        assert torch.equal(network.std, std)

        # Test that normalization is applied
        input_data = torch.ones(5, 2)
        output = network.forward(input_data)

        assert output.shape == torch.Size([5, 1])

    def test_network_weight_normalization(self):
        """Test network with weight normalization enabled."""
        network = FullyConnected(
            d_in=2, d_out=1, n_hidden_layers=2, n_neurons=10, weight_normalization=True
        )

        input_data = torch.randn(5, 2)
        output = network.forward(input_data)

        assert output.shape == torch.Size([5, 1])

    def test_network_gradient_flow(self):
        """Test that gradients flow through the network."""
        network = FullyConnected(d_in=2, d_out=1, n_hidden_layers=2, n_neurons=10)

        input_data = torch.randn(5, 2, requires_grad=True)
        output = network.forward(input_data)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_data.grad is not None
        assert not torch.isnan(input_data.grad).any()

        # Check that network parameters have gradients
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_xavier_trunc_normal_initialization(self):
        """Test custom Xavier truncated normal initialization."""
        tensor = torch.zeros(10, 10)
        xavier_trunc_normal_(tensor)

        # Check that values are within expected range
        assert tensor.std() > 0
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()

    def test_init_weights_trunc_normal(self):
        """Test weight initialization function."""
        layer = torch.nn.Linear(10, 5)
        init_weights_trunc_normal_(layer)

        # Check that weights are initialized
        assert layer.weight.std() > 0
        assert not torch.isnan(layer.weight).any()

    def test_network_different_sizes(self):
        """Test network with various input/output dimensions."""
        configs = [
            (1, 1, 1, 5),
            (3, 2, 4, 50),
            (5, 3, 2, 100),
        ]

        for d_in, d_out, n_layers, n_neurons in configs:
            network = FullyConnected(d_in, d_out, n_layers, n_neurons)
            input_data = torch.randn(10, d_in)
            output = network.forward(input_data)

            assert output.shape == torch.Size([10, d_out])
