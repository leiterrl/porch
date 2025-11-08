"""Quick verification that core functionality works with updated dependencies."""
import torch
import numpy as np

# Test basic imports
try:
    from porch.geometry import Geometry
    from porch.network import FullyConnected
    from porch.config import PorchConfig
    from porch.boundary_conditions import DirichletBC, BoundaryCondition
    print("✓ All core imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test basic functionality
try:
    # Test Geometry
    limits = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
    geom = Geometry(limits)
    samples = geom.get_random_samples(10)
    assert samples.shape == torch.Size([10, 2])
    print("✓ Geometry works")

    # Test Network
    network = FullyConnected(d_in=2, d_out=1, n_hidden_layers=2, n_neurons=10)
    output = network(torch.randn(5, 2))
    assert output.shape == torch.Size([5, 1])
    print("✓ Network works")

    # Test Config
    config = PorchConfig()
    assert config.lr > 0
    assert config.epochs > 0
    print("✓ Config works")

    # Test Boundary Condition
    bc = DirichletBC(
        "test_bc",
        geom,
        torch.tensor([True, False]),
        0.0,
        BoundaryCondition.zero_bc_fn,
        random=True
    )
    bc_samples = bc.get_samples(10)
    assert bc_samples.shape == torch.Size([10, 3])
    print("✓ Boundary conditions work")

    print("\n✓✓✓ All tests passed! ✓✓✓")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
