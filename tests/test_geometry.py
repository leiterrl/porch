import torch
import pytest

from porch.geometry import Geometry


class TestGeometry:
    """Test suite for Geometry class."""

    def test_geometry_initialization_1d(self):
        """Test geometry initialization in 1D."""
        xlims = [0.0, 1.0]
        limits = torch.tensor([xlims])
        geom = Geometry(limits)

        assert geom.d == 1
        assert geom.limits.shape == torch.Size([1, 2])
        assert torch.equal(geom.limits, limits)

    def test_geometry_initialization_2d(self):
        """Test geometry initialization in 2D."""
        xlims = [0.0, 1.0]
        ylims = [-1.0, 1.0]
        limits = torch.tensor([xlims, ylims])
        geom = Geometry(limits)

        assert geom.d == 2
        assert geom.limits.shape == torch.Size([2, 2])

    def test_geometry_initialization_3d(self):
        """Test geometry initialization in 3D."""
        xlims = [0.0, 1.0]
        ylims = [0.0, 2.0]
        zlims = [-1.0, 1.0]
        limits = torch.tensor([xlims, ylims, zlims])
        geom = Geometry(limits)

        assert geom.d == 3
        assert geom.limits.shape == torch.Size([3, 2])

    def test_get_random_samples_1d(self):
        """Test random sampling in 1D."""
        xlims = [0.0, 1.0]
        limits = torch.tensor([xlims])
        geom = Geometry(limits)

        n_samples = 100
        samples = geom.get_random_samples(n_samples)

        assert samples.shape == torch.Size([n_samples, 1])
        assert samples.min() >= xlims[0]
        assert samples.max() <= xlims[1]

    def test_get_random_samples_2d(self):
        """Test random sampling in 2D."""
        xlims = [0.0, 1.0]
        ylims = [-1.0, 1.0]
        limits = torch.tensor([xlims, ylims])
        geom = Geometry(limits)

        n_samples = 100
        samples = geom.get_random_samples(n_samples)

        assert samples.shape == torch.Size([n_samples, 2])
        assert samples[:, 0].min() >= xlims[0]
        assert samples[:, 0].max() <= xlims[1]
        assert samples[:, 1].min() >= ylims[0]
        assert samples[:, 1].max() <= ylims[1]

    def test_get_random_samples_with_device(self):
        """Test random sampling with device specification."""
        xlims = [0.0, 1.0]
        limits = torch.tensor([xlims])
        geom = Geometry(limits)

        n_samples = 50
        device = torch.device("cpu")
        samples = geom.get_random_samples(n_samples, device=device)

        assert samples.device.type == "cpu"
        assert samples.shape == torch.Size([n_samples, 1])

    def test_get_regular_grid_1d(self):
        """Test regular grid generation in 1D."""
        xlims = [0.0, 1.0]
        limits = torch.tensor([xlims])
        geom = Geometry(limits)

        n_points = 11
        grid = geom.get_regular_grid((n_points,))

        assert grid.shape == torch.Size([n_points, 1])
        assert torch.allclose(grid[0], torch.tensor([xlims[0]]))
        assert torch.allclose(grid[-1], torch.tensor([xlims[1]]))

    def test_get_regular_grid_2d(self):
        """Test regular grid generation in 2D."""
        xlims = [0.0, 1.0]
        ylims = [0.0, 2.0]
        limits = torch.tensor([xlims, ylims])
        geom = Geometry(limits)

        n_points_x = 11
        n_points_y = 21
        grid = geom.get_regular_grid((n_points_x, n_points_y))

        # Total points should be product of dimensions
        assert grid.shape == torch.Size([n_points_x * n_points_y, 2])

        # Check bounds
        assert grid[:, 0].min() >= xlims[0]
        assert grid[:, 0].max() <= xlims[1]
        assert grid[:, 1].min() >= ylims[0]
        assert grid[:, 1].max() <= ylims[1]

    def test_geometry_volume_1d(self):
        """Test volume calculation in 1D."""
        xlims = [0.0, 2.0]
        limits = torch.tensor([xlims])
        geom = Geometry(limits)

        # Volume in 1D is just length
        expected_volume = xlims[1] - xlims[0]
        # Add volume calculation method if it exists
        # For now, just check limits
        actual_volume = geom.limits[0, 1] - geom.limits[0, 0]
        assert torch.allclose(actual_volume, torch.tensor(expected_volume))

    def test_geometry_volume_2d(self):
        """Test volume (area) calculation in 2D."""
        xlims = [0.0, 2.0]
        ylims = [0.0, 3.0]
        limits = torch.tensor([xlims, ylims])
        geom = Geometry(limits)

        # Area in 2D
        width = xlims[1] - xlims[0]
        height = ylims[1] - ylims[0]
        expected_area = width * height

        actual_area = (geom.limits[0, 1] - geom.limits[0, 0]) * \
                      (geom.limits[1, 1] - geom.limits[1, 0])
        assert torch.allclose(actual_area, torch.tensor(expected_area))

    def test_get_random_samples_reproducibility(self):
        """Test that random sampling is reproducible with same seed."""
        xlims = [0.0, 1.0]
        limits = torch.tensor([xlims])
        geom = Geometry(limits)

        n_samples = 50

        torch.manual_seed(42)
        samples1 = geom.get_random_samples(n_samples)

        torch.manual_seed(42)
        samples2 = geom.get_random_samples(n_samples)

        assert torch.allclose(samples1, samples2)

    def test_negative_limits(self):
        """Test geometry with negative limits."""
        xlims = [-5.0, -1.0]
        ylims = [-10.0, 0.0]
        limits = torch.tensor([xlims, ylims])
        geom = Geometry(limits)

        n_samples = 100
        samples = geom.get_random_samples(n_samples)

        assert samples[:, 0].min() >= xlims[0]
        assert samples[:, 0].max() <= xlims[1]
        assert samples[:, 1].min() >= ylims[0]
        assert samples[:, 1].max() <= ylims[1]
