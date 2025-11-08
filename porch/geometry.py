"""Geometry module for defining computational domains and sampling points."""

import torch
from .util import get_random_samples, get_regular_grid

from scipy.stats import qmc


class Geometry:
    """
    Represents a computational geometry/domain for Physics-Informed Neural Networks.

    This class defines a hyperrectangular domain and provides methods for sampling
    points within it using various strategies (random, Latin Hypercube, regular grid).

    Args:
        limits: A torch.Tensor of shape [d, 2] where d is the number of dimensions.
               Each row contains [lower_bound, upper_bound] for that dimension.

    Attributes:
        d: Number of dimensions of the geometry
        limits: Tensor containing the bounds for each dimension

    Example:
        >>> xlims = [0.0, 1.0]
        >>> ylims = [0.0, 2.0]
        >>> limits = torch.tensor([xlims, ylims])
        >>> geom = Geometry(limits)
        >>> samples = geom.get_random_samples(100)  # 100 random points in [0,1]x[0,2]
    """

    def __init__(self, limits: torch.Tensor) -> None:
        """
        Initialize a Geometry object.

        Args:
            limits: Tensor of shape [d, 2] defining domain bounds
        """
        self._d = len(limits)
        self._limits = limits

    @property
    def d(self) -> int:
        """Return the number of dimensions of the geometry."""
        return self._d

    @property
    def limits(self) -> torch.Tensor:
        """Return the bounds tensor of shape [d, 2]."""
        return self._limits

    def get_lhs_samples(self, n: int, device=None) -> torch.Tensor:
        """
        Generate samples using Latin Hypercube Sampling.

        Latin Hypercube Sampling (LHS) provides better space-filling properties
        than random sampling, ensuring more uniform coverage of the domain.

        Args:
            n: Number of samples to generate
            device: Target device for the tensor (e.g., torch.device('cuda'))

        Returns:
            Tensor of shape [n, d] containing the sampled points
        """
        lhs_sampler = qmc.LatinHypercube(self.d)
        samples = lhs_sampler.random(n)
        scaled_samples = qmc.scale(samples, self.limits[:, 0], self.limits[:, 1])
        return torch.as_tensor(scaled_samples, device=device, dtype=torch.float32)

    def get_random_samples(self, n: int, device=None) -> torch.Tensor:
        """
        Generate uniformly random samples within the geometry bounds.

        Args:
            n: Number of samples to generate
            device: Target device for the tensor

        Returns:
            Tensor of shape [n, d] containing the sampled points
        """
        return get_random_samples(self.limits, n, device)

    def get_regular_grid_iso(self, n: int, device=None) -> torch.Tensor:
        """
        Generate an isotropic regular grid (same resolution in all dimensions).

        Args:
            n: Number of points per dimension
            device: Target device for the tensor

        Returns:
            Tensor of shape [n^d, d] containing grid points
        """
        return get_regular_grid((n,) * self._d, self.limits, device)

    def get_regular_grid_aniso(self, n: tuple, device=None) -> torch.Tensor:
        """
        Generate an anisotropic regular grid (different resolution per dimension).

        Args:
            n: Tuple of length d specifying points per dimension
            device: Target device for the tensor

        Returns:
            Tensor of shape [prod(n), d] containing grid points
        """
        return get_regular_grid(n, self.limits, device)
