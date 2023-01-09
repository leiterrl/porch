import torch
from .util import get_random_samples, get_regular_grid

from scipy.stats import qmc

class Geometry:
    def __init__(self, limits: torch.Tensor) -> None:
        self._d = len(limits)
        self._limits = limits

    @property
    def d(self) -> int:
        return self._d

    @property
    def limits(self) -> torch.Tensor:
        return self._limits

    def get_lhs_samples(self, n: int, device=None) -> torch.Tensor:
        lhs_sampler = qmc.LatinHypercube(self.d)
        samples = lhs_sampler.random(n)
        scaled_samples = qmc.scale(samples, self.limits[:,0], self.limits[:,1]) 
        return torch.as_tensor(scaled_samples, device=device, dtype=torch.float32)

    def get_random_samples(self, n: int, device=None) -> torch.Tensor:
        return get_random_samples(self.limits, n, device)

    def get_regular_grid_iso(self, n: int, device=None) -> torch.Tensor:
        return get_regular_grid((n,) * self._d, self.limits, device)

    def get_regular_grid_aniso(self, n: tuple, device=None) -> torch.Tensor:
        return get_regular_grid(n, self.limits, device)
