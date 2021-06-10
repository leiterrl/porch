import torch
from .boundary_conditions import BoundaryCondition
from .util import get_random_samples, get_regular_grid


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

    def get_random_samples(self, n: int, device=None) -> torch.Tensor:
        return get_random_samples(self.limits, n, device)

    def get_regular_grid_iso(self, n: int, device=None) -> torch.Tensor:
        return get_regular_grid((n,) * self._d, self.limits, device)

    def get_regular_grid_aniso(self, n: tuple, device=None) -> torch.Tensor:
        return get_regular_grid(n, self.limits, device)
