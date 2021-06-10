from __future__ import annotations
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry import Geometry

from abc import ABC, abstractmethod
from .util import get_random_samples
from torch import Tensor
from abc import ABC
from torch import hstack


class BoundaryCondition(ABC):
    def __init__(self, name: str, geom: Geometry) -> None:
        """
        Args:
            name: gives the boundary condition a unique name
            geom: geometry object used for sampling
            input_mask: A dx1 Tensor object with NaN values marking variable dimensions
            Must have d-1 NaN values
        """
        self.name = name
        self.geometry = geom

    @abstractmethod
    def get_samples(self, n_samples: int) -> Tensor:
        raise NotImplementedError


class DirichletBC(BoundaryCondition):
    def __init__(
        self, name: str, geom: Geometry, constant_input: float, eval_fn: Callable
    ) -> None:
        """Construct a Dirichlet Boundary Condition object

        Args:
            name: gives the boundary condition a unique name
            geom: geometry object used for sampling
        """
        super().__init__(name, geom)
        self.constant_input = constant_input
        self.eval_fn = eval_fn

    def set_axis(self, axis: Tensor):
        r"""Sets axis along which samples for this BC should be drawn

        Always has to be self.geom.d - 1
        """
        if len(axis) != self.geometry.d:
            raise RuntimeError("Boundary conditions don't match geometry dimensions.")
        self.axis = axis == False
        self.constant_idx = axis == True

    def get_samples(self, n_samples: int, device=None) -> Tensor:
        # TODO: implement different sampling techniques
        relevant_limits = self.geometry.limits
        relevant_limits[self.constant_idx, :] = self.constant_input

        input = get_random_samples(relevant_limits, n_samples, device)
        labels = self.eval_fn(input).to(device=device)

        return hstack([input, labels])


class DiscreteBC(BoundaryCondition):
    def __init__(self, name: str, geom: Geometry, data: Tensor) -> None:
        """Construct a Dirichlet Boundary Condition object

        Args:
            name: gives the boundary condition a unique name
            geom: geometry object used for sampling
        """
        super().__init__(name, geom)
        if data.shape[1] <= geom.d:
            raise RuntimeError("Dataset must have at least one output dimension.")
        self.data = data

    def get_samples(self, n_samples: int, device=None) -> Tensor:
        return self.data.to(device=device)
