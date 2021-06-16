from __future__ import annotations
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry import Geometry

from abc import ABC, abstractmethod
from .util import get_random_samples, get_regular_grid
from torch import Tensor
from torch import hstack, zeros


class BoundaryCondition(ABC):
    def __init__(self, name: str, geom: Geometry, random: bool = True) -> None:
        """
        Args:
            name: gives the boundary condition a unique name
            geom: geometry object used for sampling
            input_mask: A dx1 Tensor object with NaN values marking variable dimensions
            Must have d-1 NaN values
        """
        self.name = name
        self.geometry = geom
        self.random = random

    @abstractmethod
    def get_samples(self, n_samples: int) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def zero_bc_fn(t_in: Tensor) -> Tensor:
        return zeros([t_in.shape[0], 1])


class DirichletBC(BoundaryCondition):
    def __init__(
        self,
        name: str,
        geom: Geometry,
        axis_definition: Tensor,
        constant_input: float,
        eval_fn: Callable,
        random: bool = True,
    ) -> None:
        """Construct a Dirichlet Boundary Condition object

        Args:
            name: gives the boundary condition a unique name
            geom: geometry object used for sampling
            axis_definition: A boolean valued Tensor of shape [d] with speficially one element == True defining the dimension along which the boundary is placed.
            constant_input: Value of input along constant boundary dimension (e.g: u(x=1.0,t) -> constant_input would be set to 1.0 and axis_definiton = [True, False])
            eval_fn: Function handle ( eval_fun(t_in: Tensor) -> Tensor ) for generating output values on boundary
        """
        super().__init__(name, geom, random)
        self.constant_input = constant_input
        self.eval_fn = eval_fn
        self.set_axis(axis_definition)

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
        relevant_limits = self.geometry.limits.detach().clone()
        relevant_limits[self.constant_idx, :] = self.constant_input

        if self.random:
            input = get_random_samples(relevant_limits, n_samples, device)
        else:
            input = get_regular_grid(
                (n_samples,) * self.geometry.d, relevant_limits, device
            )

        labels = self.eval_fn(input).to(device=device)

        return hstack([input, labels])


class DiscreteBC(BoundaryCondition):
    def __init__(self, name: str, geom: Geometry, data: Tensor) -> None:
        """Construct a Discrite Boundary Condition object

        Args:
            name: gives the boundary condition a unique name
            geom: geometry object used for sampling
            data: Tensor which contains discrete in- and output values
        """
        super().__init__(name, geom)
        if data.shape[1] <= geom.d:
            raise RuntimeError("Dataset must have at least one output dimension.")
        self.data = data

    def get_samples(self, n_samples: int, device=None) -> Tensor:
        return self.data.to(device=device)
