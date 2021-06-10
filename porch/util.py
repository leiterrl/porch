import torch
from enum import Enum
import numpy as np


def gradient(y, x):
    create_graph = True
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=True
    )[0]
    return grad


class SamplingType(Enum):
    RANDOM_UNIFORM = 1
    GRID = 2
    SPARSE_GRID = 3


def get_regular_grid(N: tuple, bounds: torch.Tensor, device=None) -> torch.Tensor:
    """generate regular grid data

    Parameters
    ----------
    N : tuple
        [number of gridpoints per dimension]
    bounds : list
        [list tuples holding lower and upper bound values for each dimension]

    Returns
    -------
    [torch.Tensor]
        [a tensor (N[0]* ... * N[len(N)], len(N)) storing each cartesian coordinate combination in a sequence]
    """
    linspaces = []
    for d, n in enumerate(N):
        linspaces.append(torch.linspace(bounds[d, 0], bounds[d, 1], n))
        # linspaces.append(torch.linspace(-1.0, 1.0, n))

    regular_grid = torch.stack(torch.meshgrid(*linspaces), -1).reshape(-1, len(N))

    # return regular_grid.cuda()
    return regular_grid.to(device)


def get_random_samples(bounds: torch.Tensor, n: int, device=None) -> torch.Tensor:
    low = [bound[0] for bound in bounds]
    high = [bound[1] for bound in bounds]
    return torch.as_tensor(
        np.random.uniform(low=low, high=high, size=(n, len(bounds))),
        device=device,
        dtype=torch.float32,
    )
