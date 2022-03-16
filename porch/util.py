import torch
from enum import Enum
import numpy as np
import argparse

# from typing import Optional, List
from torch.jit.annotations import Optional, List
from torch import Tensor

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

# from tensorboardX.utils.tensorboard.summary import hparams
# from tensorboardX import SummaryWriter


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


# TODO: handle non-scalar case
# def gradient(y, x):
#     grad_outputs = torch.ones_like(y)
#     grad = torch.autograd.grad(
#         [y], [x], grad_outputs=[grad_outputs], create_graph=True, retain_graph=True
#     )[0]
#     return grad


# # TODO: handle non-scalar case
# @torch.jit.script
# def gradient(y, x):
#     # grad_outputs = [torch.ones_like(y)]
#     grad_outputs = torch.jit.annotate(Optional[Tensor], torch.ones_like(y))
#     # grad_outputs = torch.jit.annotate(Optional[List[Optional[Tensor]]], [torch.ones_like(y)]) -> "Expected a List type hint but instead got Optional[List[Optional[Tensor]]]"
#     grad = torch.autograd.grad(
#         [y], [x], [grad_outputs], create_graph=True, retain_graph=True
#     )[0]
#     return grad


@torch.jit.script
def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
    # grad_outputs = [torch.ones_like(y)]
    # grad_outputs = torch.jit.annotate(Optional[Tensor], [torch.ones_like(y)])
    grad = torch.autograd.grad(
        [
            y,
        ],
        [x],
        grad_outputs=grad_outputs,
        create_graph=True,
    )

    assert grad is not None

    # optional type refinement using an if statement
    grad = grad[0]

    assert grad is not None

    return grad  # grad can be None here, so it is Optional[torch.Tensor]


def hstack(tensor_tuple):
    return torch.cat(tensor_tuple, dim=1)


def vstack(tensor_tuple):
    return torch.cat(tensor_tuple, dim=0)


def fig_to_rgb_array(fig):
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_array


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


# TODO: use torch functions
def get_random_samples_circle(
    center: tuple, r: float, n: int, device=None
) -> torch.Tensor:
    r_array = r * np.sqrt(np.random.uniform(size=n))
    theta_array = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    x = center[0] + r_array * np.cos(theta_array)
    y = center[1] + r_array * np.sin(theta_array)
    samples = np.vstack([x, y]).T
    return torch.as_tensor(samples, device=device, dtype=torch.float32)


# TODO: use torch functions
def get_random_samples(bounds: torch.Tensor, n: int, device=None) -> torch.Tensor:
    low = [bound[0] for bound in bounds]
    high = [bound[1] for bound in bounds]
    return torch.as_tensor(
        np.random.uniform(low=low, high=high, size=(n, len(bounds))),
        device=device,
        dtype=torch.float32,
    )


def relative_l2_error(pred, truth):
    """Relative l2 error as suggested in "Supplementary Material for Hidden
    Fluid dynamics".

        pred:   Predictions
        truth:  Reference values

        returns
            Relative l2 error. If all truth values are zero, the absolute l2
            error is returned.
    """
    if len(pred) > 0 and len(truth) > 0:
        # nominator = torch.mean(torch.square(pred - truth))
        # denominator = torch.mean(torch.square(truth - torch.mean(truth)))
        nominator = torch.linalg.norm(truth - pred)
        denominator = torch.linalg.norm(truth)
        if denominator > 0.0:
            return nominator / denominator
        else:
            print(
                "Warning: Cannot compute relative error since exact value is"
                " constant, using absolute MSE instead"
            )
            return nominator
    else:
        return torch.tensor(0.0, device=pred.device)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ninterior",
        type=int,
        default=10000,
        help="Set number of interior collocation points",
    )
    parser.add_argument(
        "--nboundary", type=int, default=1000, help="Set number of boundary data points"
    )
    parser.add_argument(
        "--nrom", type=int, default=10000, help="Set number of rom data points"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Set number of epochs",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=0,
        help="Set batch size",
    )
    parser.add_argument(
        "--batchshuffle",
        action="store_true",
        help="Shuffle data in each batch",
    )
    parser.add_argument(
        "--batchcycle",
        action="store_true",
        help="Restart iteration if any dataset runs empty during batch creation",
    )
    parser.add_argument(
        "--lra",
        action="store_true",
        help="Use learning rate annealing",
    )
    parser.add_argument(
        "--dirichlet",
        action="store_true",
        help="Use learning rate annealing",
    )
    parser.add_argument(
        "--opt",
        action="store_true",
        help="Use optimal weighting",
    )
    parser.add_argument(
        "--determ",
        action="store_true",
        help="Use deterministic init",
    )
    parser.add_argument("--nbases", type=int, default=2, help="Set number of rom bases")
    parser.add_argument(
        "--lbfgs",
        action="store_true",
        help="Use learning rate annealing",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use heuristic approach",
    )

    args = parser.parse_args()

    return args
