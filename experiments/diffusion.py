from __future__ import annotations

from matplotlib.figure import Figure
from torch.types import Number
from porch.boundary_conditions import BoundaryCondition, DirichletBC
from porch.training import Trainer
from porch.dataset import NamedTensorDataset
import logging

from collections.abc import Sequence


import torch
from porch.config import PorchConfig
from porch.geometry import Geometry
from porch.model import BaseModel
from porch.network import FullyConnected
from porch.util import gradient

import matplotlib.pyplot as plt
import numpy as np

try:
    from torch import hstack, vstack
except ImportError:
    from porch.util import hstack, vstack


F = 0.5
E = 0.5
k = 0.5
D = 0.4
D_inv = 1.0 / D
L = 2.0 * torch.pi
n = 2.0
T = 5.0

# reference http://personal.ph.surrey.ac.uk/~phs1rs/teaching/l3_pdes.pdf


# Analytical Solution of the Diffusion PDE --> d^2/dx^2 (P) = 1/D * d/dt (P)
def P(x, t):
    return (F * torch.cos(k * x) + E * torch.sin(k * x)) * torch.exp(-(k**2) * D * t)


# def P(x, t):
#     return (F * torch.sin(k * x)) * torch.exp(-(k**2) * D * t)


# def P(t, x):
#     return torch.sin(2 * k**2 * F * t - x * k) * torch.exp(-k * x)


# def P(x, t):
#     return torch.sin(2 * k * x) * torch.exp(-(k**2) * D * t)


# deepxde version
def P(x, t):
    return torch.exp(-(n**2 * torch.pi**2 * D * t) / (L**2)) * torch.sin(
        n * torch.pi * x / L
    )


class DiffusionModel(BaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        boundary_conditions: "Sequence[BoundaryCondition]",
    ) -> None:
        super().__init__(network, geometry, config, boundary_conditions)

    def boundary_loss(self, loss_name) -> torch.Tensor:
        """u(x=lb,t) = u(x=ub,t) = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        return torch.pow(prediction - labels, 2)

    def interior_loss(self, loss_name: str) -> torch.Tensor:
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)

        u_x = grad_u[..., 0]
        u_t = grad_u[..., 1]

        grad_u_x = gradient(u_x, data_in)

        u_xx = grad_u_x[..., 0]

        f = D * u_xx - u_t

        return torch.pow(f - labels, 2)

    def setup_losses(self) -> None:
        self.losses = {"boundary": self.boundary_loss, "interior": self.interior_loss}

    def setup_data(self, n_boundary: int, n_interior: int) -> None:
        # spread n_boudary evenly over all boundaries (including initial condition)
        n_boundary = n_boundary // (len(self.boundary_conditions) + 1)
        bc_tensors = []
        logging.info("Generating BC data...")
        for bc in self.boundary_conditions:
            bc_data = bc.get_samples(n_boundary, device=self.config.device)
            bc_tensors.append(bc_data)
        boundary_data = torch.cat(bc_tensors)

        logging.info("Generating interior data...")
        interior_data = self.geometry.get_random_samples(
            n_interior, device=self.config.device
        )
        interior_labels = torch.zeros(
            [interior_data.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        interior_data = hstack([interior_data, interior_labels])

        complete_dataset = NamedTensorDataset(
            {"boundary": boundary_data, "interior": interior_data}
        )

        self.set_dataset(complete_dataset)

    def setup_validation_data(self, n_validation: int) -> None:
        x_linspace = torch.linspace(
            float(self.geometry.limits[0, 0]),
            float(self.geometry.limits[0, 1]),
            n_validation,
        )
        t_linspace = torch.linspace(
            float(self.geometry.limits[1, 0]),
            float(self.geometry.limits[1, 1]),
            n_validation,
        )
        xx, tt = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        z = P(xx, tt)

        val_X = hstack([xx.flatten().unsqueeze(-1), tt.flatten().unsqueeze(-1)])
        val_u = torch.as_tensor(z.flatten().unsqueeze(-1), dtype=torch.float32)

        self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

    def plot_dataset(self) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])
        for name in self.get_data_names():
            data_in = self.get_input(name).cpu().numpy()
            axs.scatter(data_in[:, 1], data_in[:, 0], label=name, alpha=0.5)

        axs.set_xlabel("t")
        axs.set_ylabel("x")
        axs.legend()
        plt.savefig("plots/dataset_diffusion.png")

    def plot_validation(self, writer, iteration) -> Figure:
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (200, 200)
        axs: list[plt.Axes]
        fig, axs = plt.subplots(2, 1, figsize=[12, 6], sharex=True)
        self.network.eval()
        prediction = self.network.forward(validation_in)
        self.network.train()

        im_data = prediction.detach().cpu().numpy()
        im_data_gt = validation_labels.detach().cpu().numpy()
        im_data = im_data.reshape(domain_shape)
        im_data_gt = im_data_gt.reshape(domain_shape)

        im1 = axs[0].imshow(
            im_data,
            interpolation="nearest",
            extent=[0.0, 10.0, 0.0, 2.0 * np.pi],
            origin="lower",
            aspect="auto",
        )
        im2 = axs[1].imshow(
            im_data_gt,
            interpolation="nearest",
            extent=[0.0, 10.0, 0.0, 2.0 * np.pi],
            origin="lower",
            aspect="auto",
        )
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        axs[1].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$x$")
        return fig


def main(n_epochs=20000, model_dir=".") -> Number:
    num_layers = 4
    num_neurons = 20
    weight_norm = False
    n_boundary = 200
    n_interior = 3000
    n_validation = 200

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device, lr=0.001, epochs=n_epochs, lra=True)

    xlims = (0.0, L)
    tlims = (0.0, T)

    # 2D in (x,t) -> u 1D out
    network = FullyConnected(2, 1, num_layers, num_neurons, weight_norm)
    network.to(device=device)

    geom = Geometry(torch.tensor([xlims, tlims]))

    def ic_func(t_in):
        x_in_space = t_in[:, 0]
        t_in_space = t_in[:, 1]
        z_in = torch.unsqueeze(P(x_in_space, t_in_space), 1)
        return z_in

    ic_axis_definition = torch.Tensor([False, True])
    ic = DirichletBC("initial_bc", geom, ic_axis_definition, tlims[0], ic_func)

    bc_axis_definition = torch.Tensor([True, False])

    bc_bottom = DirichletBC(
        "bc_bottom",
        geom,
        bc_axis_definition,
        xlims[0],
        BoundaryCondition.zero_bc_fn,
        False,
    )
    bc_top = DirichletBC(
        "bc_top",
        geom,
        bc_axis_definition,
        xlims[1],
        BoundaryCondition.zero_bc_fn,
        False,
    )

    # bc_bottom = DirichletBC(
    #     "bc_bottom",
    #     geom,
    #     bc_axis_definition,
    #     xlims[0],
    #     ic_func,
    # )
    # bc_top = DirichletBC(
    #     "bc_top",
    #     geom,
    #     bc_axis_definition,
    #     xlims[1],
    #     ic_func,
    # )

    # boundary_conditions = [ic, bc_bottom, bc_top]
    boundary_conditions = [ic]

    model = DiffusionModel(network, geom, config, boundary_conditions)

    model.setup_data(n_boundary, n_interior)
    model.setup_validation_data(n_validation)
    model.plot_dataset()

    config.lr = 0.0004

    trainer = Trainer(model, config, model_dir)

    val_err = trainer.train()

    fig = model.plot_validation(None, None)
    fig.savefig("plots/validation_diffusion.pdf")


    return val_err


if __name__ == "__main__":
    main()
