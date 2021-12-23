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
k = 2
D = 0.05


# Analytical Solution of the Diffusion PDE --> d^2/dx^2 (P) = 1/D * d/dt (P)
def P(x, t):
    return (F * torch.cos(k * x) + E * torch.sin(k * x)) * torch.exp(-(k ** 2) * D * t)


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

    # TODO: complete ic (u and u_t)
    def ic_loss(self, loss_name) -> torch.Tensor:
        """u_t(x,t=0) = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        return torch.pow(prediction - labels, 2)

    @torch.jit.script
    def jit_loss(jit_in, jit_labels, jit_prediction, jit_gu, jit_gux):
        u_t = jit_gu[..., 1]
        u_xx = jit_gux[..., 0]

        D = 0.05
        D_inv = 1.0 / D

        f = u_xx - u_t * D_inv

        return torch.pow(f - jit_labels, 2)

    # # @torch.jit.script
    # def interior_loss(self, loss_name) -> torch.Tensor:

    #     data_in = self.get_input(loss_name)
    #     if len(data_in) == 0:
    #         return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
    #     labels = self.get_labels(loss_name)
    #     prediction = self.network.forward(data_in)

    #     grad_u = gradient(prediction, data_in)
    #     u_x = grad_u[..., 0]
    #     # u_t = grad_u[..., 1]

    #     grad_u_x = gradient(u_x, data_in)

    #     @torch.jit.script
    #     def jit_loss(jit_in, jit_labels, jit_prediction, jit_gu, git_gux):
    #         u_t = jit_gu[..., 1]
    #         u_xx = git_gux[..., 0]

    #         D = 0.05
    #         D_inv = 1.0 / D

    #         f = u_xx - u_t * D_inv

    #         return torch.pow(f - jit_labels, 2)

    #     return jit_loss(data_in, labels, prediction, grad_u, grad_u_x)

    # @torch.jit.script
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

        D = 0.05
        D_inv = 1.0 / D

        f = u_xx - u_t * D_inv

        return torch.pow(f - labels, 2)
        # return self.jit_loss(data_in, labels, prediction, grad_u, grad_u_x)

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
        xx, tt = torch.meshgrid(x_linspace, t_linspace)
        z = P(xx, tt)

        val_X = hstack([xx.flatten().unsqueeze(-1), tt.flatten().unsqueeze(-1)])
        val_u = torch.as_tensor(z.flatten().unsqueeze(-1), dtype=torch.float32)

        self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

    def plot_dataset(self) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])
        for name in self.get_data_names():
            data_in = self.get_input(name).cpu().numpy()
            axs.scatter(data_in[:, 0], data_in[:, 1], label=name, alpha=0.5)

        axs.legend()
        plt.savefig("plots/dataset_diffusion.png")

    def plot_validation(self, writer, iteration) -> Figure:
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (200, 200)
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
            extent=[0.0, 2.0 * np.pi, 0.0, 10.0],
            origin="lower",
            aspect="auto",
        )
        # axs[1].imshow(im_data_gt.detach().cpu().numpy())
        im2 = axs[1].imshow(
            np.abs(im_data_gt - im_data),
            interpolation="nearest",
            extent=[0.0, 2.0 * np.pi, 0.0, 10.0],
            origin="lower",
            aspect="auto",
            # vmin=0.0,
            # vmax=1.0,
        )
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        axs[1].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$x$")
        return fig


def main(
    n_epochs=10000, model_dir="/import/sgs.local/scratch/leiterrl/1d_diffusion"
) -> Number:
    num_layers = 4
    num_neurons = 20
    weight_norm = False
    n_boundary = 200
    n_interior = 200
    n_validation = 200

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device, lr=0.001, epochs=n_epochs, lra=True)

    xlims = (0.0, 2.0 * np.pi)
    tlims = (0.0, 10.0)

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

    boundary_conditions = [ic]

    model = DiffusionModel(network, geom, config, boundary_conditions)

    model.setup_data(n_boundary, n_interior)
    model.setup_validation_data(n_validation)
    model.plot_dataset()

    trainer = Trainer(model, config, model_dir)

    return trainer.train()


if __name__ == "__main__":
    main()
