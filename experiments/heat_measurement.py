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


class HeatModel(BaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        boundary_conditions: "Sequence[BoundaryCondition]",
    ) -> None:
        super().__init__(network, geometry, config, boundary_conditions)

    def exact_solution(self, x, y):
        return 0.0 * x + 0.0 * y

    def boundary_loss(self, loss_name) -> torch.Tensor:
        """u(x=lb,t) = u(x=ub,t) = 0"""
        data_in = self.get_input(loss_name)
        # TODO: is this needed?
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
        u_y = grad_u[..., 1]

        grad_u_x = gradient(u_x, data_in)
        grad_u_y = gradient(u_y, data_in)

        u_xx = grad_u_x[..., 0]
        u_yy = grad_u_y[..., 1]

        f = u_xx + u_yy

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
        y_linspace = torch.linspace(
            float(self.geometry.limits[1, 0]),
            float(self.geometry.limits[1, 1]),
            n_validation,
        )
        xx, yy = torch.meshgrid(x_linspace, y_linspace)
        z = self.exact_solution(xx, yy)

        val_X = hstack([xx.flatten().unsqueeze(-1), yy.flatten().unsqueeze(-1)])
        val_u = torch.as_tensor(z.flatten().unsqueeze(-1), dtype=torch.float32)

        self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

    def plot_dataset(self) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])
        for name in self.get_data_names():
            data_in = self.get_input(name).cpu().numpy()
            axs.scatter(data_in[:, 0], data_in[:, 1], label=name, alpha=0.5)

        axs.legend()
        plt.savefig("plots/dataset_heat.png")

    def plot_validation(self, writer, iteration) -> Figure:
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (200, 200)
        fig, axs = plt.subplots(2, 1, figsize=[12, 12], sharex=True)
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
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=0.1,
        )
        # axs[1].imshow(im_data_gt.detach().cpu().numpy())
        im2 = axs[1].imshow(
            np.abs(im_data_gt - im_data),
            interpolation="nearest",
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            # vmin=0.0,
            # vmax=0.1,
        )
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        axs[1].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$x$")
        return fig


def main(
    n_epochs=100, model_dir="/import/sgs.local/scratch/leiterrl/heat_measurement"
) -> Number:
    num_layers = 4
    num_neurons = 20
    weight_norm = True
    n_boundary = 200
    n_interior = 2000
    n_validation = 200

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(
        device=device,
        lr=0.001,
        epochs=n_epochs,
        lra=True,
        summary_freq=10,
        print_freq=10,
    )

    xlims = (-1.0, 1.0)
    ylims = (-1.0, 1.0)

    # 2D in (x,y) -> u 1D out
    network = FullyConnected(2, 1, num_layers, num_neurons, weight_norm)
    network.to(device=device)

    geom = Geometry(torch.tensor([xlims, ylims]))

    upper_bc = DirichletBC(
        "upper_boundary",
        geom,
        torch.tensor([True, False]),
        1.0,
        BoundaryCondition.zero_bc_fn,
        False,
    )
    lower_bc = DirichletBC(
        "lower_boundary",
        geom,
        torch.tensor([True, False]),
        -1.0,
        BoundaryCondition.zero_bc_fn,
        False,
    )
    left_bc = DirichletBC(
        "left_boundary",
        geom,
        torch.tensor([False, True]),
        -1.0,
        BoundaryCondition.zero_bc_fn,
        False,
    )
    right_bc = DirichletBC(
        "right_boundary",
        geom,
        torch.tensor([False, True]),
        1.0,
        BoundaryCondition.zero_bc_fn,
        False,
    )

    boundary_conditions = [upper_bc, lower_bc, left_bc, right_bc]

    model = HeatModel(network, geom, config, boundary_conditions)

    model.setup_data(n_boundary, n_interior)
    model.setup_validation_data(n_validation)
    model.plot_dataset()

    trainer = Trainer(model, config, model_dir)

    return trainer.train()


if __name__ == "__main__":
    main()
