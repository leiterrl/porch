from inspect import BoundArguments
from torch._C import device
from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer
from porch.dataset import NamedTensorDataset
from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
import logging

import torch
from torch.utils.data import TensorDataset
from porch.config import PorchConfig
from porch.geometry import Geometry
from porch.model import BaseModel
from porch.network import FullyConnected
from porch.util import gradient

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)
import numpy as np


class WaveEquationROM(BaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        wave_speed: float,
        boundary_conditions: "list[BoundaryCondition]",
    ):
        super().__init__(network, geometry, config, boundary_conditions)
        self.wave_speed = wave_speed

    # TODO: only spatial boundary
    def boundary_loss(self, loss_name):
        """u(x=lb,t) = u(x=ub,t) = 0"""
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        # grad_u = gradient(prediction, data_in)
        # u_x = grad_u[..., 1]
        # u_t = grad_u[..., 0]

        # # TODO: fix to meet problem setup
        # gradMagnitude = torch.mean(
        #     torch.pow(torch.add(torch.abs(u_x), torch.abs(u_t)), 2)
        # )
        # self.loss_weights[loss_name] = 1.0 / gradMagnitude

        return torch.pow(prediction - labels, 2)

    # TODO: complete ic (u and u_t)
    def ic_loss(self, loss_name):
        """u_t(x,t=0) = 0"""
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)
        u_t = grad_u[..., 0]

        # gradMagnitude = torch.mean(
        #     torch.pow(torch.add(torch.abs(u_x), torch.abs(u_t)), 2)
        # )
        # self.loss_weights[loss_name] = 1.0 / gradMagnitude

        return torch.pow(u_t - labels, 2)

    def interior_loss(self, loss_name):
        """u_tt - \mu² * u_xx = 0"""
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)
        u_x = grad_u[..., 1]
        u_t = grad_u[..., 0]

        grad_u_x = gradient(u_x, data_in)
        u_xx = grad_u_x[..., 1]

        grad_u_t = gradient(u_t, data_in)
        u_tt = grad_u_t[..., 0]

        # TODO: move wave_speed to tensor
        # f = (
        #     u_tt
        #     - torch.pow(
        #         torch.as_tensor(
        #             self.wave_speed, device=self.config.device, dtype=torch.float32
        #         ),
        #         2,
        #     )
        #     * u_xx
        # )
        f = u_tt - (self.wave_speed ** 2) * u_xx

        # gradMagnitude = torch.mean(
        #     torch.pow(torch.add(torch.abs(u_x), torch.abs(u_t)), 2)
        # )
        # self.loss_weights[loss_name] = 1.0 / gradMagnitude

        return torch.pow(f - labels, 2)

    def rom_loss(self, loss_name):
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)
        return torch.pow(prediction - labels, 2)

    def setup_losses(self):
        self.losses = {
            "boundary": self.boundary_loss,
            "interior": self.interior_loss,
            "rom": self.rom_loss,
        }

    def setup_data(self, n_boundary: int, n_interior: int, n_rom: int):
        bc_tensors = []
        logging.info("Generating BC data...")
        for bc in self.boundary_conditions:
            bc_data = bc.get_samples(n_boundary, device=self.config.device)
            bc_tensors.append(bc_data)
        boundary_data = torch.cat(bc_tensors)

        logging.info("Generating interior data...")
        # interior_data = self.geometry.get_random_samples(
        #     n_interior, device=self.config.device
        # )
        interior_data = self.geometry.get_regular_grid_iso(
            int(np.abs(np.sqrt(n_interior))),
            device=self.config.device,
        )
        interior_labels = torch.zeros(
            [interior_data.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        interior_data = torch.hstack([interior_data, interior_labels])

        # Rom Data
        data = DataWaveEquationZero()
        X = data.get_input()
        u = data.get_data_rom(self.wave_speed)

        # decrease dataset size
        rand_rows = torch.randperm(X.shape[0])[:n_rom]
        X = X[rand_rows, :]
        u = u[rand_rows]

        rom_data = torch.hstack([X, u]).to(device=self.config.device)

        complete_dataset = NamedTensorDataset(
            {"boundary": boundary_data, "interior": interior_data, "rom": rom_data}
        )

        self.set_dataset(complete_dataset)

    def setup_validation_data(self) -> None:
        data = DataWaveEquationZero()
        X = data.get_input()
        val_X = X.detach().clone()
        val_u = data.get_data_fom(self.wave_speed)
        # extend val region
        val_X_extended = data.get_input_extended()
        val_u_extended = val_u.detach().clone().flip(0)

        val_X = torch.cat([val_X, val_X_extended])
        val_u = torch.cat([val_u, val_u_extended])

        self.validation_data = torch.hstack([val_X, val_u]).to(
            device=self.config.device
        )

    def plot_validation(self):
        cmap = "mako_r"
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (1000, 1001)
        fig, axs = plt.subplots(2, 1, figsize=[12, 6], sharex=True)
        self.network.eval()
        prediction = self.network.forward(validation_in)
        self.network.train()

        im_data = prediction.detach().cpu().numpy()
        im_data_gt = validation_labels.detach().cpu().numpy()
        im_data = im_data.reshape(domain_shape)
        im_data_gt = im_data_gt.reshape(domain_shape)

        im1 = axs[0].imshow(
            np.flip(im_data.T, axis=0),
            interpolation="nearest",
            extent=[0.0, 4.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )
        # axs[1].imshow(im_data_gt.detach().cpu().numpy())
        im2 = axs[1].imshow(
            np.abs(np.flip(im_data_gt.T, axis=0) - np.flip(im_data.T, axis=0)),
            interpolation="nearest",
            extent=[0.0, 4.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        cbar = fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        cbar = fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        axs[1].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$x$")
        return fig

    def plot_dataset(self) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])
        for name in self.get_data_names():
            data_in = self.get_input(name).cpu().numpy()
            axs.scatter(data_in[:, 0], data_in[:, 1], label=name, alpha=0.5)

        axs.legend()
        plt.savefig("plots/dataset_extrapolation.png")


def main():
    model_dir = "/import/sgs.local/scratch/leiterrl/1d_wave_eq_rom_extrapolation"
    num_layers = 4
    num_neurons = 50
    weight_norm = False
    wave_speed = 1.0
    n_boundary = 500
    n_interior = 2025
    n_rom = 500

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device)
    config.epochs = 40000

    xlims = (-1.0, 1.0)
    tlims = (0.0, 4.0)

    # 2D in (x,t) -> u 1D out
    network = FullyConnected(2, 1, num_layers, num_neurons, weight_norm)
    network.to(device=device)

    geom = Geometry(torch.tensor([tlims, xlims]))

    data = DataWaveEquationZero()
    X_init, u_init = data.get_initial_cond(wave_speed)
    initial_data = torch.hstack([X_init, u_init])
    ic = DiscreteBC("initial_bc", geom, initial_data)

    bc_axis = torch.Tensor([False, True])
    bc_upper = DirichletBC(
        "bc_upper", geom, bc_axis, xlims[1], BoundaryCondition.zero_bc_fn, False
    )
    bc_lower = DirichletBC(
        "bc_lower", geom, bc_axis, xlims[0], BoundaryCondition.zero_bc_fn, False
    )

    boundary_conditions = [ic, bc_upper, bc_lower]

    model = WaveEquationROM(network, geom, config, wave_speed, boundary_conditions)

    # model.loss_weights["interior"] = 0.001
    model.loss_weights["boundary"] = 10.0
    model.loss_weights["rom"] = 10.0
    model.setup_data(n_boundary, n_interior, n_rom)
    model.setup_validation_data()

    model.plot_dataset()

    trainer = Trainer(model, config, model_dir)

    trainer.train()


if __name__ == "__main__":
    main()
