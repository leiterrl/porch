from porch.boundary_conditions import BoundaryCondition

from porch.dataset import NamedTensorDataset
from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
import logging


# logging.basicConfig(level=logging.DEBUG)

# logger = logging.getLogger(__name__)
# logger.setLevel(5)

import torch
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


class WaveEquationBaseModel(BaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        wave_speed: float,
        boundary_conditions: "list[BoundaryCondition]",
        noise: float = 0.0,
        nointerior: bool = False,
    ):
        self.nointerior = nointerior
        super().__init__(network, geometry, config, boundary_conditions)
        self.wave_speed = wave_speed
        self.noise = noise
        self.data = DataWaveEquationZero()

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
        u_x = grad_u[..., 1]

        # gradMagnitude = torch.mean(
        #     torch.pow(torch.add(torch.abs(u_x), torch.abs(u_t)), 2)
        # )
        # self.loss_weights[loss_name] = 1.0 / gradMagnitude

        return torch.pow(u_t - labels, 2)

    def interior_loss(self, loss_name):
        """u_tt - \mu² * u_xx = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.tensor([], device=self.device)
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
        f = u_tt - self.wave_speed ** 2 * u_xx

        # gradMagnitude = torch.mean(
        #     torch.pow(torch.add(torch.abs(u_x), torch.abs(u_t)), 2)
        # )
        # self.loss_weights[loss_name] = 1.0 / gradMagnitude

        return torch.pow(f - labels, 2)

    def rom_loss(self, loss_name):
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        mse_cutoff_squared = self.noise ** 2
        mse_diff_squared = torch.pow(prediction - labels, 2)
        mse_diff_squared[mse_diff_squared < mse_cutoff_squared] = 0.0

        return mse_diff_squared

    def setup_losses(self):
        self.losses = {
            "boundary": self.boundary_loss,
            "rom": self.rom_loss,
        }
        if not self.nointerior:
            self.losses["interior"] = self.interior_loss

    def setup_data(self, n_boundary: int, n_interior: int, n_rom: int):
        bc_tensors = []
        logging.info("Generating BC data...")
        for bc in self.boundary_conditions:
            bc_data = bc.get_samples(n_boundary, device=self.config.device)
            bc_tensors.append(bc_data)
        boundary_data = torch.cat(bc_tensors)

        logging.info("Generating interior data...")
        if not self.nointerior:
            interior_data = self.geometry.get_random_samples(
                n_interior, device=self.config.device
            )
            interior_labels = torch.zeros(
                [interior_data.shape[0], 1],
                device=self.config.device,
                dtype=torch.float32,
            )
            interior_data = torch.hstack([interior_data, interior_labels])

        # Rom Data

        X = self.data.get_input()
        u = self.data.get_data_rom(self.wave_speed)

        # decrease dataset size
        rand_rows = torch.randperm(X.shape[0])[:n_rom]
        X = X[rand_rows, :]
        u = u[rand_rows]

        if self.noise > 0.0:
            noise_scale = (u.max() - u.min()) * self.noise
            noise_tensor = torch.rand_like(u) * noise_scale
            u += noise_tensor

        rom_data = torch.hstack([X, u]).to(device=self.config.device)

        dataset_dict = {"boundary": boundary_data, "rom": rom_data}
        if not self.nointerior:
            dataset_dict["interior"] = interior_data
        complete_dataset = NamedTensorDataset(dataset_dict)

        self.set_dataset(complete_dataset)

    def setup_validation_data(self) -> None:
        X = self.data.get_input()
        val_X = X.detach().clone()
        val_u = self.data.get_data_fom(self.wave_speed)
        self.validation_data = torch.hstack([val_X, val_u]).to(
            device=self.config.device
        )

    def plot_validation(self):
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (-1, self.data.fom.num_intervals + 1)
        # TODO simplify by flattening
        domain_extent = self.geometry.limits.flatten()
        sns.color_palette("mako", as_cmap=True)
        cmap = "mako_r"
        fig, axs = plt.subplots(2, 1, figsize=[12, 6], sharex=True)

        self.network.eval()
        prediction = self.network.forward(validation_in)
        self.network.train()

        im_data = prediction.detach().cpu().numpy()
        im_data_gt = validation_labels.detach().cpu().numpy()
        im_data = im_data.reshape(domain_shape)
        im_data_gt = im_data_gt.reshape(domain_shape)

        error = np.abs(np.flip(im_data_gt.T, axis=0) - np.flip(im_data.T, axis=0))

        im1 = axs[0].imshow(
            np.flip(im_data.T, axis=0),
            interpolation="nearest",
            extent=domain_extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )
        # axs[1].imshow(im_data_gt.detach().cpu().numpy())
        im2 = axs[1].imshow(
            error,
            interpolation="nearest",
            extent=domain_extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        cbar = fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        cbar = fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        nrom = self.get_labels("rom").shape[0]
        ninterior = self.get_labels("interior").shape[0]

        fig.suptitle(
            f"ROM-PINN Model with artificial noise: {self.noise} nrom: {nrom} nint: {ninterior}"
        )
        axs[0].set_title("Prediction")
        axs[1].set_title("Absolute Error")
        axs[1].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$x$")
        return fig

    def plot_dataset(self, name: str) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])
        for name in self.get_data_names():
            data_in = self.get_input(name).cpu().numpy()
            axs.scatter(data_in[:, 0], data_in[:, 1], label=name, alpha=0.5)

        axs.legend()
        plt.savefig(f"plots/dataset_{name}.png")

    def plot_boundary_data(self, name: str) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])

        data_in = self.get_input("boundary").cpu().numpy()
        labels = self.get_labels("boundary").cpu().numpy()

        axs.scatter(data_in[:, 0], data_in[:, 1], c=labels, label="boundary", alpha=0.5)

        axs.legend()
        plt.savefig(f"plots/boundary_{name}.png")
