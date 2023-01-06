import numpy as np
import logging
from porch.boundary_conditions import BoundaryCondition

from porch.dataset import NamedTensorDataset
from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero

try:
    from torch import hstack, vstack
except ImportError:
    from porch.util import hstack, vstack


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


class WaveEquationBaseModel(BaseModel):
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
        self.data = DataWaveEquationZero(self.config.n_bases)

    # TODO: only spatial boundary
    def boundary_loss(self, loss_name):
        """u(x=lb,t) = u(x=ub,t) = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        return torch.pow(prediction - labels, 2)

    # TODO: complete ic (u and u_t)
    def ic_loss(self, loss_name):
        """u_t(x,t=0) = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)
        u_t = grad_u[..., 0].unsqueeze(1)
        # u_x = grad_u[..., 1]

        # gradMagnitude = torch.mean(
        #     torch.pow(torch.add(torch.abs(u_x), torch.abs(u_t)), 2)
        # )
        # self.loss_weights[loss_name] = 1.0 / gradMagnitude

        return torch.pow(u_t - labels, 2)

    def interior_loss(self, loss_name):
        """u_tt - \\mu² * u_xx = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)
        u_x = grad_u[..., 1].unsqueeze(1)
        u_t = grad_u[..., 0].unsqueeze(1)

        grad_u_x = gradient(u_x, data_in)
        u_xx = grad_u_x[..., 1].unsqueeze(1)

        grad_u_t = gradient(u_t, data_in)
        u_tt = grad_u_t[..., 0].unsqueeze(1)

        f = u_tt - self.wave_speed**2 * u_xx

        return torch.pow(f - labels, 2)

    def rom_loss(self, loss_name):
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        mse_diff_squared = torch.pow(prediction - labels, 2)

        return mse_diff_squared

    def setup_losses(self):
        self.losses = {
            "boundary": self.boundary_loss,
            "ic_t": self.ic_loss,
            "interior": self.interior_loss,
            "rom": self.rom_loss,
        }

    def setup_data(self, n_boundary: int, n_interior: int, n_rom: int):
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
            [interior_data.shape[0], 1],
            device=self.config.device,
            dtype=torch.float32,
        )
        interior_data = hstack([interior_data, interior_labels])

        initial_input = self.data.get_input().to(device=self.config.device)[
            0 : self.data.fom.num_intervals + 1, :
        ]
        # downsample, TODO: this should be done in a more unified way, i guess
        len_data = len(initial_input)
        if n_boundary < len_data:
            sampling_points = np.linspace(0, len_data - 1, n_boundary, dtype=int)
            initial_input = initial_input[sampling_points]
        elif n_boundary == len_data:
            pass
        else:
            raise ValueError(
                "Cannot generate n_sample={} from data of len: {}".format(
                    n_boundary, len_data
                )
            )
        ic_t_labels = torch.zeros(
            [initial_input.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        ic_t_data = hstack([initial_input, ic_t_labels])

        # Rom Data

        # X = self.data.get_input()
        X, u = self.data.get_data_rom(self.wave_speed, self.config.subsample_rom)
        # u = self.data.get_data_fom(self.wave_speed)

        # decrease dataset size
        rand_rows = torch.randperm(X.shape[0])[:n_rom]
        X = X[rand_rows, :]
        u = u[rand_rows]

        rom_data = hstack([X, u]).to(device=self.config.device)

        dataset_dict = {
            "interior": interior_data,
            "boundary": boundary_data,
            "rom": rom_data,
            "ic_t": ic_t_data,
        }
        complete_dataset = NamedTensorDataset(dataset_dict)

        self.set_dataset(complete_dataset)

    def setup_validation_data(self) -> None:
        val_X, val_u = self.data.get_explicit_solution_data(self.wave_speed, True)
        self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

    def plot_validation(self):
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (-1, (self.data.fom.num_intervals + 1) // 6)
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
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        nrom = self.get_labels("rom").shape[0]
        ninterior = self.get_labels("interior").shape[0]

        fig.suptitle(f"ROM-PINN nrom: {nrom} nint: {ninterior}")
        axs[0].set_title("Prediction")
        axs[1].set_title("Absolute Error")
        axs[1].set_xlabel("$t$")
        axs[0].set_ylabel("$x$")
        axs[1].set_ylabel("$x$")
        return fig

    def plot_dataset(self, name: str) -> None:
        # sns.set_theme(style="whitegrid")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.size": 22,
                "axes.labelsize": 28,
                "axes.titlesize": 22,
                "legend.fontsize": 28,
                "xtick.labelsize": 28,
                "ytick.labelsize": 28,
                "lines.linewidth": 3,
            }
        )
        cm = 1 / 2.54  # centimeters in inches
        width_cm = 15
        height_cm = width_cm * 0.6
        fig, axs = plt.subplots(1, 1, figsize=[width_cm, height_cm])
        for data_name in self.get_data_names():
            if data_name == "ic_t":
                continue
            data_in = self.get_input(data_name).cpu().numpy()
            axs.scatter(data_in[:, 0], data_in[:, 1], label=data_name, alpha=0.5)

        axs.legend(loc="upper right")
        axs.set_xlabel(r"$t$")
        axs.set_ylabel(r"$\xi$")
        plt.tight_layout()
        plt.savefig(f"plots/dataset_{name}.png")

    def plot_boundary_data(self, name: str) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])

        data_in = self.get_input("boundary").cpu().numpy()
        labels = self.get_labels("boundary").cpu().numpy()

        # axs.scatter(data_in[:, 0], data_in[:, 1], c=labels, label="boundary", alpha=0.5)

        axs.legend()
        plt.savefig(f"plots/boundary_{name}.png")

    def plot_rom_data(self, name: str) -> None:

        fig, ax = plt.subplots(1, 1, figsize=[12, 6])

        data_in = self.get_input("rom").cpu().numpy()
        labels = self.get_labels("rom").cpu().numpy()
        ax.scatter(
            data_in[:, 0],
            data_in[:, 1],
            c=labels[:, 0],
            label="rom",
            alpha=0.5,
        )

        plt.savefig(f"plots/rom_{name}.png")


class WaveEquationExplicitDataModel(WaveEquationBaseModel):
    def setup_data(self, n_boundary: int, n_interior: int, n_explicit: int):
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
            [interior_data.shape[0], 1],
            device=self.config.device,
            dtype=torch.float32,
        )
        interior_data = hstack([interior_data, interior_labels])

        initial_input = self.data.get_input().to(device=self.config.device)[
            0 : self.data.fom.num_intervals + 1, :
        ]
        # downsample, TODO: this should be done in a more unified way, i guess
        len_data = len(initial_input)
        if n_boundary < len_data:
            sampling_points = np.linspace(0, len_data - 1, n_boundary, dtype=int)
            initial_input = initial_input[sampling_points]
        elif n_boundary == len_data:
            pass
        else:
            raise ValueError(
                "Cannot generate n_sample={} from data of len: {}".format(
                    n_boundary, len_data
                )
            )
        ic_t_labels = torch.zeros(
            [initial_input.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        ic_t_data = hstack([initial_input, ic_t_labels])

        # Expliction solution data
        # X, u = self.data.get_explicit_solution_data(self.wave_speed, True)
        X, u = self.data.get_explicit_solution_data(self.wave_speed, False)

        # decrease dataset size
        rand_rows = torch.randperm(X.shape[0])[:n_explicit]
        X = X[rand_rows, :]
        u = u[rand_rows]

        rom_data = hstack([X, u]).to(device=self.config.device)

        dataset_dict = {
            "interior": interior_data,
            "boundary": boundary_data,
            "rom": rom_data,
            "ic_t": ic_t_data,
        }
        complete_dataset = NamedTensorDataset(dataset_dict)

        self.set_dataset(complete_dataset)
