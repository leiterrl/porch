import torch
import numpy as np
from torch._C import device
from porch.boundary_conditions import BoundaryCondition, DirichletBC
from porch.config import PorchConfig
from porch.dataset import NamedTensorDataset
from porch.geometry import Geometry
from porch.model import BaseModel
from porch.network import FullyConnected
from porch.training import Trainer
from porch.util import gradient, parse_args

import matplotlib.pyplot as plt

diff_coeff = 0.01 / np.pi


class BurgersModel(BaseModel):
    def loss_interior(self, loss_name):
        data_in, data_out = self.get_loss_data(loss_name)

        u = self.network.forward(data_in)

        grad_u = gradient(u, data_in)
        u_x = grad_u[:, 0].unsqueeze(1)
        u_t = grad_u[:, 1].unsqueeze(1)

        gradgrad_u = gradient(grad_u, data_in)
        u_xx = gradgrad_u[:, 0].unsqueeze(1)

        f = u_t + u * u_x - diff_coeff * u_xx

        return torch.pow(f - data_out, 2)

    def setup_losses(self):
        self.losses = {
            "interior": self.loss_interior,
            "boundary": self.loss_default,
        }

    def setup_data(self):
        bc_tensors = []
        for bc in self.boundary_conditions:
            bc_data = bc.get_samples(self.config.n_boundary, device=self.config.device)
            bc_tensors.append(bc_data)
        boundary_data = torch.cat(bc_tensors)

        interior_data = self.geometry.get_random_samples(
            self.config.n_interior, device=self.config.device
        )
        interior_labels = torch.zeros(
            [interior_data.shape[0], 1],
            device=self.config.device,
            dtype=torch.float32,
        )
        interior_data = torch.hstack([interior_data, interior_labels])

        dataset = NamedTensorDataset(
            {"interior": interior_data, "boundary": boundary_data}
        )
        self.set_dataset(dataset)

    def setup_validation_data(self, n_validation: int) -> None:
        data = np.load("experiments/data/Burgers.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = exact.flatten()[:, None]
        np_data = np.hstack((X, y))
        self.validation_data = torch.as_tensor(
            np_data, dtype=torch.float32, device=self.config.device
        )

    def plot_validation(self, writer, iteration):
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (100, 256)

        domain_extent = self.geometry.limits.flatten()
        fig, axs = plt.subplots(2, 1, figsize=[12, 6], sharex=True)

        self.network.eval()
        prediction = self.network.forward(validation_in)
        self.network.train()

        im_data = prediction.detach().cpu().numpy()
        im_data_gt = validation_labels.detach().cpu().numpy()
        im_data = im_data.reshape(domain_shape).T
        im_data_gt = im_data_gt.reshape(domain_shape).T

        error = np.abs(im_data_gt - im_data)

        im1 = axs[0].imshow(
            im_data,
            interpolation="nearest",
            extent=domain_extent,
            origin="lower",
            aspect="auto",
            # cmap=cmap,
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
            # cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        # ninterior = self.get_labels("interior").shape[0]

        fig.suptitle(f"PINN Model")
        axs[0].set_title("Prediction")
        axs[1].set_title("Absolute Error")
        axs[1].set_xlabel("$x$")
        axs[0].set_ylabel("$t$")
        axs[1].set_ylabel("$t$")
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


def initial_condition_fn(data_in):
    return torch.unsqueeze(-torch.sin(np.pi * data_in[:, 0]), 1)


def main():
    config = PorchConfig()

    config.model_dir = "/data/scratch/leiterrl/burgers_new"
    # config.device = torch.device("cuda")
    config.device = torch.device("cpu")
    config.epochs = 10000
    config.n_interior = 2540
    config.n_boundary = 80
    config.lr = 5e-4

    xlims = [-1.0, 1.0]
    tlims = [0.0, 0.99]
    limits = torch.tensor([xlims, tlims])
    geom = Geometry(limits)

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
    ic = DirichletBC(
        "initial_condition",
        geom,
        torch.tensor([False, True]),
        0.0,
        initial_condition_fn,
        False,
    )

    network = FullyConnected(2, 1, 3, 20)
    network.to(config.device)
    model = BurgersModel(network, geom, config, [upper_bc, lower_bc, ic])
    model.setup_data()
    model.plot_dataset("burgers")
    model.setup_validation_data(100)

    trainer = Trainer(model, config, config.model_dir)
    trainer.train()


if __name__ == "__main__":
    main()
