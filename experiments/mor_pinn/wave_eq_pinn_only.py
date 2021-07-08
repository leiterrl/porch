import numpy as np
import logging
from porch.boundary_conditions import BoundaryCondition

from porch.dataset import NamedTensorDataset
from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero


import torch
import argparse
from porch.config import PorchConfig
from porch.geometry import Geometry
from porch.model import BaseModel
from porch.network import FullyConnected
from porch.util import gradient
from porch.boundary_conditions import DirichletBC, DiscreteBC
from porch.training import Trainer

try:
    from torch import hstack, vstack
except ImportError:
    from porch.util import hstack, vstack

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)


class WaveEquationPINN(BaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        wave_speed: float,
        boundary_conditions: "list[BoundaryCondition]",
    ):
        super().__init__(network, geometry, config, boundary_conditions)
        self.data = DataWaveEquationZero()
        self.wave_speed = wave_speed

    # # TODO: only spatial boundary
    def boundary_loss(self, loss_name):
        """u(x=lb,t) = u(x=ub,t) = 0"""
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        return torch.pow(prediction - labels, 2)

    # TODO: complete ic (u and u_t)
    def ic_loss(self, loss_name):
        """u_t(x,t=0) = 0"""
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)
        u_t = grad_u[..., 0]

        return torch.pow(u_t - labels, 2)

    def interior_loss(self, loss_name):
        """u_tt - \\mu² * u_xx = 0"""
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

        f = u_tt - self.wave_speed ** 2 * u_xx

        return torch.pow(f - labels, 2)

    def setup_losses(self):
        self.losses = {
            "boundary": self.boundary_loss,
            "ic_t": self.ic_loss,
            "interior": self.interior_loss,
        }

    def setup_data(self, n_boundary: int, n_interior: int):
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
        ic_t_labels = torch.zeros(
            [initial_input.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        ic_t_data = hstack([initial_input, ic_t_labels])

        dataset_dict = {
            "interior": interior_data,
            "boundary": boundary_data,
            "ic_t": ic_t_data,
        }

        complete_dataset = NamedTensorDataset(dataset_dict)

        self.set_dataset(complete_dataset)

    def setup_validation_data(self) -> None:
        val_X, val_u = self.data.get_explicit_solution_data(self.wave_speed)
        self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

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
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        ninterior = self.get_labels("interior").shape[0]

        fig.suptitle(f"PINN Model")
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

        axs.legend(loc="upper right")
        plt.savefig(f"plots/dataset_{name}.png")

    def plot_boundary_data(self, name: str) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])

        data_in = self.get_input("boundary").cpu().numpy()
        labels = self.get_labels("boundary").cpu().numpy()

        axs.scatter(data_in[:, 0], data_in[:, 1], c=labels, label="boundary", alpha=0.5)

        axs.legend()
        plt.savefig(f"plots/boundary_{name}.png")


def run_model(config: PorchConfig):
    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)

    # 2D in (x,t) -> u 1D out
    network = FullyConnected(
        2, 1, config.n_layers, config.n_neurons, config.weight_norm
    )
    network.to(device=config.device)

    geom = Geometry(torch.tensor([tlims, xlims]))

    data = DataWaveEquationZero()
    X_init, u_init = data.get_initial_cond_exact(config.wave_speed)
    initial_data = hstack([X_init, u_init])
    ic = DiscreteBC("initial_bc", geom, initial_data)

    bc_axis = torch.Tensor([False, True])
    bc_upper = DirichletBC(
        "bc_upper", geom, bc_axis, xlims[1], BoundaryCondition.zero_bc_fn, False
    )
    bc_lower = DirichletBC(
        "bc_lower", geom, bc_axis, xlims[0], BoundaryCondition.zero_bc_fn, False
    )

    boundary_conditions = [ic, bc_upper, bc_lower]

    model = WaveEquationPINN(
        network, geom, config, config.wave_speed, boundary_conditions
    )

    if config.optimal_weighting:
        ## optimal pinn weighting
        print("Optimal weighting")
        model.loss_weights["ic_t"] = 0.9999560910636492
        model.loss_weights["boundary"] = 0.9999560910636492
        model.loss_weights["interior"] = 4.390893635083135e-05
    else:
        ## equal weights
        model.loss_weights["ic_t"] = 1.0
        model.loss_weights["boundary"] = 1.0
        model.loss_weights["interior"] = 1.0

    model.setup_data(config.n_boundary, config.n_interior)
    model.setup_validation_data()
    # model.plot_dataset("pinn")
    # model.plot_boundary_data("pinn")

    trainer = Trainer(model, config, config.model_dir)

    trainer.train()


def main():
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
        "--lra",
        action="store_true",
        help="Use learning rate annealing",
    )

    parser.add_argument(
        "--opt",
        action="store_true",
        help="Use optimal weighting",
    )
    args = parser.parse_args()

    n_layers = 5
    n_neurons = 20

    if args.lra and args.opt:
        raise RuntimeError("Can't enable both LRA and optimal weighting")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device, lra=args.lra)
    config.weight_norm = True
    config.wave_speed = 1.0
    config.n_boundary = args.nboundary
    config.n_interior = args.ninterior
    config.model_dir = "/import/sgs.local/scratch/leiterrl/wave_eq_pinn"
    config.epochs = 10000
    config.n_neurons = n_neurons
    config.n_layers = n_layers

    run_model(config)


if __name__ == "__main__":
    main()
