import numpy as np
import logging
from porch.boundary_conditions import BoundaryCondition

from porch.dataset import NamedTensorDataset
from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
from experiments.mor_pinn.wave_eq_pinn_only import WaveEquationPINN


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

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
)


class PlotWaveEqModel(WaveEquationPINN):
    def plot_boundary_and_exact_solution(self) -> None:
        fig, axs = plt.subplots(1, 1, figsize=[12, 6])

        ### TRUE SOLUTION
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        domain_shape = (-1, self.data.fom.num_intervals + 1)
        domain_extent = self.geometry.limits.flatten()
        sns.color_palette("mako", as_cmap=True)
        cmap = "mako_r"

        im_data_gt = validation_labels.detach().cpu().numpy()
        im_data_gt = im_data_gt.reshape(domain_shape)

        im1 = axs.imshow(
            np.flip(im_data_gt.T, axis=0),
            interpolation="nearest",
            extent=domain_extent,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )
        fig.colorbar(im1, extend="both", shrink=0.9, ax=axs)

        fig.suptitle(f"")
        axs.set_title("Exact solution")
        axs.set_xlabel("$t$")
        axs.set_ylabel("$x$")

        # ### BOUNDARY POINTS
        # data_in = self.get_input("boundary").cpu().numpy()
        # labels = self.get_labels("boundary").cpu().numpy()

        # axs.scatter(
        #     data_in[:, 0],
        #     data_in[:, 1],
        #     c=labels,
        #     cmap=cmap,
        #     label="boundary",
        #     # alpha=1.0,
        # )

        # axs.legend()
        plt.savefig(f"plots/wave_eq_overview.png", bbox_inches="tight")


def main():
    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)

    # 2D in (x,t) -> u 1D out
    network = FullyConnected(2, 1, 5, 20, False)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    device = torch.device("cpu")

    config = PorchConfig(device=device)

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

    model = PlotWaveEqModel(
        network, geom, config, config.wave_speed, boundary_conditions
    )

    model.setup_data(config.n_boundary, config.n_interior)
    model.setup_validation_data()
    model.plot_boundary_and_exact_solution()


if __name__ == "__main__":
    main()
