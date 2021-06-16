from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer

from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
from experiments.mor_pinn.wave_equation_base_model import WaveEquationBaseModel
import argparse

# logging.basicConfig(level=logging.DEBUG)

# logger = logging.getLogger(__name__)
# logger.setLevel(5)

import torch

from porch.config import PorchConfig
from porch.geometry import Geometry

from porch.network import FullyConnected


import seaborn as sns

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise", type=float, default=0.0, help="Set artificial noise value for ROM"
    )
    parser.add_argument(
        "--ninterior",
        type=int,
        default=10000,
        help="Set number of interior collocation points",
    )
    parser.add_argument(
        "--nrom", type=int, default=10000, help="Set number of rom data points"
    )
    parser.add_argument(
        "--nboundary", type=int, default=500, help="Set number of rom data points"
    )
    parser.add_argument(
        "--nointerior",
        # default=False,
        action="store_true",
        help="Set artificial noise value for ROM",
    )
    args = parser.parse_args()

    model_dir = "/import/sgs.local/scratch/leiterrl/1d_wave_eq_rom"
    num_layers = 5
    num_neurons = 20
    weight_norm = False
    wave_speed = 1.0
    n_boundary = args.nboundary
    n_interior = 0 if args.nointerior else args.ninterior
    n_rom = args.nrom
    noise = args.noise

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device)
    config.epochs = 10000

    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)

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

    model = WaveEquationBaseModel(
        network,
        geom,
        config,
        wave_speed,
        boundary_conditions,
        noise=noise,
        nointerior=args.nointerior,
    )

    # model.loss_weights["boundary"] = 10.0
    model.loss_weights["interior"] = 0.001
    # model.loss_weights["boundary"] = 10.0
    # model.loss_weights["rom"] = 10.0
    model.setup_data(n_boundary, n_interior, n_rom)
    model.setup_validation_data()
    model.plot_dataset("rom")
    model.plot_boundary_data("rom")

    trainer = Trainer(model, config, model_dir)

    trainer.train()


if __name__ == "__main__":
    main()
