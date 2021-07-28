import numpy as np
import torch
import random

seed = 0

from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer

from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
from experiments.mor_pinn.wave_equation_base_model import (
    WaveEquationBaseModel,
    WaveEquationExplicitDataModel,
)
import argparse


# logging.basicConfig(level=logging.DEBUG)

# logger = logging.getLogger(__name__)
# logger.setLevel(5)


from porch.config import PorchConfig
from porch.geometry import Geometry

from porch.network import FullyConnected
from porch.util import hstack, parse_args


import seaborn as sns

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)


def run_model(config: PorchConfig):

    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)

    if config.deterministic:
        network = torch.load("./cache/model.pth")
    else:
        network = FullyConnected(
            2, 1, config.n_layers, config.n_neurons, config.weight_norm
        )
        torch.save(network, "./cache/model.pth")
    network.to(device=config.device)

    geom = Geometry(torch.tensor([tlims, xlims]))

    data = DataWaveEquationZero()
    X_init, u_init = data.get_initial_cond_exact(config.wave_speed)
    # decrease initial condition dataset size
    # rand_rows = torch.randperm(X_init.shape[0])[:n_boundary]
    # X_init = X_init[rand_rows, :]
    # u_init = u_init[rand_rows]
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

    model = WaveEquationExplicitDataModel(
        network,
        geom,
        config,
        config.wave_speed,
        boundary_conditions,
    )

    # model.loss_weights["boundary"] = 10.0
    # model.loss_weights["interior"] = 0.001
    # model.loss_weights["boundary"] = 10.0
    # model.loss_weights["rom"] = 10.0
    ## optimal pinn-rom weights
    if config.optimal_weighting:
        model.loss_weights = {
            "interior": 1.463186091733802e-05,
            "rom": 0.666768040099405,
            "boundary": 0.3332173280396778,
        }
    else:
        model.loss_weights = {
            "interior": 1.0,
            "rom": 1.0,
            "boundary": 1.0,
        }

    # model.loss_weights["rom"] = 1.0
    # model.loss_weights["interior"] = 1.0e-3
    # model.loss_weights["interior"] = 1.0
    ## equal weighting
    # model.loss_weights["rom"] = 1.0
    # model.loss_weights["boundary"] = 1.0
    # model.loss_weights["interior"] = 1.0
    ## optimal pinn weighting
    # model.loss_weights["rom"] = 1.0
    # model.loss_weights["boundary"] = 1.0
    # model.loss_weights["interior"] = 4.336622162708593e-06

    model.loss_weights["ic_t"] = model.loss_weights["boundary"]
    model.setup_data(config.n_boundary, config.n_interior, config.n_rom)
    model.setup_validation_data()
    # model.plot_dataset("rom")
    # model.plot_boundary_data("rom")

    trainer = Trainer(model, config, config.model_dir)

    trainer.train()


def main():

    args = parse_args()
    if args.determ:
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        random.seed(seed)
        np.random.seed(0)

    if args.lra and args.opt:
        raise RuntimeError("Can't enable both LRA and optimal weighting")
    
    if args.lra and args.batchsize:
        assert args.batchcycle, 'Use lra only with batchcycle, otherwise errors might occur due to empty data sets'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device, lra=args.lra)

    if config.lra:
        config.model_dir = f"/import/sgs.local/scratch/leiterrl/wave_eq_rom_explicit_lra_{args.batchsize}"
    elif args.opt:
        config.model_dir = f"/import/sgs.local/scratch/leiterrl/wave_eq_rom_explicit_opt_{args.batchsize}"
    else:
        config.model_dir = f"/import/sgs.local/scratch/leiterrl/wave_eq_rom_explicit_equal_{args.batchsize}"
    config.n_layers = 5
    config.n_neurons = 20
    config.weight_norm = False
    config.wave_speed = 1.0
    config.n_boundary = args.nboundary
    config.n_interior = args.ninterior
    config.deterministic = args.determ
    config.batch_size = args.batchsize
    config.batch_cycle = args.batchcycle

    config.epochs = args.epochs
    if args.lbfgs:
        config.optimizer_type = "lbfgs"
        config.model_dir += "lbfgs"
    else:
        config.optimizer_type = "adam"

    if args.opt:
        config.optimal_weighting = True

    run_model(config)


if __name__ == "__main__":
    main()
