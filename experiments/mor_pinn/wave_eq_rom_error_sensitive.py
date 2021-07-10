import numpy as np
import torch
import random

seed = 0

from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer
from porch.dataset import NamedTensorDataset

import logging


from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
from experiments.mor_pinn.wave_equation_base_model import WaveEquationBaseModel
import argparse

from porch.config import PorchConfig
from porch.geometry import Geometry

from porch.network import FullyConnected
from porch.util import parse_args

try:
    from torch import hstack, vstack
except ImportError:
    from porch.util import hstack, vstack


import seaborn as sns

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)


class WaveEquationErrorSensitive(WaveEquationBaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        wave_speed: float,
        boundary_conditions: "list[BoundaryCondition]",
        heuristic: bool,
    ):
        super().__init__(network, geometry, config, wave_speed, boundary_conditions)
        self.relu = torch.nn.ReLU()
        self.heuristic = heuristic

    def setup_data(self, n_boundary: int, n_interior: int):
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
            sampling_points = torch.linspace(0, len_data-1, n_boundary, dtype=int)
            initial_input = initial_input[sampling_points]
        elif n_boundary == len_data:
            pass
        else:
            raise ValueError('Cannot generate n_sample={} from data of len: {}'.format(n_boundary, len_data))    
        ic_t_labels = torch.zeros(
            [initial_input.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        ic_t_data = hstack([initial_input, ic_t_labels])

        # Rom Data

        # X = self.data.get_input()
        X, u = self.data.get_data_rom(self.wave_speed, self.config.subsample_rom)

        rom_data = hstack([X, u]).to(device=self.config.device)

        dataset_dict = {
            "interior": interior_data,
            "boundary": boundary_data,
            "rom": rom_data,
            "ic_t": ic_t_data,
        }
        complete_dataset = NamedTensorDataset(dataset_dict)
        self.set_dataset(complete_dataset)

        self.epsilon = self.data.get_epsilon(self.config.wave_speed).to(
            device=self.config.device
        )

    def rom_loss(self, loss_name):
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        # unravel labels and prediction
        domain_shape = (-1, (self.data.fom.num_intervals + 1) // 6)
        labels_spatial = labels.reshape(domain_shape)
        prediction_spatial = prediction.reshape(domain_shape)

        loss_spatial = torch.sqrt(
            (labels_spatial - prediction_spatial).pow(2).mean(dim=1, keepdim=True)
        )

        if self.heuristic:
            l_d_2 = self.relu(loss_spatial - self.epsilon[::6, :]).pow(2)
        else:
            l_d_2 = (
                2 * self.epsilon[::6, :]
                + self.relu(loss_spatial - self.epsilon[::6, :])
            ).pow(2)

        l_d = (l_d_2).mean(dim=0, keepdim=True)

        return l_d


def main():

    args = parse_args()

    if args.determ:
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        random.seed(seed)
        np.random.seed(0)

    model_dir = "/import/sgs.local/scratch/leiterrl/wave_eq_rom_error_sensitive"
    num_layers = 5
    num_neurons = 20
    weight_norm = False
    wave_speed = 1.0
    n_boundary = args.nboundary
    n_interior = args.ninterior
    # n_rom = args.nrom

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device, lra=args.lra)
    config.deterministic = args.determ
    config.epochs = args.epochs
    config.optimal_weighting = args.opt
    config.n_bases = args.nbases

    config.optimizer_type = "adam"

    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)

    if config.deterministic:
        network = torch.load("./cache/model.pth")
    else:
        network = FullyConnected(
            2, 1, config.n_layers, config.n_neurons, config.weight_norm
        )
        torch.save(network, "./cache/model.pth", _use_new_zipfile_serialization=False)
    network.to(device=config.device)

    geom = Geometry(torch.tensor([tlims, xlims]))

    data = DataWaveEquationZero(config.n_bases)
    X_init, u_init = data.get_initial_cond_exact(wave_speed)

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

    model = WaveEquationErrorSensitive(
        network, geom, config, wave_speed, boundary_conditions, args.heuristic
    )

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

    model.loss_weights["ic_t"] = model.loss_weights["boundary"]
    model.setup_data(n_boundary, n_interior)
    model.setup_validation_data()
    model.plot_dataset("rom")
    model.plot_boundary_data("rom")

    trainer = Trainer(model, config, model_dir)

    trainer.train()


if __name__ == "__main__":
    main()
