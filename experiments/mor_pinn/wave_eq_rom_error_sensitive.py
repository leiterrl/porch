from torch._C import device
from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer
from porch.dataset import NamedTensorDataset

import logging


from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
from experiments.mor_pinn.wave_equation_base_model import WaveEquationBaseModel
import argparse

import torch

from porch.config import PorchConfig
from porch.geometry import Geometry

from porch.network import FullyConnected

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
            interior_data = hstack([interior_data, interior_labels])

        initial_input = self.data.get_input().to(device=self.config.device)[
            0 : self.data.fom.num_intervals + 1, :
        ]
        ic_t_labels = torch.zeros(
            [initial_input.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        ic_t_data = hstack([initial_input, ic_t_labels])

        # Rom Data

        X = self.data.get_input()
        u = self.data.get_data_rom(self.wave_speed)

        rom_data = hstack([X, u]).to(device=self.config.device)

        dataset_dict = {
            "interior": interior_data,
            "boundary": boundary_data,
            "rom": rom_data,
            "ic_t": ic_t_data,
        }
        complete_dataset = NamedTensorDataset(dataset_dict)
        self.set_dataset(complete_dataset)

        dt = self.data.get_dt()
        dxi = self.data.get_dxi()
        self.epsilon = self.data.get_epsilon(self.config.wave_speed).to(
            device=self.config.device
        )
        self.d_t = torch.as_tensor(dt, device=self.config.device, dtype=torch.float32)
        self.d_xi = torch.as_tensor(dxi, device=self.config.device, dtype=torch.float32)

    def rom_loss(self, loss_name):
        data_in = self.get_input(loss_name)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        # unravel labels and prediction
        domain_shape = (-1, self.data.fom.num_intervals + 1)
        labels_spatial = labels.reshape(domain_shape)
        prediction_spatial = prediction.reshape(domain_shape)

        loss_spatial = torch.sqrt(
            (labels_spatial - prediction_spatial).abs().pow(2).sum(dim=1, keepdim=True)
            * self.d_xi
        )

        if self.heuristic:
            l_d_2 = self.relu(loss_spatial - self.epsilon).pow(2)
        else:
            l_d_2 = (2 * self.epsilon + self.relu(loss_spatial - self.epsilon)).pow(2)

        l_d = self.d_t * (l_d_2).sum(dim=0, keepdim=True)

        return l_d


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ninterior",
        type=int,
        default=10000,
        help="Set number of interior collocation points",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Set number of epochs",
    )
    # parser.add_argument(
    #     "--nrom", type=int, default=10000, help="Set number of rom data points"
    # )
    parser.add_argument(
        "--nboundary", type=int, default=1000, help="Set number of rom data points"
    )
    parser.add_argument(
        "--lra",
        action="store_true",
        help="Use learning rate annealing",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use heuristic approach",
    )
    parser.add_argument(
        "--opt",
        action="store_true",
        help="Use optimal weighting",
    )
    args = parser.parse_args()

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
    config.epochs = args.epochs
    config.optimal_weighting = args.opt

    config.optimizer_type = "adam"

    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)

    # 2D in (x,t) -> u 1D out
    network = FullyConnected(2, 1, num_layers, num_neurons, weight_norm)
    network.to(device=device)

    geom = Geometry(torch.tensor([tlims, xlims]))

    data = DataWaveEquationZero()
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
            "interior": 2.2954030700590293e-06,
            "rom": 0.5164989906081394,
            "boundary": 0.4834987139887905,
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
