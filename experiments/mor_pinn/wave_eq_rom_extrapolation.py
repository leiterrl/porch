from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer
from porch.dataset import NamedTensorDataset
from experiments.mor_pinn.wave_mor_data_generation import DataWaveEquationZero
from experiments.mor_pinn.wave_equation_base_model import WaveEquationBaseModel
import logging

import torch
from porch.config import PorchConfig
from porch.geometry import Geometry
from porch.network import FullyConnected

# import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)


class WaveEquationExtrapolation(WaveEquationBaseModel):
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
        # X = data.get_input()
        X, u = data.get_data_rom(self.wave_speed, self.config.subsample_rom)

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

    model = WaveEquationExtrapolation(
        network, geom, config, wave_speed, boundary_conditions
    )

    # model.loss_weights["interior"] = 0.001
    model.loss_weights["boundary"] = 10.0
    model.loss_weights["rom"] = 10.0
    model.setup_data(n_boundary, n_interior, n_rom)
    model.setup_validation_data()

    model.plot_dataset("extrapolation")
    model.plot_boundary_data("extrapolation")

    trainer = Trainer(model, config, model_dir)

    trainer.train()


if __name__ == "__main__":
    main()
