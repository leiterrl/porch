import os
import sys
from pymordemos.explicit_solution_scenario_zero import expl_sol_handle
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# from experiments.mor_pinn.wave_mor_utils import (
from pymordemos.wave_so import (
    generate_fom,
    load_rom_data,
    compute_rom,
    save_rom_data,
    Parameter,
)


class DataWaveEquationZero:
    """
    Data generation for 1d wave eq model with 'zero' scenario
    """

    def __init__(self):
        """
        Init rom
        """
        scenario = "zero"
        self.fom, self.n_wave_speed, self.max_extensions, self.rtol = generate_fom(
            scenario
        )
        self.dt = self.fom.T / self.fom.time_stepper.nt

        # compute ROM with greedy
        load_save = True
        cache_path = os.path.join("cache", "wave_so", scenario, "rom_data.pkl")
        try:
            if not load_save:
                raise FileNotFoundError
            logging.info("Loading saved rom from: {}".format(cache_path))
            self.rom, self.reductor, self.greedy_data = load_rom_data(cache_path)
        except FileNotFoundError:
            logging.info("Computing ROM")
            training_set = self.fom.parameter_space.sample_uniformly(
                {"wave_speed": self.n_wave_speed}
            )
            self.rom, self.reductor, self.greedy_data = compute_rom(
                self.fom,
                training_set,
                max_extensions=self.max_extensions,
                rtol=self.rtol,
            )
            if load_save:
                save_rom_data(self.rom, self.reductor, self.greedy_data, cache_path)

    def get_input_extended(self):
        grid_x = np.linspace(-1.0, 1.0, self.fom.num_intervals + 1)
        grid_t = np.linspace(self.fom.T, self.fom.T * 2.0, self.fom.time_stepper.nt)

        tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
        return torch.as_tensor(
            np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
        )

    # TODO: fix hardcoded limits
    def get_input_parametric(self, wave_speeds):
        grid_x = np.linspace(-1.0, 1.0, self.fom.num_intervals + 1)
        grid_t = np.linspace(0.0, self.fom.T, self.fom.time_stepper.nt)

        tt, xx, ee = np.meshgrid(grid_t, grid_x, wave_speeds, indexing="ij")
        return torch.as_tensor(
            np.vstack([tt.ravel(), xx.ravel(), ee.ravel()]).T, dtype=torch.float32
        )

    def get_data_rom_parametric(self, wave_speeds):
        """
        docstring
        """
        mu = Parameter({"wave_speed": wave_speeds[0]})
        u = self.rom.solve(mu=mu)
        U_rom = self.reductor.reconstruct(u).to_numpy().ravel()[..., np.newaxis]

        for wave_speed in wave_speeds[1:]:
            mu = Parameter({"wave_speed": wave_speed})
            u = self.rom.solve(mu=mu)
            U_rom = np.concatenate(
                (
                    U_rom,
                    self.reductor.reconstruct(u).to_numpy().ravel()[..., np.newaxis],
                ),
                axis=0,
            )
        return torch.as_tensor(U_rom, dtype=torch.float32)

    def get_data_fom_parametric(self, wave_speeds):
        """
        docstring
        """
        mu = Parameter({"wave_speed": wave_speeds[0]})
        U = self.fom.solve(mu).to_numpy().ravel()[..., np.newaxis]

        for wave_speed in wave_speeds[1:]:
            mu = Parameter({"wave_speed": wave_speed})
            U = np.concatenate(
                (U, self.fom.solve(mu).to_numpy().ravel()[..., np.newaxis]), axis=0
            )
        return torch.as_tensor(U, dtype=torch.float32)

    # TODO: fix hardcoded limits
    def get_input(self):
        grid_x = np.linspace(-1.0, 1.0, self.fom.num_intervals + 1)
        grid_t = np.linspace(0.0, self.fom.T, self.fom.time_stepper.nt)

        tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
        return torch.as_tensor(
            np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
        )

    def get_explicit_solution_data(
        self, wave_speed: float
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        input = self.get_input()
        expl_sol, _, _ = expl_sol_handle(wave_speed)

        U = expl_sol(input[:, 1], input[:, 0])
        return input, torch.as_tensor(U[..., np.newaxis], dtype=torch.float32)

    def get_data_rom(self, wave_speed):
        """
        docstring
        """
        mu = Parameter({"wave_speed": wave_speed})
        u = self.rom.solve(mu=mu)
        U_rom = self.reductor.reconstruct(u).to_numpy()
        return torch.as_tensor(U_rom.ravel()[..., np.newaxis], dtype=torch.float32)

    def get_data_fom(self, wave_speed):
        """
        docstring
        """
        mu = Parameter({"wave_speed": wave_speed})
        U = self.fom.solve(mu).to_numpy()
        return torch.as_tensor(U.ravel()[..., np.newaxis], dtype=torch.float32)

    def get_extended_bc_data(self, num_points):
        grid_t = np.linspace(self.fom.T, self.fom.T * 2.0, num_points)
        grid_x = np.array([1.0, -1.0])
        tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
        input = torch.as_tensor(
            np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
        )
        U = torch.zeros((input.shape[0], 1), dtype=torch.float32)
        return input, U

    def get_full_bc_data(self, num_points):
        grid_t = np.linspace(0.0, self.fom.T, num_points)
        grid_x = np.array([1.0, -1.0])
        tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
        input = torch.as_tensor(
            np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
        )
        U = torch.zeros((input.shape[0], 1), dtype=torch.float32)
        return input, U

    def get_initial_cond_fom(self, wave_speed):
        initial_input = self.get_input()[0 : self.fom.num_intervals + 1, :]

        mu = Parameter({"wave_speed": wave_speed})
        U = self.fom.solve(mu).to_numpy()[0, :]
        return initial_input, torch.as_tensor(
            U.ravel()[..., np.newaxis], dtype=torch.float32
        )

    def get_initial_cond_exact(self, wave_speed):
        initial_input = self.get_input()[0 : self.fom.num_intervals + 1, :]
        expl_sol, _, _ = expl_sol_handle(wave_speed)

        U = expl_sol(initial_input[:, 1], initial_input[:, 0])
        return initial_input, torch.as_tensor(U[..., np.newaxis], dtype=torch.float32)
