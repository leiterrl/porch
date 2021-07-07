import os
import sys
from pymordemos.explicit_solution_scenario_zero import expl_sol_handle
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# from experiments.mor_pinn.wave_mor_utils import (
from pymordemos.mor_reproduction_wave_so import (
    generate_fom,
    load_rom_data,
    eval_rom,
    compute_mse,
    compute_rom,
    compute_U_exact,
    l2_estimate_to_mse_estimate,
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
        rb_size = 4
        mus = [Parameter({"wave_speed": 1.0})]
        recompute_rom = False

        self.fom, _, _, _ = generate_fom(scenario)
        # compute exact solution
        U_exact = compute_U_exact(self.fom, mus[0])
        # compute FOM solution
        U_fom = self.fom.solve(mus[0])

        cache_folder = os.path.join(
            "cache", "wave_so", scenario, "rom_data_reconstruct.pkl"
        )
        if recompute_rom:
            # compute reductor
            reductor, basis_gen_data = compute_rom(self.fom, mus)
            # save rom data
            save_rom_data(reductor, basis_gen_data, cache_folder)
        else:
            # load rom data
            reductor, basis_gen_data = load_rom_data(cache_folder)

        self.dt = self.fom.T / self.fom.time_stepper.nt
        self.dxi = 2.0 / self.fom.num_intervals

        U_rom, l2_err_est = eval_rom(reductor, mus[0], rb_size=rb_size)
        # compute mse error
        rom_mse = compute_mse(self.fom, U_rom, U_exact)
        # compute mse error estimator
        # IMPORTANT: this quantity is used for error-sensitve training
        mse_est = l2_estimate_to_mse_estimate(self.fom, l2_err_est)
        # sanity check
        l2_err = self.fom.l2_norm(U_fom - U_rom)

        self.mse_est = mse_est
        self.U_rom = U_rom.to_numpy()
        self.U_fom = U_fom.to_numpy()
        # self.U_exact = U_exact.to_numpy()

        # # compute ROM with greedy
        # load_save = True
        # cache_path = os.path.join("cache", "wave_so", scenario, "rom_data.pkl")
        # try:
        #     if not load_save:
        #         raise FileNotFoundError
        #     logging.info("Loading saved rom from: {}".format(cache_path))
        #     self.rom, self.reductor, self.greedy_data = load_rom_data(cache_path)
        # except FileNotFoundError:
        #     logging.info("Computing ROM")
        #     training_set = self.fom.parameter_space.sample_uniformly(
        #         {"wave_speed": self.n_wave_speed}
        #     )
        #     self.rom, self.reductor, self.greedy_data = compute_rom(
        #         self.fom,
        #         training_set,
        #         max_extensions=self.max_extensions,
        #         rtol=self.rtol,
        #     )
        #     if load_save:
        #         save_rom_data(self.rom, self.reductor, self.greedy_data, cache_path)

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

    # def get_data_rom_parametric(self, wave_speeds):
    #     """
    #     docstring
    #     """
    #     mu = Parameter({"wave_speed": wave_speeds[0]})
    #     u = self.rom.solve(mu=mu)
    #     U_rom = self.reductor.reconstruct(u).to_numpy().ravel()[..., np.newaxis]

    #     for wave_speed in wave_speeds[1:]:
    #         mu = Parameter({"wave_speed": wave_speed})
    #         u = self.rom.solve(mu=mu)
    #         U_rom = np.concatenate(
    #             (
    #                 U_rom,
    #                 self.reductor.reconstruct(u).to_numpy().ravel()[..., np.newaxis],
    #             ),
    #             axis=0,
    #         )
    #     return torch.as_tensor(U_rom, dtype=torch.float32)

    # def get_data_fom_parametric(self, wave_speeds):
    #     """
    #     docstring
    #     """
    #     mu = Parameter({"wave_speed": wave_speeds[0]})
    #     U = self.fom.solve(mu).to_numpy().ravel()[..., np.newaxis]

    #     for wave_speed in wave_speeds[1:]:
    #         mu = Parameter({"wave_speed": wave_speed})
    #         U = np.concatenate(
    #             (U, self.fom.solve(mu).to_numpy().ravel()[..., np.newaxis]), axis=0
    #         )
    #     return torch.as_tensor(U, dtype=torch.float32)

    def get_epsilon(self, wave_speed: float):
        # plot error over time
        # estimated error
        # mu = Parameter({"wave_speed": wave_speed})
        # u = self.rom.solve(mu=mu)
        # err_est = self.rom.estimator.estimate(
        #     u, mu=mu, m=self.rom, return_error_sequence=True
        # )
        # true error

        # err_norms = l2_norm(Err)
        return torch.as_tensor(self.mse_est[..., np.newaxis], dtype=torch.float32)

    def get_dt(self):
        return self.dt

    # TODO: fix hardcoded spatial extend
    def get_dxi(self):
        # fom.grid.domain ?
        return self.dxi

    def get_input(self, subsample: bool = False):
        n_x = self.fom.num_intervals + 1
        n_t = self.fom.time_stepper.nt

        if subsample:
            n_x = n_x // 6
            n_t = n_t // 6

        domain = self.fom.grid.domain
        grid_x = np.linspace(domain[0], domain[1], n_x)
        grid_t = np.linspace(0.0, self.fom.T, n_t)

        tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
        return torch.as_tensor(
            np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
        )

    def get_explicit_solution_data(
        self, wave_speed: float, subsample: bool = False
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        input = self.get_input(subsample)
        expl_sol, _, _ = expl_sol_handle(wave_speed)

        U = expl_sol(input[:, 1], input[:, 0])

        return input, torch.as_tensor(U.ravel()[..., np.newaxis], dtype=torch.float32)

    def get_data_rom(self, wave_speed):
        """
        docstring
        """
        # mu = Parameter({"wave_speed": wave_speed})
        # u = self.rom.solve(mu=mu)
        # U_rom = self.reductor.reconstruct(u).to_numpy()
        return torch.as_tensor(self.U_rom.ravel()[..., np.newaxis], dtype=torch.float32)

    def get_data_fom(self, wave_speed):
        """
        docstring
        """
        # mu = Parameter({"wave_speed": wave_speed})
        # U = self.fom.solve(mu).to_numpy()
        return torch.as_tensor(self.U_fom.ravel()[..., np.newaxis], dtype=torch.float32)

    # def get_extended_bc_data(self, num_points):
    #     grid_t = np.linspace(self.fom.T, self.fom.T * 2.0, num_points)
    #     grid_x = np.array([1.0, -1.0])
    #     tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
    #     input = torch.as_tensor(
    #         np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
    #     )
    #     U = torch.zeros((input.shape[0], 1), dtype=torch.float32)
    #     return input, U

    # def get_full_bc_data(self, num_points):
    #     grid_t = np.linspace(0.0, self.fom.T, num_points)
    #     grid_x = np.array([1.0, -1.0])
    #     tt, xx = np.meshgrid(grid_t, grid_x, indexing="ij")
    #     input = torch.as_tensor(
    #         np.vstack([tt.ravel(), xx.ravel()]).T, dtype=torch.float32
    #     )
    #     U = torch.zeros((input.shape[0], 1), dtype=torch.float32)
    #     return input, U

    # def get_initial_cond_fom(self, wave_speed):
    #     initial_input = self.get_input()[0 : self.fom.num_intervals + 1, :]

    #     mu = Parameter({"wave_speed": wave_speed})
    #     U = self.fom.solve(mu).to_numpy()[0, :]
    #     return initial_input, torch.as_tensor(
    #         U.ravel()[..., np.newaxis], dtype=torch.float32
    #     )

    def get_initial_cond_exact(self, wave_speed):
        initial_input = self.get_input()[0 : self.fom.num_intervals + 1, :]
        expl_sol, _, _ = expl_sol_handle(wave_speed)

        U = expl_sol(initial_input[:, 1], initial_input[:, 0])
        return initial_input, torch.as_tensor(U[..., np.newaxis], dtype=torch.float32)
