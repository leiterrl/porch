import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from experiments.mor_pinn.wave_mor_utils import (
    generate_fom,
    induced_norm,
    load_rom_data,
    compute_rom,
    save_rom_data,
    Parameter,
    eval_rom,
)


class DataWaveEquationZeroParametric:
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
        self.l2_norm = induced_norm(self.fom.products["l2"])
        self.h1_semi_norm = induced_norm(self.fom.h1_product)
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

        tt, xx, ee = np.meshgrid(grid_t, grid_x, wave_speeds)
        return torch.as_tensor(
            np.vstack(
                [tt.ravel(order="F"), xx.ravel(order="F"), ee.ravel(order="F")]
            ).T,
            dtype=torch.float32,
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

    def get_initial_cond(self, wave_speed):
        initial_input = self.get_input()[0 : self.fom.num_intervals + 1, :]

        mu = Parameter({"wave_speed": wave_speed})
        U = self.fom.solve(mu).to_numpy()[0, :]
        return initial_input, torch.as_tensor(
            U.ravel()[..., np.newaxis], dtype=torch.float32
        )


# def single_wave_mor_data():
#     scenario = 'GlasEtAl'
#     fom, n_wave_speed, max_extensions, rtol = generate_fom(scenario)
#     l2_norm = induced_norm(fom.products['l2'])
#     h1_semi_norm = induced_norm(fom.h1_product)
#     dt = fom.T / fom.time_stepper.nt

#     # compute ROM with greedy
#     load_save = False
#     cache_path = os.path.join('cache', 'wave_so', scenario, 'rom_data.pkl')
#     try:
#         if not load_save:
#             raise FileNotFoundError
#         rom, reductor, greedy_data = load_rom_data(cache_path)
#     except FileNotFoundError:
#         training_set = fom.parameter_space.sample_uniformly({'wave_speed': n_wave_speed})
#         rom, reductor, greedy_data = compute_rom(fom, training_set, max_extensions=max_extensions, rtol=rtol)
#         if load_save:
#             save_rom_data(rom, reductor, greedy_data, cache_path)

#     # investigate quality visually for one parameter mu
#     mu = Parameter({'wave_speed': 1.5})
#     U = fom.solve(mu)
#     u = rom.solve(mu=mu)
#     U_rom = reductor.reconstruct(u)
#     Err = U-U_rom

#     # plot solutions and error
#     # fom.visualizer.visualize(U, None, title='fom solution over time', block=False)
#     # fom.visualizer.visualize(U_rom, None, title='rom solution over time', block=False)
#     # fom.visualizer.visualize(Err, None, title='error over time', block=True)

#     # # plot greedy error
#     # plt.semilogy(greedy_data['max_errs'])
#     # plt.title('Decrease of max error during greedy')
#     # plt.xlabel('Numer of greedy iterations')
#     # plt.ylabel('Maximal L2 error at final time t_end')
#     # plt.show()

#     # plot error over time
#     # estimated error
#     err_est = rom.estimator.estimate(u, mu=mu, m=rom, return_error_sequence=True)
#     # true error

#     err_norms = l2_norm(Err)
#     # time-derivative of the true error
#     # should in theory also be bounded by the estimate in Glas et al.
#     alpha = np.sqrt(reductor.coercivity_estimator(mu))
#     V_Err = Err[0].space.empty(len(Err)-1)
#     for i in range(len(Err) - 2):
#         # central difference
#         V = Err[i+2] - Err[i]
#         V.scal(1/(2*dt))
#         V_Err.append(V)
#     # compute Riesz representant
#     V_Err = fom.h1_product.apply_inverse(V_Err)
#     v_err_norms = np.sqrt(1/alpha) * h1_semi_norm(V_Err)
#     # plot
#     # plt.semilogy(err_norms, label='L2 error')
#     # plt.semilogy(v_err_norms, label='1/4throot(alpha) * H1 velocity error')
#     # plt.semilogy(err_est, label='error bound')
#     # plt.legend(loc="upper left")
#     # plt.title('Comparison of errors and computed error bound')
#     # plt.xlabel('Time t')
#     # plt.show()

#     # test update_initial_value
#     mu = Parameter({'wave_speed': 1.5})
#     idx_pick_up = int(len(U_rom)/2)
#     initial_data_numpy = U_rom[idx_pick_up].to_numpy()
#     initial_velocity_numpy = ((U_rom[idx_pick_up+1] - U_rom[idx_pick_up]) * (1/(dt))).to_numpy()
#     space = fom.operator.source
#     new_initial_data = space.make_array(initial_data_numpy)
#     new_initial_velocity = space.make_array(initial_velocity_numpy)

#     U_new_rom, err_est = eval_rom(rom, reductor, mu, dt*idx_pick_up, fom.T, fom.time_stepper.nt - idx_pick_up, new_initial_data, new_initial_velocity)
#     Err = U[idx_pick_up:] - U_new_rom

# if __name__ == "__main__":
#     single_wave_mor_data()
