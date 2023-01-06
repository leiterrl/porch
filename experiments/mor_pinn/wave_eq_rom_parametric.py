import numpy as np
import torch
import random

seed = 0

from porch.boundary_conditions import BoundaryCondition, DirichletBC, DiscreteBC
from porch.training import Trainer
from porch.util import gradient
from porch.dataset import NamedTensorDataset

from experiments.mor_pinn.wave_mor_data_parametric import DataWaveEquationZeroParametric
from experiments.mor_pinn.wave_equation_base_model import WaveEquationBaseModel
import argparse

import matplotlib.pyplot as plt


# logging.basicConfig(level=logging.DEBUG)

# logger = logging.getLogger(__name__)
# logger.setLevel(5)


from porch.config import PorchConfig
from porch.geometry import Geometry

from porch.network import FullyConnected
from porch.util import hstack
from porch.util import parse_args


import seaborn as sns

sns.set_theme(style="white", palette="mako")
sns.color_palette("mako", as_cmap=True)


class WaveEquationParametric(WaveEquationBaseModel):
    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        boundary_conditions: "list[BoundaryCondition]",
    ):
        super().__init__(network, geometry, config, 1.0, boundary_conditions)
        self.data = DataWaveEquationZeroParametric()
        elims = geometry.limits[2, :]
        self.espace = np.linspace(elims[0], elims[1], 18)

    def interior_loss(self, loss_name):
        """u_tt - \\mu² * u_xx = 0"""
        data_in = self.get_input(loss_name)
        if len(data_in) == 0:
            return torch.zeros([1] + list(data_in.shape)[1:], device=self.config.device)
        labels = self.get_labels(loss_name)
        prediction = self.network.forward(data_in)

        grad_u = gradient(prediction, data_in)
        u_x = grad_u[..., 1].unsqueeze(1)
        u_t = grad_u[..., 0].unsqueeze(1)

        grad_u_x = gradient(u_x, data_in)
        u_xx = grad_u_x[..., 1].unsqueeze(1)

        grad_u_t = gradient(u_t, data_in)
        u_tt = grad_u_t[..., 0].unsqueeze(1)

        f = u_tt - data_in[..., 2].unsqueeze(1) ** 2 * u_xx

        return torch.pow(f - labels, 2)

    def setup_losses(self):
        self.losses = {
            "boundary": self.boundary_loss,
            "ic_t": self.ic_loss,
            "interior": self.interior_loss,
            "rom": self.rom_loss,
        }

    def setup_data(self, n_boundary: int, n_interior: int, n_rom: int):
        # spread n_boudary evenly over all boundaries (including initial condition)
        n_boundary = n_boundary // (len(self.boundary_conditions) + 1)
        bc_tensors = []
        for bc in self.boundary_conditions:
            bc_data = bc.get_samples(n_boundary, device=self.config.device)
            bc_tensors.append(bc_data)
        boundary_data = torch.cat(bc_tensors)

        interior_data = self.geometry.get_random_samples(
            n_interior, device=self.config.device
        )
        interior_labels = torch.zeros(
            [interior_data.shape[0], 1],
            device=self.config.device,
            dtype=torch.float32,
        )
        interior_data = hstack([interior_data, interior_labels])

        data = DataWaveEquationZeroParametric()
        X_init, u_init = data.get_initial_cond(self.espace[0])
        X_wavespeed_column = torch.ones((X_init.shape[0], 1))
        X_init_total = torch.hstack((X_init, X_wavespeed_column * self.espace[0]))
        u_init_total = u_init

        for ws in self.espace[1:]:
            X_init_new = torch.hstack((X_init, X_wavespeed_column * ws))
            rand_rows = torch.randperm(X_init.shape[0])[: int(self.config.n_rom / 18)]
            X_init_new = X_init_new[rand_rows, :]
            u_init_tmp = u_init[rand_rows]

            X_init_total = torch.cat([X_init_total, X_init_new])
            u_init_total = torch.cat([u_init_total, u_init_tmp])

        # initial_input = self.data.get_input().to(device=self.config.device)[
        #     0 : self.data.fom.num_intervals + 1, :
        # ]
        # # downsample, TODO: this should be done in a more unified way, i guess
        # len_data = len(initial_input)
        # if n_boundary < len_data:
        #     sampling_points = np.linspace(0, len_data - 1, n_boundary, dtype=int)
        #     initial_input = initial_input[sampling_points]
        # elif n_boundary == len_data:
        #     pass
        # else:
        #     raise ValueError(
        #         "Cannot generate n_sample={} from data of len: {}".format(
        #             n_boundary, len_data
        #         )
        #     )
        X_init_total = X_init_total.to(device=self.config.device)
        ic_t_labels = torch.zeros(
            [X_init_total.shape[0], 1], device=self.config.device, dtype=torch.float32
        )
        ic_t_data = hstack([X_init_total, ic_t_labels])

        # Rom Data

        # X = self.data.get_input()
        # TODO: remove magic number
        X = self.data.get_input_parametric(self.espace)
        u = self.data.get_data_rom_parametric(self.espace)

        # decrease dataset size
        rand_rows = torch.randperm(X.shape[0])[:n_rom]
        X = X[rand_rows, :]
        u = u[rand_rows]

        rom_data = hstack([X, u]).to(device=self.config.device)

        dataset_dict = {
            "interior": interior_data,
            "boundary": boundary_data,
            "rom": rom_data,
            "ic_t": ic_t_data,
        }
        complete_dataset = NamedTensorDataset(dataset_dict)

        self.set_dataset(complete_dataset)

    def setup_validation_data(self) -> None:
        # val_X, val_u = self.data.get_explicit_solution_data(self.wave_speed, True)
        # self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

        val_X = self.data.get_input_parametric(self.espace)
        val_u = self.data.get_data_fom_parametric(self.espace)
        # decrease dataset size
        # n_validation = int(1e5)
        # rand_rows = torch.randperm(val_X.shape[0])[:n_validation]
        # val_X = val_X[rand_rows, :]
        # val_u = val_u[rand_rows]

        self.validation_data = hstack([val_X, val_u]).to(device=self.config.device)

    def plot_validation(self):
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        # domain_shape = (-1, (self.data.fom.num_intervals + 1) // 6)
        # # TODO simplify by flattening
        # domain_extent = self.geometry.limits.flatten()
        # sns.color_palette("mako", as_cmap=True)
        # cmap = "mako_r"
        # fig, axs = plt.subplots(2, 1, figsize=[12, 6], sharex=True)

        # self.network.eval()
        # prediction = self.network.forward(validation_in)
        # self.network.train()

        # im_data = prediction.detach().cpu().numpy()
        # im_data_gt = validation_labels.detach().cpu().numpy()
        # im_data = im_data.reshape(domain_shape)
        # im_data_gt = im_data_gt.reshape(domain_shape)

        # error = np.abs(np.flip(im_data_gt.T, axis=0) - np.flip(im_data.T, axis=0))

        # im1 = axs[0].imshow(
        #     np.flip(im_data.T, axis=0),
        #     interpolation="nearest",
        #     extent=domain_extent,
        #     origin="lower",
        #     aspect="auto",
        #     cmap=cmap,
        #     vmin=-1.0,
        #     vmax=1.0,
        # )
        # # axs[1].imshow(im_data_gt.detach().cpu().numpy())
        # im2 = axs[1].imshow(
        #     error,
        #     interpolation="nearest",
        #     extent=domain_extent,
        #     origin="lower",
        #     aspect="auto",
        #     cmap=cmap,
        #     vmin=0.0,
        #     vmax=1.0,
        # )
        # fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0])
        # fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1])

        # nrom = self.get_labels("rom").shape[0]
        # ninterior = self.get_labels("interior").shape[0]

        # fig.suptitle(f"ROM-PINN nrom: {nrom} nint: {ninterior}")
        # axs[0].set_title("Prediction")
        # axs[1].set_title("Absolute Error")
        # axs[1].set_xlabel("$t$")
        # axs[0].set_ylabel("$x$")
        # axs[1].set_ylabel("$x$")
        # return fig
        n_param_samples = 18
        domain_shape = (500, 1001)
        fig, axs = plt.subplots(2, 2, figsize=[12, 6], sharex=True)
        self.network.eval()
        # with torch.no_grad():
        prediction_1 = self.network.forward(validation_in[::18, :])
        prediction_2 = self.network.forward(validation_in[17::18, :])
        # losses = loss_burgers(prediction, val_input, val_output, params={'m' : 1.0, 'k' : 1.0, 'nu': 0.01/np.pi, 'diff_nu': 0.1, 'ref_weight': 1.0}, model_input_mask=torch.ones_like(val_output).cuda())
        self.network.train()

        # train_points = model_input.detach().cpu().numpy()

        # mask_mse = mask.detach().cpu().numpy() == 0
        # mask_mse = mask_mse.squeeze()
        # mask_collocation = mask.detach().cpu().numpy() == 1
        # mask_collocation = mask_collocation.squeeze()

        im_data_1 = prediction_1.detach().cpu().numpy()[:]
        im_data_2 = prediction_2.detach().cpu().numpy()[:]

        val_size = 500 * 1001
        im_data_gt_1 = validation_labels[:val_size, :].detach().cpu().numpy()
        # im_data_gt_2 = val_output[6*val_size:7*val_size,:].detach().cpu().numpy()
        im_data_gt_2 = validation_labels[-val_size:, :].detach().cpu().numpy()

        im_data_1 = im_data_1.reshape(domain_shape)
        im_data_gt_1 = im_data_gt_1.reshape(domain_shape)
        im_data_2 = im_data_2.reshape(domain_shape)
        im_data_gt_2 = im_data_gt_2.reshape(domain_shape)
        # im_data_loss = losses['physical_loss'].detach().cpu().numpy()
        # im_data_loss = im_data_loss.reshape((num_points,num_points))
        im1 = axs[0, 0].imshow(
            np.flip(im_data_1.T, axis=0),
            interpolation="nearest",
            extent=[0.0, 2.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        # axs[1].imshow(im_data_gt.detach().cpu().numpy())
        # im3 = axs[1,0].imshow(np.flip(im_data_gt_1.T, axis=0) - np.flip(im_data_1.T, axis=0), interpolation='nearest',
        im3 = axs[1, 0].imshow(
            np.flip(im_data_gt_1.T, axis=0),
            interpolation="nearest",
            extent=[0.0, 2.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        im2 = axs[0, 1].imshow(
            np.flip(im_data_2.T, axis=0),
            interpolation="nearest",
            extent=[0.0, 2.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        # im4 = axs[1,1].imshow(np.flip(im_data_gt_2.T, axis=0) - np.flip(im_data_2.T, axis=0), interpolation='nearest',
        im4 = axs[1, 1].imshow(
            np.flip(im_data_gt_2.T, axis=0),
            interpolation="nearest",
            extent=[0.0, 2.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        # im2 = axs[1].imshow(np.flip(im_data_gt.T, axis=0), interpolation='nearest',
        #                 extent=[0.0, 4.0, -1.0, 1.0],
        #                 origin='lower', aspect='auto',
        #                 vmin=-0.3, vmax = 0.3)
        cbar = fig.colorbar(im1, extend="both", shrink=0.9, ax=axs[0, 0])
        cbar = fig.colorbar(im2, extend="both", shrink=0.9, ax=axs[1, 0])
        cbar = fig.colorbar(im3, extend="both", shrink=0.9, ax=axs[0, 1])
        cbar = fig.colorbar(im4, extend="both", shrink=0.9, ax=axs[1, 1])
        # im2 = axs[1].imshow(im_data_loss.T, interpolation='nearest',
        #                 extent=[0.0, 1.0, 0.0, 1.0],
        #                 origin='lower', aspect='auto')
        # axs[0].axis('equal')
        # axs[1].axis('equal')
        # cbar = fig.colorbar(im2, extend='both', shrink=0.9, ax=axs[1])
        # axs[0].plot(train_points[mask_mse,0], train_points[mask_mse,1], 'kx', label = 'ROM Data (%d points)' % (train_points[mask_mse,0].shape[0]), markersize = 1, clip_on = False)
        # axs[0].plot(train_points[mask_collocation,0], train_points[mask_collocation,1], 'rx', label = 'Collocation (%d points)' % (train_points[mask_collocation,0].shape[0]), markersize = 2, clip_on = False)
        # axs[0].axvline(x=2.0, linewidth=2.0, color='k')
        # axs[0].legend()
        axs[1, 1].set_xlabel("$t$")
        axs[1, 0].set_xlabel("$t$")
        axs[0, 0].set_ylabel("$x$")
        axs[1, 0].set_ylabel("$x$")
        # writer.add_figure('Result', fig, total_steps)

        # video_array = np.zeros(
        #     shape=(1, n_param_samples, 3, 3*100, 6*100),
        #     dtype=np.uint8)

        # for t in range(n_param_samples):
        #     model.eval()
        #     # with torch.no_grad():
        #     prediction = model(val_input[t::18,:])
        #     # losses = loss_burgers(prediction, val_input, val_output, params={'m' : 1.0, 'k' : 1.0, 'nu': 0.01/np.pi, 'diff_nu': 0.1, 'ref_weight': 1.0}, model_input_mask=torch.ones_like(val_output).cuda())
        #     model.train()
        #     im_data = prediction['model_out'].detach().cpu().numpy()[:]
        #     im_data = im_data.reshape(domain_shape)
        #     tmp_fig, tmp_axs = plt.subplots(1, 1, figsize=[6, 3])
        #     tmp_axs.imshow(np.flip(im_data.T, axis=0), interpolation='nearest',
        #                 extent=[0.0, 2.0, -1.0, 1.0],
        #                 origin='lower', aspect='auto',
        #                 vmin=0.0, vmax = 1.0)
        #     frame_data = fig_to_rgb_array(tmp_fig)
        #     # writer.add_figure('Result', tmp_fig, t)
        #     plt.close(tmp_fig)
        #     video_array[0, t, :, :, :] = np.transpose(frame_data, (2,0,1))

        # writer.add_video('Result_video', video_array, total_steps, fps=9)
        return fig

    def plot_dataset(self, name: str) -> None:
        # sns.set_theme(style="whitegrid")
        plt.rcParams.update(
            {
                #        "text.usetex": True,
                #        "font.family": "serif",
                #        "font.serif": ["Times New Roman"],
                "font.size": 22,
                "axes.labelsize": 28,
                "axes.titlesize": 22,
                "legend.fontsize": 28,
                "xtick.labelsize": 28,
                "ytick.labelsize": 28,
                "lines.linewidth": 3,
            }
        )
        cm = 1 / 2.54  # centimeters in inches
        width_cm = 15
        height_cm = width_cm * 0.6
        fig, axs = plt.subplots(
            1, 1, figsize=[width_cm, height_cm], subplot_kw={"projection": "3d"}
        )
        for data_name in self.get_data_names():
            if data_name == "ic_t":
                continue
            data_in = self.get_input(data_name).cpu().numpy()
            axs.scatter(
                data_in[:, 0], data_in[:, 1], data_in[:, 2], label=data_name, alpha=0.5
            )

        axs.legend(loc="upper right")
        axs.set_xlabel(r"$t$")
        axs.set_ylabel(r"$\xi$")
        plt.tight_layout()
        plt.savefig(f"plots/dataset_{name}.png")

    def plot_boundary_data(self, name: str) -> None:
        # fig, axs = plt.subplots(1, 1, figsize=[12, 6])
        fig = plt.figure(figsize=[12, 6])
        ax = fig.add_subplot(projection="3d")

        data_in = self.get_input("boundary").cpu().numpy()
        labels = self.get_labels("boundary").cpu().numpy()

        ax.scatter(
            data_in[:, 0],
            data_in[:, 1],
            data_in[:, 2],
            c=labels,
            label="boundary",
            alpha=0.5,
        )

        ax.legend()
        plt.savefig(f"plots/boundary_{name}.png")


def run_model(config: PorchConfig):

    num_parameter_points = 18

    xlims = (-1.0, 1.0)
    tlims = (0.0, 2.0)
    elims = (0.1, 0.2)
    espace = np.linspace(elims[0], elims[1], num_parameter_points)

    network = FullyConnected(
        3, 1, config.n_layers, config.n_neurons, config.weight_norm
    )
    network.to(device=config.device)

    geom = Geometry(torch.tensor([tlims, xlims, elims]))

    data = DataWaveEquationZeroParametric()
    X_init, u_init = data.get_initial_cond(espace[0])
    X_wavespeed_column = torch.ones((X_init.shape[0], 1))
    X_init_total = torch.hstack((X_init, X_wavespeed_column * espace[0]))
    u_init_total = u_init

    for ws in espace[1:]:
        X_init_new = torch.hstack((X_init, X_wavespeed_column * ws))
        # rand_rows = torch.randperm(X_init.shape[0])[:1000]
        # X_init_new = X_init_new[rand_rows, :]
        # u_init_tmp = u_init[rand_rows]

        X_init_total = torch.cat([X_init_total, X_init_new])
        u_init_total = torch.cat([u_init_total, u_init])

    initial_data = hstack([X_init_total, u_init_total])
    ic = DiscreteBC("initial_bc", geom, initial_data)

    bc_axis = torch.Tensor([False, True, False])
    bc_upper = DirichletBC(
        "bc_upper", geom, bc_axis, xlims[1], BoundaryCondition.zero_bc_fn, False
    )
    bc_lower = DirichletBC(
        "bc_lower", geom, bc_axis, xlims[0], BoundaryCondition.zero_bc_fn, False
    )

    boundary_conditions = [ic, bc_upper, bc_lower]
    # boundary_conditions = [bc_upper, bc_lower]

    model = WaveEquationParametric(
        network,
        geom,
        config,
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
    # model.plot_dataset("rom_parametric")
    # model.plot_boundary_data("rom_paramteric")

    trainer = Trainer(model, config, config.model_dir)

    trainer.train()


def main():

    args = parse_args()

    if args.determ:
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        random.seed(seed)
        np.random.seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = PorchConfig(device=device, lra=args.lra)

    if args.lra:
        config.model_dir = f"/import/sgs.local/scratch/leiterrl/wave_eq_rom_parametric_lra_{args.nbases}"
    elif args.opt:
        config.model_dir = f"/import/sgs.local/scratch/leiterrl/wave_eq_rom_parametric_opt_{args.nbases}"
    else:
        config.model_dir = f"/import/sgs.local/scratch/leiterrl/wave_eq_rom_parametric_equal_{args.nbases}"

    config.n_layers = 4
    config.n_neurons = 20
    config.weight_norm = False
    config.wave_speed = 1.0
    config.n_boundary = args.nboundary
    config.n_interior = args.ninterior
    config.n_rom = args.nrom
    config.optimal_weighting = args.opt
    config.n_bases = args.nbases
    config.deterministic = args.determ
    # config.summary_freq = 10
    config.print_freq = 500

    config.epochs = args.epochs
    if args.lbfgs:
        config.optimizer_type = "lbfgs"
        config.model_dir += "lbfgs"
    else:
        config.optimizer_type = "adam"

    run_model(config)


if __name__ == "__main__":
    main()
