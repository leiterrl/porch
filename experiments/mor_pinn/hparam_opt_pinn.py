import argparse
from porch.config import PorchConfig

import torch
from torch.multiprocessing import Pool, Process, set_start_method

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

import numpy as np
from numpy.random import default_rng

rng = default_rng()
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.mor_pinn.pinn_only import run_model


lower_bounds = [2, 20, 1e-5]

upper_bounds = [8, 60, 1e-2]


num_runs = 100

parser = argparse.ArgumentParser()


parser.add_argument(
    "--lra",
    action="store_true",
    help="Use learning rate annealing",
)

parser.add_argument(
    "--opt",
    action="store_true",
    help="Use optimal weighting",
)
args = parser.parse_args()

config = PorchConfig()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

config.device = device
config.lra = False
config.optimal_weighting = False
config.weight_norm = False
config.wave_speed = 1.0
config.epochs = 20000


if args.opt:
    config.optimal_weighting = True
    config.model_dir = "/import/sgs.local/scratch/leiterrl/h_param_wave_eq_pinn_opt"
elif args.lra:
    config.lra = True
    config.model_dir = "/import/sgs.local/scratch/leiterrl/h_param_wave_eq_pinn_lra"
else:
    config.model_dir = "/import/sgs.local/scratch/leiterrl/h_param_wave_eq_pinn_equal"


def h_param_wrapper(n_layers, n_neurons, lr):

    config.n_neurons = n_neurons
    config.n_layers = n_layers
    config.lr = lr

    run_model(config)


def wrapper_function():
    sample = np.random.randint(lower_bounds[:-1], upper_bounds[:-1]).astype(int)
    sample_float = round(np.random.uniform(lower_bounds[-1], upper_bounds[-1]), 5)
    sample = (sample[0].item(), sample[1].item(), sample_float)
    print(sample)
    return h_param_wrapper(*sample)


def main():
    runs = 0
    while runs < num_runs:
        wrapper_function()
        torch.cuda.empty_cache()
        runs += 1
        print(f"{runs} / {num_runs}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    main()
