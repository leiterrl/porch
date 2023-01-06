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
from experiments.mor_pinn.wave_eq_pinn_only import run_model


num_runs = 100

col_points = np.arange(30000, 9000000, num_runs, dtype=int)


config = PorchConfig()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

config.device = device
config.optimal_weighting = True
config.wave_speed = 1.0
config.epochs = 15000
config.model_dir = "/import/sgs.local/scratch/leiterrl/num_col_pinn_only_study"


def main():
    runs = 0
    for num_col in col_points:
        config.n_interior = int(num_col)
        run_model(config)
        torch.cuda.empty_cache()
        runs += 1
        print(f"{runs} / {num_runs}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
