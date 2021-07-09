from dataclasses import dataclass
import torch


@dataclass
class PorchConfig:
    """Basic configuration class for meta-data such as optimizer type, learning rate, etc."""

    lr: float = 0.001
    optimizer_type: str = "adam"
    weight_norm: bool = False
    normalize_input: bool = False
    lra: bool = False
    lra_alpha: float = 0.9
    optimal_weighting: bool = False
    device: torch.device = torch.device("cpu")
    epochs: int = 10000
    print_freq: int = 100
    summary_freq: int = 1000

    wave_speed: float = 1.0

    n_neurons: int = 20
    n_layers: int = 4

    n_boundary: int = 1000
    n_interior: int = 10000
    n_rom: int = 10000
    n_bases: int = 2

    model_dir: str = "./run"

    subsample_rom = True
    deterministic = False

    def __post__init(self):
        if self.lra and self.optimal_weighting:
            raise RuntimeError(
                "ERROR: Learning rate annealing and optimal loss weighting is mutually exclusive!"
            )
