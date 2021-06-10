from dataclasses import dataclass
import torch


@dataclass
class PorchConfig:
    """Basic configuration class for meta-data such as optimizer type, learning rate, etc."""

    lr: float = 0.001
    optimizer_type: str = "adam"
    weight_norm: bool = False
    lra: bool = False
    optimal_weighting: bool = False
    device: torch.device = torch.device("cpu")
    epochs: int = 10000
    print_freq: int = 100
    summary_freq: int = 1000

    def __post__init(self):
        if self.lra and self.optimal_weighting:
            raise RuntimeError(
                "ERROR: Learning rate annealing and optimal loss weighting is mutually exclusive!"
            )
