from .network import FullyConnected
from .dataset import NamedTensorDataset
from torch._C import DictType
from .config import PorchConfig
from .geometry import Geometry
from .boundary_conditions import BoundaryCondition
import torch
import logging


class BaseModel:
    _input: NamedTensorDataset

    def __init__(
        self, network: FullyConnected, geometry: Geometry, config: PorchConfig
    ):
        self.network = network
        self.geometry = geometry
        self.config = config

    def get_data_names(self) -> list:
        return self.losses.keys()

    def compute_losses(self):
        """Iterate loss functionals and evaluate on current self.input"""
        output = {}
        for loss_name, loss_fn in self.losses.items():
            self.dataset[loss_name].requires_grad_(True)
            output[loss_name] = loss_fn(loss_name)
            self.dataset[loss_name].requires_grad_(False)

        return output

    def get_input(self, name: str):
        d_in = self.network.d_in
        return self.dataset[name][:, :d_in]

    def get_labels(self, name: str):
        d_out = self.network.d_out
        return self.dataset[name][:, -d_out:]

    def setup_losses(self):
        raise NotImplementedError

    def setup_loss_weights(self):
        self.loss_weights = {}
        for name in self.losses.keys():
            self.loss_weights[name] = 1.0

    def set_dataset(self, dataset: NamedTensorDataset):
        for name, value in dataset._dataset.items():
            if value.shape[1] != self.network.d_in + self.network.d_out:
                raise RuntimeError(
                    f"Dataset {name} column number has to equal d_in + d_out."
                )

        self.dataset = dataset._dataset

    def set_boundary_conditions(self, boundary_conditions: "list[BoundaryCondition]"):
        self.boundary_conditions = boundary_conditions
