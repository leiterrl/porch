import logging

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from porch.geometry import Geometry

from .config import PorchConfig

# class PorchDatasetBase:
#     def __init__(self, config: PorchConfig, geometry: Geometry) -> None:
#         self.config = config
#         self.geometry = geometry
#         logging.info("Init dataset...")

#     def generate_data(self):
#         bc_tensors = []
#         logging.info("Generating BC data...")
#         for bc in self.geometry.get_boundary_conditions():
#             bc_data = bc.get_data(device=self.config.device)
#             bc_tensors.append(bc_data)
#         boundary_data = torch.cat(bc_tensors)

#         logging.info("Generating interior data...")
#         interior_data = self.geometry.get_random_samples(device=self.config.device)
#         self.dataset = TensorDataset(boundary_data, interior_data)


class NamedTensorDataset:
    _dataset: "dict[str, Tensor]"
    _names: "list[str]"

    def __init__(self, tensor_dict: "dict[str, Tensor]") -> None:
        self._dataset = tensor_dict
        self._names = list(tensor_dict.keys())

    def get_dataset(self, name: str) -> Tensor:
        return self._dataset[name]
