from .network import FullyConnected
from .dataset import NamedTensorDataset
from .config import PorchConfig
from .geometry import Geometry
from .boundary_conditions import BoundaryCondition
import torch


class BaseModel:
    _input: NamedTensorDataset

    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        boundary_conditions: "list[BoundaryCondition]",
    ):
        self.network = network
        self.geometry = geometry
        self.config = config
        self.set_boundary_conditions(boundary_conditions)
        self.setup_losses()
        self.setup_loss_weights()

    def get_data_names(self) -> list:
        return self.losses.keys()

    # TODO: merge methods below
    def compute_losses_unweighted(self):
        """Iterate loss functionals and evaluate on current self.input"""
        output = {}
        for loss_name, loss_fn in self.losses.items():
            self.dataset[loss_name].requires_grad_(True)
            output[loss_name] = loss_fn(loss_name)
            self.dataset[loss_name].requires_grad_(False)

        return output

    def compute_losses(self):
        """Iterate loss functionals and evaluate on current self.input"""
        output = {}
        for loss_name, loss_fn in self.losses.items():
            self.dataset[loss_name].requires_grad_(True)
            output[loss_name] = loss_fn(loss_name) * self.loss_weights[loss_name]
            self.dataset[loss_name].requires_grad_(False)

        return output

    def compute_magnitude_normalization(self):
        raise NotImplementedError

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

    def setup_validation_data(self, n_validation: int) -> None:
        raise NotImplementedError

    def set_dataset(self, dataset: NamedTensorDataset):
        for name, value in dataset._dataset.items():
            if value.shape[1] != self.network.d_in + self.network.d_out:
                raise RuntimeError(
                    f"Dataset {name} column number has to equal d_in + d_out."
                )

        self.dataset = dataset._dataset

    def compute_validation_error(self):
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        self.network.eval()
        prediction = self.network.forward(validation_in)
        self.network.train()

        return torch.mean(torch.pow(prediction - validation_labels, 2))
        # return torch.sqrt(torch.sum(torch.pow(prediction - validation_labels, 2)))

    def set_boundary_conditions(self, boundary_conditions: "list[BoundaryCondition]"):
        self.boundary_conditions = boundary_conditions
