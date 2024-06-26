from porch.util import relative_l2_error
from .network import FullyConnected
from .dataset import BatchedNamedTensorDataset, NamedTensorDataset
from .config import PorchConfig
from .geometry import Geometry
from .boundary_conditions import BoundaryCondition
import torch

from collections.abc import Sequence


class BaseModel:
    _input: NamedTensorDataset

    def __init__(
        self,
        network: FullyConnected,
        geometry: Geometry,
        config: PorchConfig,
        boundary_conditions: "Sequence[BoundaryCondition]",
    ):
        self.network = network
        self.geometry = geometry
        self.config = config
        self.losses = {}
        self.set_boundary_conditions(boundary_conditions)
        self.setup_losses()
        self.setup_loss_weights()

    def get_data_names(self):
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

    # TODO: possibly merge with get_labels above (replace)
    def get_labels_neumann(self, name: str):
        d_in = self.network.d_in
        return self.dataset[name][:, d_in:]

    def setup_losses(self):
        raise NotImplementedError

    def setup_loss_weights(self):
        self.loss_weights = {}
        for name in self.losses.keys():
            self.loss_weights[name] = 1.0

    def plot_validation(self, writer, iteration):
        raise NotImplementedError

    def setup_validation_data(self, n_validation: int) -> None:
        raise NotImplementedError

    def set_dataset(self, dataset: NamedTensorDataset):
        for name, value in dataset._dataset.items():
            if value.shape[1] < self.network.d_in + self.network.d_out:
                raise RuntimeError(
                    f"Dataset {name} column number has to equal d_in + d_out."
                )

        # enable mini batching
        if self.config.batch_size > 0:
            assert (
                not self.config.optimizer_type == "lbfgs"
            ), "mini-batching and lbfgs are incompatible"
            dataset = BatchedNamedTensorDataset.from_named_tensor_dataset(
                dataset,
                self.config.batch_size,
                self.config.batch_shuffle,
                self.config.batch_cycle,
            )

        self.dataset = dataset

    def compute_validation_error(self):
        validation_in = self.validation_data[:, : self.network.d_in]
        validation_labels = self.validation_data[:, -self.network.d_out :]

        # decrease dataset size
        # decrease_validation = False
        # n_validation = int(1e5)
        # rand_rows = torch.randperm(validation_in.shape[0])[:n_validation]
        # if decrease_validation:
        #     validation_in = validation_in[rand_rows, :]
        #     validation_labels = validation_labels[rand_rows]

        self.network.eval()
        prediction = self.network.forward(validation_in)
        self.network.train()

        # return torch.mean(torch.pow(prediction - validation_labels, 2))
        return relative_l2_error(prediction, validation_labels)
        # return torch.sqrt(torch.sum(torch.pow(prediction - validation_labels, 2)))

    def set_boundary_conditions(
        self, boundary_conditions: "Sequence[BoundaryCondition]"
    ):
        self.boundary_conditions = boundary_conditions

    def get_number_of_batches(self) -> int:
        if hasattr(self.dataset, "get_number_of_batches"):
            return self.dataset.get_number_of_batches()
        else:
            return 1

    def get_loss_data(self, loss_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input and labels for a given loss name"""
        data_in = self.get_input(loss_name)
        data_out = self.get_labels(loss_name)
        return data_in, data_out

    def loss_default(self, loss_name: str) -> torch.Tensor:
        """Default loss function"""
        data_in, labels = self.get_loss_data(loss_name)
        prediction = self.network.forward(data_in)
        return torch.pow(prediction - labels, 2)

    # TODO: alternatively, a callback to step the dataset might be added to a list of callbacks executed during training
    def training_step(self) -> None:
        if hasattr(self.dataset, "step"):
            self.dataset.step()
