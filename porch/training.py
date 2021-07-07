# from torch.types import Number
from porch.network import FullyConnected
import torch
from .config import PorchConfig
from .model import BaseModel
import logging
from tqdm import tqdm

# from tensorboardX import SummaryWriter
from .util import CorrectedSummaryWriter as SummaryWriter
import datetime
import os

# logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: BaseModel, config: PorchConfig, model_dir: str) -> None:
        self.model = model
        self.model_dir = os.path.join(model_dir, "runs")
        self.config = config
        self.postfix_dict = {"loss": "", "val": "NaN"}
        self.progress_bar = tqdm(
            desc="Epoch: ", total=config.epochs, postfix=self.postfix_dict, delay=0.5
        )

        if config.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.network.parameters(), lr=config.lr)
        elif config.optimizer_type == "lbfgs":
            # self.optimizer = torch.optim.LBFGS(model.network.parameters())
            self.optimizer = torch.optim.LBFGS(
                model.network.parameters(),
                lr=0.1,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1e-07,
                tolerance_change=1e-09,
                history_size=100,
                # line_search_fn="strong_wolfe",
            )
        else:
            raise RuntimeError("No optimizer type specified!")
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(self.model_dir + "_" + timestamp, flush_secs=10)
        self.epoch = 0
        self.total_loss = torch.tensor([0.0])

    @staticmethod
    def compute_loss_grads(network: FullyConnected, loss: torch.Tensor):
        loss.backward(retain_graph=True)
        grads = []
        for param in network.parameters():
            if param.grad is not None:
                grads.append(torch.flatten(param.grad))
        return torch.cat(grads).clone()

    def train_epoch(self):

        if self.config.lra:
            self.optimizer.zero_grad()
            losses = self.model.compute_losses_unweighted()
            loss_grads = {}
            for name, loss in losses.items():
                self.optimizer.zero_grad()
                loss_grads[name] = self.compute_loss_grads(
                    self.model.network, torch.mean(loss)
                )

            # TODO: this is bad to assume i guess...
            first_loss_max_grad = torch.max(torch.abs(loss_grads["interior"]))
            for name, grad in loss_grads.items():
                if name == "interior":
                    continue

                update = first_loss_max_grad / torch.mean(torch.abs(grad))
                # update weights
                self.model.loss_weights["interior"] = 1.0
                self.model.loss_weights[name] = (
                    1.0 - self.config.lra_alpha
                ) * self.model.loss_weights[name] + self.config.lra_alpha * update

        if self.config.optimizer_type == "lbfgs":

            def closure() -> float:
                self.optimizer.zero_grad()

                # TODO: remove duplicate code below
                losses = self.model.compute_losses()
                closure_loss = 0.0
                for name, loss in losses.items():
                    # TODO: unify norm calculation
                    loss_mean = loss.mean()
                    self.writer.add_scalar("training/" + name, loss_mean, self.epoch)
                    closure_loss += loss_mean
                self.writer.add_scalar("training/total_loss", closure_loss, self.epoch)
                if self.epoch % self.config.print_freq == 0:
                    self.postfix_dict["loss"] = format(closure_loss, ".3e")

                closure_loss.backward()
                return closure_loss

            self.optimizer.step(closure)

        else:
            self.optimizer.zero_grad()
            losses = self.model.compute_losses()
            total_loss = 0.0
            for name, loss in losses.items():
                # TODO: unify norm calculation
                loss_mean = loss.mean()
                if self.epoch % self.config.print_freq == 0:
                    self.writer.add_scalar("training/" + name, loss_mean, self.epoch)
                total_loss += loss_mean
            if self.epoch % self.config.print_freq == 0:
                self.writer.add_scalar("training/total_loss", total_loss, self.epoch)
                self.postfix_dict["loss"] = format(total_loss, ".3e")

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return

    def update_progress(self) -> None:
        self.progress_bar.update(1)

    def train(self):
        logging.info("Start training...")
        self.epoch = 0
        val_error = 0.0
        for epoch in range(self.config.epochs + 1):
            self.epoch = epoch
            self.train_epoch()

            if epoch % self.config.print_freq == 0:
                val_error = self.model.compute_validation_error()
                self.writer.add_scalar(
                    "validating/validation_error", val_error, self.epoch
                )
                self.postfix_dict["val"] = format(val_error, ".3e")
                self.progress_bar.set_postfix(self.postfix_dict)

            if epoch % self.config.summary_freq == 0:
                fig = self.model.plot_validation()
                self.writer.add_figure("Prediction", fig, self.epoch)
                self.writer.flush()
            self.update_progress()

        total_neurons = self.config.n_layers * self.config.n_neurons

        h_dict = {
            "n_neurons": self.config.n_neurons,
            "n_layers": self.config.n_layers,
            "n_tot": total_neurons,
            "lr": self.config.lr,
        }
        h_metrics = {"val_error": val_error.item()}

        self.writer.add_hparams(h_dict, h_metrics)
        self.writer.close()
        return val_error.item()
