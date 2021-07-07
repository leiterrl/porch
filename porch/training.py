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
        # self.dataset = dataset

        self.optimizer = torch.optim.Adam(model.network.parameters(), lr=config.lr)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(self.model_dir + "_" + timestamp, flush_secs=10)

    @staticmethod
    def compute_loss_grads(network: FullyConnected, loss: torch.Tensor):
        loss.backward(retain_graph=True)
        grads = []
        for param in network.parameters():
            if param.grad is not None:
                grads.append(torch.flatten(param.grad))
        return torch.cat(grads).clone()

    def train_epoch(self) -> None:

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

        self.optimizer.zero_grad()
        losses = self.model.compute_losses()
        total_loss = 0
        for name, loss in losses.items():
            # TODO: unify norm calculation
            loss_mean = loss.mean()
            self.writer.add_scalar("training/" + name, loss_mean, self.epoch)
            total_loss += loss_mean
        self.writer.add_scalar("training/total_loss", total_loss, self.epoch)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def update_progress(self) -> None:
        self.progress_bar.update(1)

    def train(self) -> float:
        logging.info("Start training...")
        self.epoch = 0
        for epoch in range(self.config.epochs + 1):
            self.epoch = epoch
            loss_value = self.train_epoch()
            val_error = self.model.compute_validation_error()
            self.writer.add_scalar("validating/validation_error", val_error, self.epoch)
            self.writer.add_scalars(
                "training/loss_weights", self.model.loss_weights, epoch
            )

            if epoch % self.config.print_freq == 0:
                self.postfix_dict["loss"] = format(loss_value, ".3e")
                self.postfix_dict["val"] = format(val_error, ".3e")
                self.progress_bar.set_postfix(self.postfix_dict)

            if epoch % self.config.summary_freq == 0:
                fig = self.model.plot_validation()
                self.writer.add_figure("Prediction", fig, self.epoch)
                self.writer.flush()
            self.update_progress()

        # self.writer.add_hparams(h_dict, h_metrics)
        self.writer.close()
        return val_error


def relative_l2_error(pred, truth):
    """Relative l2 error as suggested in "Supplementary Material for Hidden
    Fluid dynamics".

        pred:   Predictions
        truth:  Reference values

        returns
            Relative l2 error. If all truth values are zero, the absolute l2
            error is returned.
    """
    if len(pred) > 0 and len(truth) > 0:
        nominator = torch.mean(torch.square(pred - truth))
        denominator = torch.mean(torch.square(truth - torch.mean(truth)))
        if denominator > 0.0:
            return nominator / denominator
        else:
            print(
                "Warning: Cannot compute relative error since exact value is"
                " constant, using absolute MSE instead"
            )
            return nominator
    else:
        return torch.tensor(0.0, device=pred.device)
