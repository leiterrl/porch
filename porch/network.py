import torch
import torch.nn as nn
import math
from torch.nn.init import _no_grad_trunc_normal_, _calculate_fan_in_and_fan_out


class FullyConnected(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_hidden_layers: int,
        n_neurons: int,
        weight_normalization: bool = False,
    ) -> None:
        super().__init__()
        layers = []

        self.d_in = d_in
        self.d_out = d_out

        # TODO: add input normalization

        # layers.append(NormLayer_WaveEq())
        if weight_normalization:
            print("Weight normalization enabled")
            layers.append(nn.utils.weight_norm(nn.Linear(d_in, n_neurons)))
        else:
            layers.append(nn.Linear(d_in, n_neurons))
        layers.append(nn.Tanh())

        for i in range(n_hidden_layers):
            if weight_normalization:
                layers.append(nn.utils.weight_norm(nn.Linear(n_neurons, n_neurons)))
            else:
                layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        layer = nn.Linear(n_neurons, d_out)
        layers.append(layer)

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights_trunc_normal_)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.net(model_input)


def xavier_trunc_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Modified copy of PyTorch nn.init.xavier_normal_ that uses a truncated normal distribution instead

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_trunc_normal_(tensor, 0.0, std, -2.0 * std, 2.0 * std)


def init_weights_trunc_normal_(m: torch.Tensor) -> torch.Tensor:
    r"""Wrapper function for applying glorot/xavier truncated normal init to torch modules

    Args:
        m: torch module
    """
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            return xavier_trunc_normal_(m.weight)
