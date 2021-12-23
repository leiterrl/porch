import torch
import torch.nn as nn
import math

try:
    from torch.nn.init import _no_grad_trunc_normal_, _calculate_fan_in_and_fan_out
except ImportError:

    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
            )

        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            raise RuntimeError(
                "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect."
            )

        with torch.no_grad():
            # Values are generated by using a truncated uniform distribution and
            # then using the inverse CDF for the normal distribution.
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)

            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()

            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)

            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor


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
        self.mean = None
        self.std = None

        self.d_in = d_in
        self.d_out = d_out

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

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            model_input = (model_input - self.mean) / self.std
        return self.net(model_input)


def xavier_trunc_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Modified copy of PyTorch nn.init.xavier_normal_ that uses a truncated normal distribution instead

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # compensate std dev for truncation (https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
    std /= 0.87962566103423978

    return _no_grad_trunc_normal_(tensor, 0.0, std, -2.0 * std, 2.0 * std)


def init_weights_trunc_normal_(m: torch.nn.Module) -> None:
    r"""Wrapper function for applying glorot/xavier truncated normal init to torch modules

    Args:
        m: torch module
    """
    if type(m) == nn.Linear:
        if hasattr(m, "weight") and type(m.weight) is torch.Tensor:
            xavier_trunc_normal_(m.weight)
