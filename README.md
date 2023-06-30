# Porch

Porch is a Python library that enables users to train Physics-Informed Neural Networks (PINNs) using the PyTorch framework. PINNs combine the power of neural networks with the governing laws of physics, allowing for the solution of complex physical problems with limited or noisy data.

## Features

- Flexible and intuitive API for defining and training Physics-Informed Neural Networks.
- Verbose definition of a PINN [BaseModel](porch/model.py). No details are hidden from the user.
- Built-in utilities for data preprocessing, visualization, and evaluation of trained models.
- Utility functionality for generating geometry and dealing with boundary and initial conditions.

## Installation

Porch requires PyTorch to be installed as well. Please refer to the official [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions specific to your platform. This avoids pip installing the CPU-only version of PyTorch on a system with a GPU.

To install Porch, you can use pip, Python's package installer:

```shell
pip install porch
```

## Usage

For a detailed example of how to use Porch, please refer to the [experiments](experiments) directory.
A good place to start is the [Burgers' equation](experiments/burgers.py) example, which demonstrates how to use Porch to solve a simple non-linear PDE.

<!-- ## Contributing

Porch is an open-source project, and contributions from the community are welcome! If you encounter any issues, have suggestions for improvements, or would like to contribute code, please refer to the [contribution guidelines](https://github.com/leiterrl/porch/contributing.md). -->

## License

Porch is released under the Mozilla Public License 2.0 (MPL 2.0). See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions, suggestions, or comments, please feel free open an issue here on GitHub.