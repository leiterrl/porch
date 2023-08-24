import torch

from porch.geometry import Geometry
from porch.util import gradient


class TestGradient:
    def execute_test_gradient(self):
        xlims = (0.0, 1.0)

        limits = torch.tensor([xlims])

        geom = Geometry(limits)
        data_in = geom.get_random_samples(100)

        data_in.requires_grad = True
        test_function = lambda x: torch.pow(x, 3)
        u = test_function(data_in)

        grad_u = gradient(u, data_in)
        u_x = grad_u[:, 0].unsqueeze(1)

        gradgrad_u = gradient(u_x, data_in)
        u_xx = gradgrad_u[:, 0].unsqueeze(1)

        u_x_analytical = torch.pow(data_in, 2) * 3.0
        u_xx_analytical = data_in * 6.0

        assert torch.allclose(u_x, u_x_analytical)
        assert torch.allclose(u_xx, u_xx_analytical)

    def execute_test_gradient_2d(self):
        xlims = (0.0, 1.0)
        ylims = (0.0, 1.0)

        limits = torch.tensor([xlims, ylims])

        geom = Geometry(limits)
        data_in = geom.get_random_samples(100)

        data_in.requires_grad = True
        test_function = lambda data: torch.pow(data[:, 0], 3) + torch.pow(data[:, 1], 4)
        u = test_function(data_in).unsqueeze(1)

        grad_u = gradient(u, data_in)
        u_x = grad_u[:, 0].unsqueeze(1)
        u_y = grad_u[:, 1].unsqueeze(1)

        gradgrad_u_x = gradient(u_x, data_in)
        u_xx = gradgrad_u_x[:, 0].unsqueeze(1)
        u_xy = gradgrad_u_x[:, 1].unsqueeze(1)

        gradgrad_u_y = gradient(u_y, data_in)
        u_yx = gradgrad_u_y[:, 0].unsqueeze(1)
        u_yy = gradgrad_u_y[:, 1].unsqueeze(1)

        u_x_analytical = torch.pow(data_in[:, 0], 2).unsqueeze(1) * 3.0
        u_y_analytical = torch.pow(data_in[:, 1], 3).unsqueeze(1) * 4.0

        u_xx_analytical = data_in[:, 0].unsqueeze(1) * 6.0
        u_xy_analytical = torch.zeros_like(u_xx_analytical)
        u_yx_analytical = torch.zeros_like(u_xx_analytical)
        u_yy_analytical = torch.pow(data_in[:, 1], 2).unsqueeze(1) * 12.0

        assert torch.allclose(u_x, u_x_analytical)
        assert torch.allclose(u_y, u_y_analytical)

        assert torch.allclose(u_xx, u_xx_analytical)
        assert torch.allclose(u_xy, u_xy_analytical)
        assert torch.allclose(u_yx, u_yx_analytical)
        assert torch.allclose(u_yy, u_yy_analytical)

    def test_gradient(self):
        self.execute_test_gradient()
        self.execute_test_gradient_2d()
