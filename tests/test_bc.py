import torch

from porch.boundary_conditions import DirichletBC, BoundaryCondition
from porch.geometry import Geometry


class TestBC:
    xlims = (-5.0, 5.0)
    ylims = (0.0, 20.0)

    limits = torch.tensor([xlims, ylims])

    geom = Geometry(limits)
    # boundary_func = BoundaryCondition.zero_bc_fn

    axis_def = torch.tensor([True, False])

    constant_value = 1.2
    num_samples = 10

    def construct_test_bc(self, random):
        return DirichletBC(
            "test_bc",
            self.geom,
            self.axis_def,
            self.constant_value,
            BoundaryCondition.zero_bc_fn,
            random=random,
        )

    def execute_test_bc(self, bc):
        bc_samples = bc.get_samples(self.num_samples)

        assert bc_samples.shape == torch.Size([self.num_samples, 3])

        assert bc_samples[:, 0].min() >= self.xlims[0]
        assert bc_samples[:, 0].max() <= self.xlims[1]

        assert bc_samples[:, 1].min() >= self.ylims[0]
        assert bc_samples[:, 1].max() <= self.ylims[1]

        assert torch.all(
            bc_samples[:, 0] == torch.ones(self.num_samples) * self.constant_value
        )

        assert torch.all(bc_samples[:, 2] == torch.zeros(self.num_samples))

    def test_constant_dirichlet_bc(self):
        self.execute_test_bc(self.construct_test_bc(random=False))
        self.execute_test_bc(self.construct_test_bc(random=True))


# def test_constant_dirichlet_bc():

#     bc_random = DirichletBC(
#         "test_bc_random", geom, axis_def, constant_value, boundary_func, random=True
#     )
#     bc_grid = DirichletBC(
#         "test_bc_grid", geom, axis_def, constant_value, boundary_func, random=False
#     )
