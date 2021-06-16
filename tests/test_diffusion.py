from importlib import import_module

mod_diffusion = import_module(".1d_diffusion", package="experiments")


def test_diffusion_experiment():
    assert mod_diffusion.main(300, "plots") < 0.1
