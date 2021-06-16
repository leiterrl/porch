from importlib import import_module

mod_diffusion = import_module("experiments.1d_diffusion")


def test_diffusion_experiment():
    assert mod_diffusion.main(300) < 0.1
