from experiments import diffusion


def test_diffusion_experiment():
    assert diffusion.main(1000, "plots") < 3.0
