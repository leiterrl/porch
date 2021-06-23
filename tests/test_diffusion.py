from experiments import diffusion


def test_diffusion_experiment():
    assert diffusion.main(300, "plots") < 0.1
