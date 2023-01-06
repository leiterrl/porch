import argparse
from porch.util import parse_args
from porch.config import PorchConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["rom", "es", "pinn", "expl"])
    args = parser.parse_args()

    config = PorchConfig()
    if args.model == "rom":
        config.



if __name__ == "__main__":
    main()


# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --opt --heuristic --epochs=40000 --nbases=4; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --opt --heuristic --epochs=40000 --nbases=8; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --opt --heuristic --epochs=40000 --nbases=12; done

# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --lra --heuristic --epochs=40000 --nbases=4; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --lra --heuristic --epochs=40000 --nbases=8; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --lra --heuristic --epochs=40000 --nbases=12; done

# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom.py --opt --epochs=40000 --nbases=4; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom.py --opt --epochs=40000 --nbases=8; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom.py --opt --epochs=40000 --nbases=12; done

# for run in {1..10}; do CUDA_AVAILABLE_DEVICES=0 python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --heuristic --epochs=40000 --nbases=4; done
# for run in {1..10}; do CUDA_AVAILABLE_DEVICES=1 python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --heuristic --epochs=40000 --nbases=8; done
# for run in {1..10}; do CUDA_AVAILABLE_DEVICES=2 python experiments/mor_pinn/wave_eq_rom_error_sensitive.py --heuristic --epochs=40000 --nbases=12; done

# for run in {1..10}; do CUDA_AVAILABLE_DEVICES=3 python experiments/mor_pinn/wave_eq_rom.py --epochs=40000 --nbases=4; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom.py --epochs=40000 --nbases=8; done
# for run in {1..10}; do python experiments/mor_pinn/wave_eq_rom.py --epochs=40000 --nbases=12; done