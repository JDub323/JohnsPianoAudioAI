# import the evaluation function
from omegaconf import DictConfig
import hydra
from pathlib import Path

from scripts.utils import setup_logging

this_dir = Path(__file__).parent
config_path = this_dir.parents[1] / "configs"

# arguments are automatically parsed by hydra and configs updated
@hydra.main(config_path=str(config_path), config_name="config.yaml", version_base=None)
def main(configs: DictConfig):
    # set up the verbosity of logging
    setup_logging(configs)

    # run the evaluation function
    return

if __name__ == '__main__':
    main()

