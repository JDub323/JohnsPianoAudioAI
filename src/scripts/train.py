from omegaconf import DictConfig
from src.training.train_model import train
import hydra
from pathlib import Path
from .utils import setup_logging

this_dir = Path(__file__).parent
config_path = this_dir.parents[1] / "configs"

@hydra.main(config_path=str(config_path), config_name="config.yaml", version_base=None)
def main(configs: DictConfig):
    # set up the verbosity of logging
    setup_logging(configs)

    # arguments are automatically parsed by hydra and configs updated

    # run eval with updated paths. Inside this method is where the rest of the configs will be adjusted
    train(configs=configs)

if __name__ == '__main__':
    main()
