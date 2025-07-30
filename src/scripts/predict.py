# import the prediction function
from omegaconf import DictConfig
import hydra
from pathlib import Path

this_dir = Path(__file__).parent
config_path = this_dir.parents[1] / "configs"

@hydra.main(config_path=str(config_path), config_name="config.yaml", version_base=None)
def main(configs: DictConfig):
    # arguments are automatically parsed by hydra and configs updated

    # run the prediction function
    return

if __name__ == '__main__':
    main()
