from scripts.utils import setup_logging
from src.corpus.build_corpus import build_corpus
from omegaconf import DictConfig
import hydra
from pathlib import Path

this_dir = Path(__file__).parent
config_path = this_dir.parents[1] / "configs"

@hydra.main(config_path=str(config_path), config_name="config.yaml", version_base=None)
def main(configs: DictConfig):
    # set up logging statements for the rest of the program
    setup_logging(configs)

    # arguments are automatically parsed by hydra and configs updated
    # run the corpus building function
    build_corpus(configs)

if __name__ == '__main__':
    main()
