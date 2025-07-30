
from omegaconf import DictConfig

def train(configs: DictConfig, checkpoint_path: str) -> None:
    print("Configs:")
    print(configs)


