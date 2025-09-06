from pathlib import Path
from datetime import datetime
import wandb 
import logging

# sets up option to mute/show logging statements
def setup_logging(configs):
    logging.basicConfig(
        level=logging.DEBUG if configs.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s"
    )

def create_output_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment.name}_{timestamp}"
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir

def setup_wandb(config, output_dir):
    wandb.init(
        project=config.experiment.project_name,
        name=config.experiment.name,
        config=config,
        dir=str(output_dir),  
        mode="online"  # or "offline" for local logging
    )

