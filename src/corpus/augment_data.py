

from omegaconf import DictConfig


def augment_data(configs: DictConfig):
    # if not going to augment data, return
    if not configs.dataset.augment_audio_data:
        return
    # otherwise, augment data pipeline should go here

