from datatypes import AugmentationData
import numpy as np

from omegaconf import DictConfig

# take in a normal array and return the ndarray augmented, 
def augment_data(configs: DictConfig, wav: np.ndarray, augmentations: AugmentationData) -> np.ndarray:
    # TODO: make sure that audio is resampled to mono if that is what configs say to do 
    # if not going to augment data, return
    if not configs.dataset.augment_audio_data:
        return wav
    # otherwise, augment data pipeline should go here
    raise ValueError('This functionality was not made yet')

def generate_augmentations(configs: DictConfig) -> AugmentationData:
    print()
    
