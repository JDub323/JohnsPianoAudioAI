import numpy as np
from audiomentations import AddBackgroundNoise, AddGaussianSNR, ApplyImpulseResponse, Compose, PitchShift, ClippingDistortion
from omegaconf import DictConfig
from typing import Any
import logging


class DataAugmenter:
    """
    Object which augments data, or does nothing if configs.dataset.augment_audio_data is false
    there are seven transformations, which is based on the Mobile-AMT framework:
        - shift pitch 
        - add speech noise 
        - add environmental noise 
        - convolve with a room impulse response 
        - add stationary noise (gaussian white)
        - convolve with a device impulse response 
        - clamp to simulate audio clipping (rarer than the rest )

    There are only two functions:
        Initialization function: creates a compose object and saves it, along with relevant configs
                                 - enable flags for debugging: if I want to disable a certain 
                                 augmentation, I set that bool to false in the augment constructor
                                 (not the prettiest, but the fastest way to debug)

        augment_data: takes a numpy array (represents .wav file) and applies the augmentations using
                      the audiomentations library. Returns the new array and what augmentations were done
                      in a tuple
    """
    def __init__(self, configs: DictConfig, enable_list=[True,True,True,True,True,True,True]) -> None:
        # figure out if this object should actually do anything
        self._augments_data = configs.dataset.augment_audio_data


        # save sampling rate 
        self._sample_rate = configs.dataset.sample_rate

        # make a dict to map indices to the name of the transform (go in ascending order of transformation)
        self.name_mapper = {
                0: 'pitch_shift',
                1: 'speech',
                2: 'environmental',
                3: 'RIR',
                4: 'stationary',
                5: 'DIR',
                6: 'clip'
                }

        # make Compose objects
        # consider making deconvolution algorithms for new model, trained on the "undone" RIR and DIR 
        # convolutions by measuring RIR and DIR for the individual device
        # consider using dataset https://micirp.blogspot.com/search/label/AKG to get more device impulse responses

        # all transformations in one object: 
        self.mega = Compose(
                transforms=[
                    PitchShift(
                            min_semitones=-0.1,
                            max_semitones=0.1,
                            p=configs.dataset.augmentation_probabilities.pitch_shift if enable_list[0] else 0
                        ),
                    AddBackgroundNoise(sounds_path=configs.dataset.noise_paths.speech,
                           min_snr_db=0,
                           max_snr_db=20,
                           p=configs.dataset.augmentation_probabilities.speech if enable_list[1] else 0
                        ),
                    AddBackgroundNoise(sounds_path=configs.dataset.noise_paths.environmental,
                           min_snr_db=0,
                           max_snr_db=20,
                           p=configs.dataset.augmentation_probabilities.environmental if enable_list[2] else 0
                        ),
                    ApplyImpulseResponse(ir_path=configs.dataset.noise_paths.RIR, 
                                         p=configs.dataset.augmentation_probabilities.RIR if enable_list[3] else 0),
                    AddGaussianSNR(
                            min_snr_db=0,
                            max_snr_db=20,
                            p=configs.dataset.augmentation_probabilities.stationary if enable_list[4] else 0
                        ),
                    ApplyImpulseResponse(ir_path=configs.dataset.noise_paths.DIR, 
                                         p=configs.dataset.augmentation_probabilities.DIR if enable_list[5] else 0),
                    ClippingDistortion(p=configs.dataset.augmentation_probabilities.clamp if enable_list[6] else 0)
                    ]
                )


        # I need to make 3 compose objects since audiomentations is a little bad at logging what transforms were 
        # done: if two of the same kind of transformations are done, only one of "what happened" will be saved. This 
        # is not acceptible so I make three objects and gather each of their histories. The pipeline remains in the same order
        # self.first = Compose(
        #         transforms=[
        #             PitchShift(
        #                     min_semitones=-0.1,
        #                     max_semitones=0.1,
        #                     p=configs.dataset.augmentation_probabilities.pitch_shift if enable_list[0] else 0
        #                 ),
        #             AddBackgroundNoise(sounds_path=configs.dataset.noise_paths.speech,
        #                    min_snr_db=0,
        #                    max_snr_db=20,
        #                    p=configs.dataset.augmentation_probabilities.speech if enable_list[1] else 0
        #                 ),
        #             ]
        #         )
        # self.second = Compose(
        #         transforms=[
        #             AddBackgroundNoise(sounds_path=configs.dataset.noise_paths.environmental,
        #                    min_snr_db=0,
        #                    max_snr_db=20,
        #                    p=configs.dataset.augmentation_probabilities.environmental if enable_list[2] else 0
        #                 ),
        #             ApplyImpulseResponse(ir_path=configs.dataset.noise_paths.RIR, 
        #                                  p=configs.dataset.augmentation_probabilities.RIR if enable_list[3] else 0),
        #             ]
        #         )
        #
        # self.third = Compose(
        #         transforms=[
        #             AddGaussianSNR(
        #                     min_snr_db=0,
        #                     max_snr_db=20,
        #                     p=configs.dataset.augmentation_probabilities.stationary if enable_list[4] else 0
        #                 ),
        #             ApplyImpulseResponse(ir_path=configs.dataset.noise_paths.DIR, 
        #                                  p=configs.dataset.augmentation_probabilities.DIR if enable_list[5] else 0),
        #             ClippingDistortion(p=configs.dataset.augmentation_probabilities.clamp if enable_list[6] else 0)
        #             ]
        #         )


    # take in a normal array and return the ndarray augmented, and the augmentations: a dictionary
    # This data augmentation pipeline comes from the Mobile-AMT paper
    def augment_data(self, wav: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        # pipeline:
            # pitch_shift: 
            # speech_noise: 
            # environment_noise: # do this before room impulse response
            # RIR_conv: # room impulse response convolution
            # stationary_noise: 
            # DIR_conv: # device impulse response convolution
            # clamp: # audio clipping

        logging.info('Augmenting data') # log

        # if not going to augment data, return
        if not self._augments_data:
            logging.info('Skipping Data Augmentation')
            return (wav, {}) 

        # first make sure the .wav file is valid for audiomentations
        assert wav.dtype == np.float32 and np.all((wav>= -1.0) & (wav<= 1.0))

        # run the augmentation and store the new result
        new_wav = self.mega(wav, self._sample_rate) # pass the wav with a bit of silence through the aug pipe

        # save the transforms which were applied
        classes = [] 
        params = []

        # self.add_augmentations_to_lists(classes=classes, params=params, transforms=self.first.transforms)
        # self.add_augmentations_to_lists(classes=classes, params=params, transforms=self.second.transforms)
        # self.add_augmentations_to_lists(classes=classes, params=params, transforms=self.third.transforms)
        self.add_augmentations_to_lists(classes, params, transforms=self.mega.transforms)
        
        log = dict(zip(classes, params))

        # return the audio and logs as a tuple
        return (new_wav, log)
        
    def add_augmentations_to_lists(self, classes, params, transforms) -> None:
        for index, transform in enumerate(transforms):
            classes.append(self.name_mapper[index])
            params.append(transform.parameters)

