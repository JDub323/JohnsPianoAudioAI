import torch
from dataclasses import dataclass

# This data augmentation pipeline comes from the Mobile-AMT paper
# This object saves the degree which the augmentation should be/was applied
@dataclass()
class AugmentationData():
    pitch_shift: float
    speech_noise: float
    environment_noise: float # do this before room impulse response
    RIR_conv: float # room impulse response convolution
    stationary_noise: float
    DIR_conv: float # device impulse response convolution
    clamp: float # audio clipping

# this will be used throughout the project: It is what the midi labels are turned into,
# and what the model predicts. 
@dataclass()
class NoteLabels():
    activation_matrix: torch.Tensor
    onset_matrix: torch.Tensor
    velocity_matrix: torch.Tensor

@dataclass()
class ProcessedAudioSegment():
    model_input: torch.Tensor
    ground_truth: NoteLabels
    # consider adding other metadata here
