import torch
from dataclasses import dataclass

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
    def __post_init__(self):
            # Ensure all are torch.Tensors
            if not isinstance(self.activation_matrix, torch.Tensor):
                print("activ was not a torch tensor to begin with")
                self.activation_matrix = torch.tensor(self.activation_matrix)
            if not isinstance(self.onset_matrix, torch.Tensor):
                print("onset was not a torch tensor to begin with")
                self.onset_matrix = torch.tensor(self.onset_matrix)
            if not isinstance(self.velocity_matrix, torch.Tensor):
                print("velo was not a torch tensor to begin with")
                self.velocity_matrix = torch.tensor(self.velocity_matrix)

@dataclass()
class ProcessedAudioSegment():
    model_input: torch.Tensor
    ground_truth: NoteLabels
    # consider adding other metadata here
