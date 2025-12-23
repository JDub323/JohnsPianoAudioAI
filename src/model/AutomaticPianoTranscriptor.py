# model should take an input of the past 4 seconds or so of piano audio, and a pre-defined 
# amount of audio from the future (using a delay), and return a set of predictions 
# for the current moment (really, a couple moments ago). Only predict for one frame for low latency
# remember to use MBConv layers: they are fast and efficient, and it is what Mobile-AMT uses :)

import torch
import torch.nn as nn
from torch.nn.modules import activation
from ..model.ProprietaryModel import model0
from .ProprietaryModel import AcousticModel

# pneumonic: automatic piano transcriptor
class APT0(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.velo = model0(configs)
        self.onset = model0(configs)
        self.activation = model0(configs)
        
        self.fc_onset = nn.Linear(88*2, 88)
        self.fc_activation = nn.Linear(88*2, 88)

    # returns a torch tensor whose first channel is velocity predictions, second channel is 
    # note onset predictions, and third channel is activation predictions
    def forward(self, x):
        # with settings of 10/1/2025, the size of x is [16, 88, 157]
        vpred = self.velo(x)
        opred = self.onset(x) 
        apred = self.activation(x)

        # TODO: add a gru here
        onset_input = torch.cat([vpred, opred], dim=1) # concat on the channel dimension
        onset_input = onset_input.transpose(1, 2)         # now (B, 157, 176)
        # onset input is of size: (16, 176 = 2*88, 157) 
        # THIS WILL CAUSE AN ERROR: YOU ARE ONLY SUPPOSED TO HAVE SHAPE (B, num) enter a fully connected layer
        opred = self.fc_onset(onset_input).transpose(1,2)
        # needed to transpose to apply the onset on the correct axis, now can transpose back


        activation_input = torch.cat([opred, apred], dim=1).transpose(1, 2) # concat on the channel dimension, swap axes
        # now (B, 157, 176)
        apred = self.fc_activation(activation_input).transpose(1,2) 
        # needed to transpose to apply the onset on the correct axis, now can transpose back

        # need to add a channel dimension to concatenate along
        apred = apred.unsqueeze(1)
        opred = opred.unsqueeze(1)
        vpred = vpred.unsqueeze(1)

        # breakpoint()

        # my loss function expects the order to be activation, onset, velocity
        return torch.cat([apred, opred, vpred], dim=1)

#####################################################
# everything below was copied from gemini

class PredictionHead(nn.Module):
    """
    Separate prediction head for each task (onset/frame/velocity)
    """
    def __init__(self, input_size=768, hidden_size=256, output_size=88):
        super().__init__()
        
        # Based on Mobile-AMT: Linear → Uni-GRU → Linear → Sigmoid
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False  # Unidirectional for real-time
        )
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, T, 768)
        x = self.fc1(x)  # (B, T, 256)
        x, _ = self.gru(x)  # (B, T, 256)
        x = self.fc2(x)  # (B, T, 88)
        x = self.activation(x)  # Apply sigmoid
        return x

# Corrected implementation based on actual Mobile-AMT paper
class CorrectedAPT(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Three separate acoustic models (feature extractors)
        # These produce feature representations, not predictions
        self.acoustic_model_1 = AcousticModel()
        self.acoustic_model_2 = AcousticModel()
        self.acoustic_model_3 = AcousticModel()
        
        # After concatenating 3 acoustic models, we have 256*3 = 768 features
        # Separate prediction heads for each task
        self.onset_head = PredictionHead(input_size=768, output_size=88)
        self.frame_head = PredictionHead(input_size=768, output_size=88)  # "activation" in your code
        self.velocity_head = PredictionHead(input_size=768, output_size=88)
    
    def forward(self, x):
        # x: (B, 88, T) - add channel dimension for conv
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, 88, T)
        
        # Extract features from three acoustic models
        # Each produces (B, 256, T) after internal processing
        feat1 = self.acoustic_model_1(x)
        feat2 = self.acoustic_model_2(x)
        feat3 = self.acoustic_model_3(x)
        
        # Concatenate features along channel dimension
        # Result: (B, 768, T)
        combined_features = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Transpose for GRU processing: (B, T, 768)
        combined_features = combined_features.transpose(1, 2)
        
        # Separate prediction heads
        frame_pred = self.frame_head(combined_features)     # (B, T, 88)
        onset_pred = self.onset_head(combined_features)     # (B, T, 88)
        velocity_pred = self.velocity_head(combined_features)  # (B, T, 88)
        
        # Transpose back to (B, 88, T) and add channel dimension
        frame_pred = frame_pred.transpose(1, 2).unsqueeze(1)     # (B, 1, 88, T)
        onset_pred = onset_pred.transpose(1, 2).unsqueeze(1)     # (B, 1, 88, T)
        velocity_pred = velocity_pred.transpose(1, 2).unsqueeze(1)  # (B, 1, 88, T)
        
        # Stack: (B, 3, 88, T) - order: frame, onset, velocity
        return torch.cat([frame_pred, onset_pred, velocity_pred], dim=1)

