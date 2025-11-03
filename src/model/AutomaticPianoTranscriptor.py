# model should take an input of the past 4 seconds or so of piano audio, and a pre-defined 
# amount of audio from the future (using a delay), and return a set of predictions 
# for the current moment (really, a couple moments ago). Only predict for one frame for low latency
# remember to use MBConv layers: they are fast and efficient, and it is what Mobile-AMT uses :)

import torch
import torch.nn as nn
from torch.nn.modules import activation
from ..model.ProprietaryModel import model0

# pneumonic: automatic piano transcriptor
class APT(nn.Module):
    def __init__(self):
        super().__init__()
        self.velo = model0()
        self.onset = model0()
        self.activation = model0()
        
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
