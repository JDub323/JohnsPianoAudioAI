# model should take an input of the past 4 seconds or so of piano audio, and a pre-defined 
# amount of audio from the future (using a delay), and return a set of predictions 
# for the current moment (really, a couple moments ago). Only predict for one frame for low latency
# remember to use MBConv layers: they are fast and efficient, and it is what Mobile-AMT uses :)

import torch
import torch.nn as nn
from ..model.ProprietaryModel import model0

# pneumonic: automatic piano transcriptor
class APT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velo = model0()
        self.onset = model0()
        self.activation = model0()
        
        self.fc_onset = nn.Linear(88*2, 88)
        self.fc_activation = nn.Linear(88*2, 88)

    # returns a torch tensor whose first channel is velocity predictions, second channel is 
    # note onset predictions, and third channel is activation predictions
    def forward(self, x):
        vpred = self.velo(x)
        opred = self.onset(x) 
        apred = self.activation(x)

        opred = self.fc_onset(torch.cat([vpred, opred], dim=1)) # concat on the channel dimension
        apred = self.fc_activation(torch.cat([opred, apred], dim=1)) # concat on the channel dimension

        return torch.cat([vpred, opred, apred], dim=1)
