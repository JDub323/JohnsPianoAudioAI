# TODO: REPLACE WITH SMARTER MODEL
# model should take an input of the past 4 seconds or so of piano audio, and a pre-defined 
# amount of audio from the future (using a delay), and return a set of predictions 
# for the current moment (really, a couple moments ago). Only predict for one frame for low latency

import torch.nn as nn
import torch.nn.functional as F

# pneumonic: automatic piano transcriptor
class APT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # idk what this actually does lol
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
