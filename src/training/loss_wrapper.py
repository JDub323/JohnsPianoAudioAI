# it is best to make a wrapper, which computes three loss functions and returns the weighted average for my purposes

import torch.nn as nn
import torch
from . import train_utils

# adaptive to what configs are for what the loss functions will be
class LossWrapper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # collect loss functions
        self.loss_functions = [
            train_utils.get_basic_loss_fxn(configs.training.loss_function.activations),
            train_utils.get_basic_loss_fxn(configs.training.loss_function.onsets),
            train_utils.get_basic_loss_fxn(configs.training.loss_function.velocities),
                ]

        # collect weights
        self.intraweights = torch.tensor([
            configs.training.loss_weights.activations,
            configs.training.loss_weights.onsets,
            configs.training.loss_weights.velocities,
            ])


    # this only works if outputs are of shape [batch, channel, height, width], and labels is a torch tensor
    def forward(self, outputs, labels):
        total = 0
        
        for index, loss_fxn in enumerate(self.loss_functions):
            total += loss_fxn(outputs[:, index, ...], labels[:, index, ...]) 

        return total
