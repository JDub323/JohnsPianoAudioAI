# it is best to make a wrapper, which computes three loss functions and returns the weighted average for my purposes

import torch.nn as nn
import torch
from . import train_utils

# adaptive to what configs are for what the loss functions will be
# TODO: FIX all labels being sent to each module, rather than only one channel each
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
        
        # print("loss function sizes:")
        # print(f"outputs: {outputs.shape}")
        # print(f"labels: {labels.shape}")
        
        # current, wrong sizes: outputs = (16, 88 * 3, 157). labels = (16, 88, 157 * 3)
        for i, loss_fxn in enumerate(self.loss_functions):
            total += self.intraweights[i] * loss_fxn(outputs[:, i, ...], labels[:, i, ...]) 

        return total

