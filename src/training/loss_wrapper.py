# it is best to make a wrapper, which computes three loss functions and returns the weighted average for my purposes

import torch.nn as nn
import torch
from . import train_utils

# adaptive to what configs are for what the loss functions will be
def get_loss_wrapper(configs):
    if configs.training.model_name == "APT1":
        return LW1(configs)
    elif configs.training.model_name == "BetterModel0":
        return CorrectedLossWrapper(configs) 
    elif configs.training.model_name == "APT0":
        return LossWrapper(configs)
    else: 
        raise ValueError("Invalid loss wrapper name")

# loss function for use with APT1
class LW1(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # collect loss functions
        self.activation_loss = torch.nn.BCELoss()
        self.onset_loss = torch.nn.BCELoss()
        self.velo_loss = torch.nn.MSELoss()

        # collect weights
        self.activation_weight = configs.training.loss_weights.activations
        self.onset_weight = configs.training.loss_weights.onsets
        self.velocity_weight = configs.training.loss_weights.velocities

    # same as self, outputs, labels. predictions is already separated, targets are still in tensor form
    def forward(self, predictions: tuple, targets: torch.Tensor):
        al = self.activation_loss(predictions[0], targets[:, 0, ...])
        ol = self.onset_loss(predictions[1], targets[:, 1, ...])
        vl = self.velo_loss(predictions[2], targets[:, 2, ...])

        sum_loss = al * self.activation_weight + ol * self.onset_weight + vl * self.velocity_weight
        return sum_loss
        

# TODO: FIX all labels being sent to each module, rather than only one channel each
# update: no idea what this is about... abandoning this code spaghetti for now
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
        
        for i, loss_fxn in enumerate(self.loss_functions):
            total += self.intraweights[i] * loss_fxn(outputs[:, i, ...], labels[:, i, ...]) 

        return total

# AI generated loss function (to fix my current one which I wrote)
class CorrectedLossWrapper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # For binary tasks (onset, frame), use BCE
        # For velocity (regression), use MSE
        self.frame_loss = nn.BCELoss()
        self.onset_loss = nn.BCELoss()
        self.velocity_loss = nn.MSELoss()
        
        # Weights from configs
        self.weights = torch.tensor([
            configs.training.loss_weights.activations,  # frame
            configs.training.loss_weights.onsets,
            configs.training.loss_weights.velocities,
        ])
    
    def forward(self, outputs, labels):
        """
        outputs: (B, 3, 88, T) - [frame, onset, velocity]
        labels: (B, 3, 88, T) - same order
        """
        # Ensure shapes match
        assert outputs.shape == labels.shape, f"Shape mismatch: {outputs.shape} vs {labels.shape}"
        
        # Extract each prediction/label
        frame_pred = outputs[:, 0, :, :]
        onset_pred = outputs[:, 1, :, :]
        velocity_pred = outputs[:, 2, :, :]
        
        frame_label = labels[:, 0, :, :]
        onset_label = labels[:, 1, :, :]
        velocity_label = labels[:, 2, :, :]
        
        # Compute individual losses
        loss_frame = self.frame_loss(frame_pred, frame_label)
        loss_onset = self.onset_loss(onset_pred, onset_label)
        loss_velocity = self.velocity_loss(velocity_pred, velocity_label)
        
        # Weighted sum
        total_loss = (
            self.weights[0] * loss_frame +
            self.weights[1] * loss_onset +
            self.weights[2] * loss_velocity
        )
        
        return total_loss
