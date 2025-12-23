# Code for the model which takes input of spectrogram (B, 1, F, T) (where B is the number
# of batches, F is frequency bucket count, and T is time frame count) and outputs 
# a (B, 1, 88, T) (or (B, 88, T)) note prediction matrix.

# the basic structure (subject to some changes later on using NAS) is 
# a couple of MBConv layers, followed by some fully connected and uniGRUs
# this will process both temporal and structural aspects of the spectrogram
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_conv_block(in_ch, out_ch, conv_type="conv", kernel_size=(3,3)):
    if conv_type == "conv":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
    elif conv_type == "mbconv":
        # Placeholder – you could plug in your MBConv implementation
        raise NotImplementedError("MBConv not yet implemented.")
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


class model0(nn.Module):
    def __init__(self, configs,
                 freq_bins=88,
                 time_frames=1000,
                 conv_channels=[32, 64, 128],
                 conv_type="conv",
                 rnn_hidden=128,
                 num_gru_layers=2
    ) -> None:
        super().__init__()
        
        # Build conv stack dynamically
        convs = []
        in_ch = 1
        for out_ch in conv_channels:
            convs.append(make_conv_block(in_ch, out_ch, conv_type))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*convs)

        csos = (configs.training.batch_size, 128, 11, 157) # pneumonic: conv_stack_output_shape

        # After conv: (batch, C, F', T) → reshape for GRU
        self.rnn = nn.GRU(
            input_size=csos[2] * csos[1],  # flatten freq × channels
            hidden_size=rnn_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True # MAKING THE GRU BIDIRECTIONAL FOR NOW!!!!!!!!
        )

        # Output: one channel per note (freq bin)
        self.fc = nn.Linear(rnn_hidden, freq_bins)

    def forward(self, x):
        # breakpoint()
        # with settings of 10/1/2025, the size of x is [16, 88, 157]
        # since I need to reshape x, I will add the following line to add the channel dimension:
        x = x.unsqueeze(1)

        # x is now: (batch, 1, freq_bins, time_frames)
        x = self.conv_stack(x)   # (B, C, F, T')
        
        # x is now: (batch, 128, 11, 157) after going through the cnn stack
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)

        # x is now: (B, T, C, F) = (16, 157, 128, 11)
        x = x.view(B, T, C * F)  # flatten for GRU

        # x is now: (16, 157, 1408 = 128 * 11)
        out, _ = self.rnn(x)  # (B, T, H)
        out = self.fc(out)    # (B, T, freq_bins)
        out = out.permute(0, 2, 1)  # (B, F, T) → matches spectrogram
        return out

#####################################################
# everything below this was copied from Gemini

class AcousticModel(nn.Module):
    """
    Single acoustic model (feature extractor)
    Based on Table I in Mobile-AMT paper
    """
    def __init__(self, freq_bins=88, output_channels=256):
        super().__init__()
        
        # Build conv layers with explicit padding calculations
        # Each layer: stride=(2,1) with maxpool=(2,1) → 4x frequency reduction per block
        
        # Block 1: 88 → 44 (conv stride 2) → 22 (maxpool 2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))
        
        # Block 2: 22 → 11 (conv stride 2) → 5 (maxpool 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))
        
        # Block 3: 5 → 3 (conv stride 2, with padding 1: (5+2*1-3)/2+1 = 3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU()
        
        # Calculate actual output frequency dimension dynamically
        # This is more robust than hardcoding
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, freq_bins, 100)
            dummy_output = self._conv_forward(dummy_input)
            _, C, F, _ = dummy_output.shape
            self.freq_after_conv = F
            self.channels_after_conv = C
        
        self.fc = nn.Linear(self.channels_after_conv * self.freq_after_conv, 512)
        
        # Unidirectional GRU for real-time inference
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=output_channels,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3
        )
    
    def _conv_forward(self, x):
        """Helper to compute conv stack output"""
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return x
    
    def forward(self, x):
        # x: (B, 1, 88, T)
        x = self._conv_forward(x)  # (B, 128, F', T)
        
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        x = x.view(B, T, C * F)  # Flatten spatial dims
        
        x = self.fc(x)  # (B, T, 512)
        x, _ = self.gru(x)  # (B, T, 256)
        
        # Return as (B, 256, T) for concatenation
        return x.transpose(1, 2)
