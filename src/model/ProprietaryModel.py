# Code for the model which takes input of spectrogram (B, 1, F, T) (where B is the number
# of batches, F is frequency bucket count, and T is time frame count) and outputs 
# a (B, 1, 88, T) (or (B, 88, T)) note prediction matrix.

# the basic structure (subject to some changes later on using NAS) is 
# a couple of MBConv layers, followed by some fully connected and uniGRUs
# this will process both temporal and structural aspects of the spectrogram
import torch.nn as nn
import torch.nn.functional as F

# --- Configurable conv block ---
def make_conv_block(in_ch, out_ch, conv_type="conv", kernel_size=(3,3)):
    if conv_type == "conv":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))  # preserve time, downsample freq TODO: check
        )
    elif conv_type == "mbconv":
        # Placeholder – you could plug in your MBConv implementation
        raise NotImplementedError("MBConv not yet implemented.")
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


class model0(nn.Module):
    def __init__(self, 
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

        # After conv: (batch, C, F', T) → reshape for GRU
        self.rnn = nn.GRU(
            input_size=freq_bins * in_ch,  # flatten freq × channels
            hidden_size=rnn_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # Output: one channel per note (freq bin)
        self.fc = nn.Linear(rnn_hidden, freq_bins)

def forward(self, x):
        # x: (batch, 1, freq_bins, time_frames)
        x = self.conv_stack(x)   # (B, C, F, T')
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        x = x.view(B, T, C * F)  # flatten for GRU

        out, _ = self.rnn(x)  # (B, T, H)
        out = self.fc(out)    # (B, T, freq_bins)
        out = out.permute(0, 2, 1)  # (B, F, T) → matches spectrogram
        return out

