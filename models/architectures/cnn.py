"""
1D CNN for Hurst exponent estimation.
Exact replication of Stone (2020), "Calibrating rough volatility models:
a convolutional neural network approach", Quantitative Finance, 20:3, 379-392.

From Section 3.3 of the paper:
- Three convolutional layers, kernel size 20 for each
- LeakyReLU activation (alpha=0.1) after each conv layer
- MaxPooling of size 3 after each conv layer
- Zero-padding in convolutional and pooling layers
- Dropout between layers (rates: 0.25, 0.25, 0.4, 0.3)
- Dense layer with 128 units + LeakyReLU
- Output: 1 unit
- Optimizer: Adam
- Loss: MSE
- Batch size: 64
- Epochs: 30
"""

import torch
import torch.nn as nn


class HurstCNN(nn.Module):
    """
    Exact CNN architecture from Stone (2020).

    Structure (from paper Section 3.3):
        Conv1d(1→32, kernel=20, zero-padding) + LeakyReLU(0.1) + MaxPool(3)
        Dropout(0.25)
        Conv1d(32→64, kernel=20, zero-padding) + LeakyReLU(0.1) + MaxPool(3)
        Dropout(0.25)
        Conv1d(64→128, kernel=20, zero-padding) + LeakyReLU(0.1) + MaxPool(3)
        Dropout(0.4)
        Flatten → Dense(128) + LeakyReLU(0.1) + Dropout(0.3) → Dense(1)
    """

    def __init__(self, input_size: int = 100):
        super().__init__()

        # "zero-padding" preserves input dimensions: padding = (kernel_size - 1) // 2
        # For kernel_size=20 (even), we use padding=10 on the left side.
        # PyTorch Conv1d only supports symmetric padding, so we pad to keep
        # the output close to input length. padding=10 gives output = input+1,
        # which MaxPool(3) handles fine.
        pad = 10  # approximately (20-1)//2

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=20, padding=pad),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3, padding=1),
        )
        self.drop1 = nn.Dropout(0.25)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=20, padding=pad),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3, padding=1),
        )
        self.drop2 = nn.Dropout(0.25)

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=20, padding=pad),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3, padding=1),
        )
        self.drop3 = nn.Dropout(0.4)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size)
            x = self.conv_block1(dummy)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            self.flat_size = x.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x shape: (batch, 100) → need (batch, 1, 100) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.drop1(self.conv_block1(x))
        x = self.drop2(self.conv_block2(x))
        x = self.drop3(self.conv_block3(x))

        x = x.view(x.size(0), -1)
        return self.fc_layers(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
