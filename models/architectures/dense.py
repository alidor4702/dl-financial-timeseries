"""
Dense (fully connected) neural network architectures for Hurst exponent estimation.

We define multiple configurations to compare:
- DenseSmall: shallow, fewer parameters (baseline)
- DenseMedium: balanced depth and width (main model)
- DenseLarge: deeper, more capacity
"""

import torch
import torch.nn as nn


class DenseSmall(nn.Module):
    """Shallow dense network. 2 hidden layers."""

    def __init__(self, input_size: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DenseMedium(nn.Module):
    """Medium dense network. 4 hidden layers with batch norm and dropout."""

    def __init__(self, input_size: int = 100, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DenseLarge(nn.Module):
    """Larger dense network. 5 hidden layers, wider."""

    def __init__(self, input_size: int = 100, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dense_model(size: str = "medium", **kwargs) -> nn.Module:
    models = {
        "small": DenseSmall,
        "medium": DenseMedium,
        "large": DenseLarge,
    }
    return models[size](**kwargs)
