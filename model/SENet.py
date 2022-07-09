"""
SENet模块
"""
import torch.nn as nn


class SENet(nn.Module):
    def __init__(self, input_dim, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return y
