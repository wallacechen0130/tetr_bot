"""MLP 基線：只吃 dense 特徵，用來在 CPU 上快速驗證整個訓練管線。"""

from __future__ import annotations

import torch
from torch import nn


class MLPFeatureExtractor(nn.Module):
    """多層感知機特徵抽取器。"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, layers)):
            modules += [nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.net = nn.Sequential(*modules)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
