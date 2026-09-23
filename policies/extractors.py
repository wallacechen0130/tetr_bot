"""四種棋盤特徵抽取器：SmallCNN / ResNet / CNN+Attention / Transformer。"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """輕量 CNN：CPU 友善，適合作為快速基線。"""

    def __init__(self, in_channels: int = 2, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 10 * 5, out_dim), nn.ReLU(inplace=True))
        self.output_dim = out_dim

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(board))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x.mean(dim=(2, 3)))
        return x * weights[:, :, None, None]


class ResidualBlock(nn.Module):
    """標準 residual block（Conv-BN-ReLU-Conv-BN + skip）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class ResNetBoardExtractor(nn.Module):
    """預設主線：stem + 5 個 residual block + global average pooling。"""

    def __init__(self, in_channels: int = 2, channels: int = 64, blocks: int = 5, out_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels, out_dim), nn.ReLU(inplace=True))
        self.output_dim = out_dim

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(board))
        return self.head(self.pool(x))


class CNNWithAttention(nn.Module):
    """SE 通道注意力 + 空間注意力。"""

    def __init__(self, in_channels: int = 2, channels: int = 32, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(channels)
        self.spatial = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * 20 * 10, out_dim), nn.ReLU(inplace=True))
        self.output_dim = out_dim

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        features = self.se(self.conv(board))
        attention = torch.sigmoid(self.spatial(features))
        return self.head(features * attention)


class TransformerBoardExtractor(nn.Module):
    """把棋盤切成 2x2 patch 後送入小型 Transformer encoder。"""

    def __init__(
        self,
        in_channels: int = 2,
        rows: int = 20,
        cols: int = 10,
        patch: int = 2,
        dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        tokens = (rows // patch) * (cols // patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(dim, out_dim), nn.ReLU(inplace=True))
        self.output_dim = out_dim

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        batch = board.shape[0]
        tokens = self.patch_embed(board).flatten(2).transpose(1, 2)
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.shape[1] + 1]
        x = self.encoder(x)
        return self.head(x[:, 0])


BOARD_ENCODERS = {
    "small_cnn": SmallCNN,
    "cnn": SmallCNN,
    "resnet": ResNetBoardExtractor,
    "cnn_attention": CNNWithAttention,
    "attention": CNNWithAttention,
    "transformer": TransformerBoardExtractor,
}


def build_board_encoder(name: str, *, in_channels: int = 2) -> nn.Module:
    """依名稱建立棋盤特徵抽取器。"""

    key = str(name).lower()
    if key not in BOARD_ENCODERS:
        raise KeyError(f"未知的網路架構：{name}（可用：{sorted(BOARD_ENCODERS)}）")
    return BOARD_ENCODERS[key](in_channels=in_channels)
