"""網路工廠：組裝棋盤編碼器 + 向量編碼器 + 動作頭。"""

from __future__ import annotations

import torch
from torch import nn

from envs.gym.obs_encoder import VECTOR_DIM, VECTOR_KEYS
from policies.action_heads import PolicyValueHead
from policies.extractors import BOARD_ENCODERS, build_board_encoder
from policies.mlp import MLPFeatureExtractor

NETWORK_REGISTRY = tuple(BOARD_ENCODERS)


class TetrisNetwork(nn.Module):
    """主網路：board encoder + vector MLP → fusion → policy/value head。"""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        encoder_name: str = "resnet",
        vector_dim: int = VECTOR_DIM,
        hidden_dim: int = 256,
        n_actions: int = 80,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.board_encoder = encoder if encoder is not None else build_board_encoder(encoder_name)
        board_out = int(getattr(self.board_encoder, "output_dim", 128))
        self.vector_encoder = MLPFeatureExtractor(vector_dim, hidden_dim=hidden_dim, dropout=dropout)
        fusion_layers: list[nn.Module] = [
            nn.Linear(board_out + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))
        fusion_layers.append(nn.LayerNorm(hidden_dim))
        self.fusion = nn.Sequential(*fusion_layers)
        self.head = PolicyValueHead(hidden_dim, n_actions=n_actions)
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.n_actions = n_actions

    def features(self, board: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        board_features = self.board_encoder(board)
        vector_features = self.vector_encoder(vector)
        return self.fusion(torch.cat([board_features, vector_features], dim=-1))

    def forward(
        self,
        board: torch.Tensor,
        vector: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.features(board, vector), mask)


def build_encoder(name: str, *, in_channels: int = 2) -> nn.Module:
    """建立棋盤編碼器（對外便利函式）。"""

    return build_board_encoder(name, in_channels=in_channels)


def build_network(
    name: str = "resnet",
    *,
    vector_dim: int = VECTOR_DIM,
    hidden_dim: int = 256,
    n_actions: int = 80,
    dropout: float = 0.0,
) -> TetrisNetwork:
    """依名稱建立完整網路。"""

    return TetrisNetwork(
        encoder_name=name,
        vector_dim=vector_dim,
        hidden_dim=hidden_dim,
        n_actions=n_actions,
        dropout=dropout,
    )


def count_parameters(network: nn.Module, *, trainable_only: bool = True) -> int:
    """計算參數量。"""

    params = list(network.parameters())
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def split_observation(observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """把 Dict observation 拆成 (board, vector) 兩個 tensor。"""

    board = observation["board"]
    if board.dim() == 3:
        board = board.unsqueeze(0)
    vectors = []
    for key in VECTOR_KEYS:
        value = observation[key]
        vectors.append(value.reshape(value.shape[0], -1))
    return board, torch.cat(vectors, dim=-1)
