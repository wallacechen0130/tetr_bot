"""Stable-Baselines3 特徵抽取器：包裝 TetrisNetwork 供 MaskablePPO 使用。"""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.gym.obs_encoder import VECTOR_KEYS
from policies.factory import TetrisNetwork, build_network


class TetrisFeaturesExtractor(BaseFeaturesExtractor):
    """把 Dict observation 轉成 embedding，policy/value 頭由 SB3 負責。"""

    def __init__(
        self,
        observation_space: Any,
        features_dim: int = 256,
        *,
        network: str = "resnet",
        vector_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, features_dim=int(features_dim))
        self.network_name = network
        vector_dim = vector_dim or self._infer_vector_dim(observation_space)
        self.tetris_network: TetrisNetwork = build_network(
            network,
            vector_dim=vector_dim,
            hidden_dim=int(features_dim),
            dropout=dropout,
        )

    @staticmethod
    def _infer_vector_dim(observation_space: Any) -> int:
        total = 0
        for key in VECTOR_KEYS:
            shape = observation_space[key].shape
            dim = 1
            for size in shape:
                dim *= int(size)
            total += dim
        return total

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        board = observations["board"]
        vectors = [observations[key].reshape(observations[key].shape[0], -1) for key in VECTOR_KEYS]
        vector = torch.cat(vectors, dim=-1)
        return self.tetris_network.features(board, vector)

    # ------------------------------------------------------------------ 權重轉移
    def load_pretrained(self, state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
        """從 IL checkpoint 載入 encoder 權重（形狀不符者跳過）。"""

        target = self.tetris_network.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if key in target and tuple(target[key].shape) == tuple(value.shape)
        }
        self.tetris_network.load_state_dict(matched, strict=False)
        return {"loaded": len(matched), "total": len(target)}
