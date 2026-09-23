"""動作頭：把 embedding 轉成 80 維 logits 與 state value。"""

from __future__ import annotations

import torch
from torch import nn

NEG_INF = -1.0e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor | None, *, neg_inf: float = NEG_INF) -> torch.Tensor:
    """把不合法動作的 logits 壓到極小值（mask 為 1 表示合法）。"""

    if mask is None:
        return logits
    mask = mask.to(dtype=logits.dtype)
    if mask.dim() == 1:
        mask = mask[None, :]
    return logits * mask + (1.0 - mask) * neg_inf


class PolicyValueHead(nn.Module):
    """Actor-Critic 頭：同時輸出動作 logits 與 value。"""

    def __init__(self, hidden_dim: int, n_actions: int = 80, head_dim: int | None = None) -> None:
        super().__init__()
        head_dim = head_dim or hidden_dim
        self.policy = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(inplace=True), nn.Linear(head_dim, n_actions))
        self.value = nn.Sequential(nn.Linear(hidden_dim, head_dim), nn.ReLU(inplace=True), nn.Linear(head_dim, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(features)
        if mask is not None:
            logits = masked_logits(logits, mask)
        value = self.value(features)
        return logits, value
