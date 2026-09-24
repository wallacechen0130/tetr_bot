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


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    neg_inf: float = NEG_INF,
) -> torch.Tensor:
    """只在合法動作上計算 cross-entropy（支援 label smoothing）。

    為什麼不能直接用 ``torch.nn.functional.cross_entropy(label_smoothing>0)``：
    本專案會把不合法動作的 logits 壓到 ``-1e9``，而 label smoothing 的均勻項是
    「對所有類別」取平均，會把這些 ``-1e9`` 一起平均進去。
    以實際盤面（約 47% 動作不合法）為例，loss 會從 ~3.4 爆到 ~1e7。
    這裡改成只在合法動作上取均勻分佈，並且自己套用 mask，避免呼叫端忘記。
    """

    masked = masked_logits(logits, mask, neg_inf=neg_inf)
    log_probs = torch.log_softmax(masked, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    if label_smoothing <= 0.0:
        return nll.mean()
    valid = (mask > 0).to(dtype=log_probs.dtype)
    if valid.dim() == 1:
        valid = valid[None, :]
    count = valid.sum(dim=-1).clamp(min=1.0)
    uniform = -(log_probs * valid).sum(dim=-1) / count
    loss = (1.0 - label_smoothing) * nll + label_smoothing * uniform
    return loss.mean()


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
