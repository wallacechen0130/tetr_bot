"""打法風格偏好：T-Spin / Perfect Clear / Combo / Hold / 攻擊的重排序。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 只為型別檢查
    from agents.heuristic_agent import Candidate


@dataclass(frozen=True)
class StyleWeights:
    """各風格偏好的強度（0 = 不影響）。"""

    tspin: float = 0.0
    perfect_clear: float = 0.0
    combo: float = 0.0
    hold: float = 0.0
    attack: float = 0.0

    @classmethod
    def from_profile(cls, profile: Any) -> StyleWeights:
        return cls(
            tspin=float(getattr(profile, "tspin_preference", 0.0)),
            perfect_clear=float(getattr(profile, "pc_preference", 0.0)),
            combo=float(getattr(profile, "combo_preference", 0.0)),
            hold=float(getattr(profile, "hold_usage", 0.0)),
            # 高階玩家偏好主動攻擊，低階玩家幾乎不為攻擊改變選擇
            attack=(
                0.6 * float(getattr(profile, "combo_preference", 0.0))
                + 0.5 * float(getattr(profile, "tspin_preference", 0.0))
                + 0.3 * float(getattr(profile, "pc_preference", 0.0))
            ),
        )


class StyleBias:
    """把風格偏好轉成可加到候選分數上的調整量。"""

    def __init__(self, weights: StyleWeights | None = None, *, scale: float = 1.0) -> None:
        self.weights = weights or StyleWeights()
        self.scale = float(scale)

    def set_weights(self, weights: StyleWeights) -> None:
        self.weights = weights

    def adjust(self, candidate: Candidate) -> float:
        """回傳風格調整分數（會與原始分數相加）。"""

        weights = self.weights
        base = abs(float(candidate.score)) or 1.0
        adjustment = 0.0
        if candidate.is_tspin:
            adjustment += weights.tspin * base * 0.25
        if candidate.perfect_clear:
            adjustment += weights.perfect_clear * base * 0.25
        if candidate.lines_cleared >= 2 and candidate.features.get("combo_bonus", 0.0) > 0:
            adjustment += weights.combo * base * 0.15
        if candidate.uses_hold:
            adjustment += (weights.hold - 0.5) * base * 0.05
        if candidate.attack > 0:
            adjustment += weights.attack * candidate.attack * (1.0 + 0.08 * base)
        return float(adjustment) * self.scale

    def rerank(self, candidates: list[Candidate]) -> tuple[list[Candidate], np.ndarray]:
        """回傳 (候選清單, 調整後分數)。順序維持原樣，由 controller 排序。"""

        adjusted = np.array([candidate.score + self.adjust(candidate) for candidate in candidates], dtype=np.float64)
        return candidates, adjusted
