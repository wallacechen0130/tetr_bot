"""腳本 agent 與腳本對手模型。

ScriptedPlayer：用「固定 PPS + 固定 APM」的方式遊玩，用來模擬不同等級的對手。
ScriptedOpponent：描述對手攻擊曲線（APM、cheese 比例），由 env 的 AttackPacer 執行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.base import BaseAgent
from agents.heuristic_agent import HeuristicAgent


@dataclass
class ScriptedOpponent:
    """腳本對手的攻擊設定（垃圾行注入由 env 端執行）。"""

    name: str = "scripted"
    apm: float = 60.0
    cheese_ratio: float = 0.0
    attack_variance: float = 0.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def attack_per_piece(self, piece_time: float) -> int:
        """依每顆方塊耗時換算攻擊行數（含可選變異）。"""

        base = self.apm / 60.0 * max(piece_time, 0.0)
        if self.attack_variance > 0:
            base *= float(1.0 + self._rng.normal(0.0, self.attack_variance))
        return int(max(0.0, base) // 1)

    def as_env_kwargs(self) -> dict[str, Any]:
        return {"opponent_apm": self.apm}


class ScriptedPlayer(BaseAgent):
    """以啟發式為底、再套用固定 PPS/APM 限制的腳本玩家。"""

    name = "scripted_player"

    def __init__(
        self,
        *,
        target_pps: float = 2.0,
        target_apm: float = 60.0,
        depth: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.target_pps = float(target_pps)
        self.target_apm = float(target_apm)
        self.heuristic = HeuristicAgent(depth=depth, seed=seed)

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)
        self.heuristic.reset(seed=seed)

    def piece_time(self) -> float:
        return 1.0 / max(self.target_pps, 1e-6)

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        candidates = self.heuristic.rank(info["snapshot"], info["placements"])
        if not candidates:
            return 0
        attack_budget = self.target_apm / 60.0 * self.piece_time()
        if attack_budget < 1.0:
            # APM 受限：優先選攻擊行數較少的落點
            candidates.sort(key=lambda cand: (cand.attack, -cand.score))
        return int(candidates[0].action)
