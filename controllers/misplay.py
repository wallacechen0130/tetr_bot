"""操作失誤模型：依排名機率挑選次佳落點。"""

from __future__ import annotations

import numpy as np


class MisplayModel:
    """以 ``chance`` 的機率做出非最佳選擇。"""

    def __init__(self, chance: float = 0.0, *, temperature: float = 1.0, seed: int | None = None) -> None:
        self.chance = float(np.clip(chance, 0.0, 1.0))
        self.temperature = max(float(temperature), 1e-3)
        self.rng = np.random.default_rng(seed)
        self.mistakes = 0
        self.decisions = 0

    def set_chance(self, chance: float) -> None:
        self.chance = float(np.clip(chance, 0.0, 1.0))

    def choose(self, ranked_scores: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """回傳排名位置（0 = 最佳）。``ranked_scores`` 需已由高到低排序。"""

        rng = rng or self.rng
        scores = np.asarray(ranked_scores, dtype=np.float64).reshape(-1)
        self.decisions += 1
        if scores.size <= 1 or self.chance <= 0.0:
            return 0
        if float(rng.random()) >= self.chance:
            return 0
        self.mistakes += 1
        rest = scores[1:]
        spread = max(1e-6, float(np.std(scores)))
        scaled = (rest - rest.max()) / (self.temperature * spread)
        weights = np.exp(scaled)
        weights = weights / weights.sum()
        return int(rng.choice(np.arange(1, scores.size), p=weights))
