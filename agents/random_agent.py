"""隨機 agent：基線用途，只在合法落點中均勻取樣。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """均勻隨機選擇合法落點。"""

    name = "random"

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        actions = self.legal_actions(info)
        if not actions:
            return 0
        return int(self.rng.choice(actions))
