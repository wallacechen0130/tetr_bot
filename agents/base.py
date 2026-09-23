"""Agent 介面（Protocol + ABC），所有 agent 都吃同樣的 obs/info。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """最小 agent 介面：給定 observation 與 info，回傳離散動作索引。"""

    name: str

    def reset(self, *, seed: int | None = None) -> None:  # pragma: no cover - protocol
        ...

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:  # pragma: no cover
        ...


class BaseAgent(ABC):
    """提供種子與選取工具的共同基底。"""

    name: str = "base"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        """回傳動作索引。"""

    @staticmethod
    def legal_actions(info: dict[str, Any]) -> list[int]:
        placements = info.get("placements")
        if placements:
            return sorted(int(key) for key in placements)
        mask = info.get("action_mask")
        if mask is not None:
            return [int(i) for i, value in enumerate(np.asarray(mask).reshape(-1)) if value]
        return []

    def __repr__(self) -> str:  # pragma: no cover - 便於除錯
        return f"{self.__class__.__name__}(seed={self.seed})"
