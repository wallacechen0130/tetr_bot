"""時間模型：時鐘、token bucket（PPS 控制）與人類反應時間。"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class Clock(Protocol):
    """時間來源（可替換成假時鐘以便測試）。"""

    def now(self) -> float:  # pragma: no cover - protocol
        ...


class SystemClock:
    """以 ``time.monotonic`` 為基礎的真實時鐘。"""

    def now(self) -> float:
        return time.monotonic()


class FakeClock:
    """可手動推進的假時鐘（測試與確定性模擬用）。"""

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        self._now += float(max(0.0, seconds))
        return self._now

    def set(self, value: float) -> None:
        self._now = float(value)


class TokenBucket:
    """限制每秒可執行的動作數（PPS）。

    ``rate`` 為目標 PPS，``capacity`` 為允許的瞬時爆發量。
    """

    def __init__(self, rate: float, *, capacity: float = 1.0, start: float = 0.0) -> None:
        self.rate = max(float(rate), 1e-6)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self._last = float(start)

    def set_rate(self, rate: float) -> None:
        self.rate = max(float(rate), 1e-6)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, float(now) - self._last)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self._last = float(now)

    def wait_time(self, now: float, amount: float = 1.0) -> float:
        """還需要等幾秒才能消耗 ``amount`` 個 token。"""

        self._refill(now)
        if self.tokens + 1e-9 >= amount:
            return 0.0
        return float((amount - self.tokens) / self.rate)

    def consume(self, now: float, amount: float = 1.0) -> bool:
        """嘗試消耗 token；浮點誤差容忍 1e-9，避免節奏被基準延遲插隊。"""

        if self.wait_time(now, amount) > 0.0:
            return False
        self.tokens = max(0.0, self.tokens - amount)
        return True

    def reset(self, tokens: float | None = None) -> None:
        self.tokens = self.capacity if tokens is None else float(tokens)


class ReactionModel:
    """人類反應與思考時間模型（含抖動）。"""

    def __init__(
        self,
        reaction_ms: float = 200.0,
        think_ms: float = 80.0,
        jitter_ms: float = 25.0,
        *,
        seed: int | None = None,
    ) -> None:
        self.reaction_ms = float(reaction_ms)
        self.think_ms = float(think_ms)
        self.jitter_ms = float(jitter_ms)
        self.rng = np.random.default_rng(seed)

    def set_profile(self, reaction_ms: float, think_ms: float, jitter_ms: float | None = None) -> None:
        self.reaction_ms = float(reaction_ms)
        self.think_ms = float(think_ms)
        if jitter_ms is not None:
            self.jitter_ms = float(jitter_ms)

    def delay_seconds(self, *, first_piece: bool = False) -> float:
        """回傳這顆方塊的決策延遲（秒）。"""

        base = (self.reaction_ms + self.think_ms) / 1000.0
        if first_piece:
            base += self.reaction_ms / 1000.0
        if self.jitter_ms > 0:
            base += abs(float(self.rng.normal(0.0, self.jitter_ms / 1000.0)))
        return float(max(0.0, base))
