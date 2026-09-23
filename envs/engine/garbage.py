"""垃圾行管理：接收、抵消（offset）、落下與洞的產生。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.engine.board import Board


@dataclass(slots=True)
class GarbageConfig:
    rows_per_batch: int = 1
    delay_pieces: int = 0
    cheese_ratio: float = 0.0
    hole_pattern: str = "random"
    max_pending: int = 20


@dataclass(slots=True)
class _Batch:
    amount: int
    delay: int
    cheese: bool
    hole_col: int


class GarbageManager:
    """管理待落下的垃圾行佇列。

    規則：玩家送出的攻擊先抵消待落下的垃圾，剩下的才送到對手；
    收到垃圾時記錄延遲，延遲結束後一次落下並產生洞。
    """

    def __init__(self, config: GarbageConfig | None = None, rng: np.random.Generator | None = None) -> None:
        self.config = config or GarbageConfig()
        self.rng = rng or np.random.default_rng()
        self.batches: list[_Batch] = []
        self.total_received = 0
        self.total_applied = 0
        self.total_sent = 0

    # ------------------------------------------------------------------ 查詢
    @property
    def pending(self) -> int:
        return sum(batch.amount for batch in self.batches)

    @property
    def ready(self) -> int:
        return sum(batch.amount for batch in self.batches if batch.delay <= 0)

    def reset(self) -> None:
        self.batches.clear()
        self.total_received = 0
        self.total_applied = 0
        self.total_sent = 0

    # ------------------------------------------------------------------ 操作
    def receive(self, amount: int, *, cheese: bool | None = None) -> int:
        """收到來自對手的垃圾行，回傳實際入列數量。"""

        if amount <= 0:
            return 0
        is_cheese = self.rng.random() < self.config.cheese_ratio if cheese is None else cheese
        hole_col = int(self.rng.integers(0, 10))
        allowed = max(0, self.config.max_pending - self.pending)
        amount = min(int(amount), allowed)
        if amount:
            self.batches.append(
                _Batch(amount=amount, delay=self.config.delay_pieces, cheese=is_cheese, hole_col=hole_col)
            )
            self.total_received += amount
        return amount

    def offset(self, attack: int) -> tuple[int, int]:
        """用攻擊抵消待落下的垃圾，回傳 (實際抵消數, 剩餘攻擊)。"""

        remaining = int(attack)
        cancelled = 0
        while remaining > 0 and self.batches:
            batch = self.batches[0]
            used = min(batch.amount, remaining)
            batch.amount -= used
            remaining -= used
            cancelled += used
            if batch.amount <= 0:
                self.batches.pop(0)
        if remaining > 0:
            self.total_sent += remaining
        return cancelled, remaining

    def advance(self, pieces: int = 1) -> None:
        """每鎖定一顆方塊呼叫一次，遞減延遲計數。"""

        for batch in self.batches:
            if batch.delay > 0:
                batch.delay -= pieces

    def apply(self, board: Board) -> int:
        """把所有已就緒的垃圾行加到棋盤底部，回傳落下列數。"""

        applied = 0
        for batch in list(self.batches):
            if batch.delay > 0 or batch.amount <= 0:
                continue
            rows = batch.amount
            for _ in range(rows):
                hole = batch.hole_col if (batch.cheese or self.config.hole_pattern == "same_hole") else int(
                    self.rng.integers(0, board.cols)
                )
                board.add_garbage_row(int(hole))
            applied += rows
            self.batches.remove(batch)
        self.total_applied += applied
        return applied
