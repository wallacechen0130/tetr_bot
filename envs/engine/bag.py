"""7-bag 隨機袋（seed 可重現）。"""

from __future__ import annotations

from collections import deque

import numpy as np

from envs.engine.piece import PIECE_KINDS


class SevenBag:
    """標準 7-bag：每袋包含七種方塊各一次，袋內順序隨機。"""

    def __init__(self, rng: np.random.Generator, queue_length: int = 14) -> None:
        self.rng = rng
        self.queue: deque[str] = deque()
        self.queue_length = int(queue_length)
        self.fill()

    def _new_bag(self) -> list[str]:
        bag = list(PIECE_KINDS)
        self.rng.shuffle(bag)
        return bag

    def fill(self) -> None:
        while len(self.queue) < self.queue_length:
            self.queue.extend(self._new_bag())

    def next(self) -> str:
        self.fill()
        return self.queue.popleft()

    def peek(self, count: int) -> tuple[str, ...]:
        self.fill()
        items = list(self.queue)[:count]
        while len(items) < count:
            items.append(PIECE_KINDS[0])
        return tuple(items)

    def reset(self) -> None:
        self.queue.clear()
        self.fill()
