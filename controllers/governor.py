"""APM 治理：把實際攻擊率限制在目標值附近。"""

from __future__ import annotations

from collections import deque


class APMGovernor:
    """以滾動視窗追蹤攻擊輸出，超標時要求放慢或改選低攻擊落點。"""

    def __init__(self, target_apm: float = 0.0, *, window_s: float = 12.0, max_extra_delay: float = 0.35) -> None:
        self.target_apm = float(target_apm)
        self.window_s = float(window_s)
        self.max_extra_delay = float(max_extra_delay)
        self.events: deque[tuple[float, float]] = deque()
        self.total_attack = 0

    def set_target(self, target_apm: float) -> None:
        self.target_apm = float(target_apm)

    def reset(self) -> None:
        self.events.clear()
        self.total_attack = 0

    def note_attack(self, attack: int | float, now: float) -> None:
        if attack <= 0:
            return
        self.events.append((float(now), float(attack)))
        self.total_attack += int(attack)
        self._evict(now)

    def _evict(self, now: float) -> None:
        horizon = float(now) - self.window_s
        while self.events and self.events[0][0] < horizon:
            self.events.popleft()

    def current_apm(self, now: float) -> float:
        """滾動視窗內的實際 APM。"""

        self._evict(now)
        attack = sum(value for _, value in self.events)
        return float(attack / self.window_s * 60.0)

    def over_budget(self, now: float) -> bool:
        if self.target_apm <= 0:
            return False
        return self.current_apm(now) > self.target_apm

    def extra_delay(self, now: float) -> float:
        """超出目標 APM 時要額外等待的秒數。"""

        if self.target_apm <= 0:
            return 0.0
        current = self.current_apm(now)
        if current <= self.target_apm:
            return 0.0
        ratio = (current - self.target_apm) / max(self.target_apm, 1e-6)
        return float(min(self.max_extra_delay, ratio * 0.1))

    def prefer_low_attack(self, now: float) -> bool:
        """是否應該改選攻擊較低的落點。"""

        return self.over_budget(now)
