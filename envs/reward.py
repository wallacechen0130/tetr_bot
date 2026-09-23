"""Reward System：把 StepEvents 轉成純量 reward 與各分量明細。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.engine.board import bumpiness_from_grid, holes_from_grid, max_height_from_grid
from envs.engine.events import StepEvents
from envs.engine.rules import ClearType
from envs.engine.simulator import GameSnapshot

DEFAULT_REWARD: dict[str, Any] = {
    "survival_per_step": 0.01,
    "single": 1.0,
    "double": 3.0,
    "triple": 6.0,
    "tetris": 12.0,
    "tspin_mini": 4.0,
    "tspin_single": 8.0,
    "tspin_double": 14.0,
    "tspin_triple": 20.0,
    "combo_bonus_per_level": 0.5,
    "combo_bonus_cap": 10,
    "b2b_multiplier": 1.5,
    "perfect_clear": 30.0,
    "top_out": -20.0,
    "garbage_sent_per_line": 0.4,
    "garbage_received_per_line": -0.3,
    "garbage_efficiency": 0.2,
    "new_hole": -0.5,
    "hole_state": -0.05,
    "height_state": -0.02,
    "height_threshold": 12.0,
    "bumpiness_state": -0.05,
    "schedule": {
        "survival_phase_fraction": 0.2,
        "survival_multiplier": 1.5,
        "attack_multiplier_late": 1.5,
    },
}

_LINE_CLEAR_KEYS: dict[ClearType, str] = {
    ClearType.SINGLE: "single",
    ClearType.DOUBLE: "double",
    ClearType.TRIPLE: "triple",
    ClearType.TETRIS: "tetris",
}

_TSPIN_KEYS: dict[ClearType, str] = {
    ClearType.TSPIN_MINI: "tspin_mini",
    ClearType.TSPIN_MINI_SINGLE: "tspin_mini",
    ClearType.TSPIN_MINI_DOUBLE: "tspin_mini",
    ClearType.TSPIN_MINI_TRIPLE: "tspin_mini",
    ClearType.TSPIN_SINGLE: "tspin_single",
    ClearType.TSPIN_DOUBLE: "tspin_double",
    ClearType.TSPIN_TRIPLE: "tspin_triple",
}


@dataclass
class RewardCalculator:
    """依 config 計算 reward；同時回傳各分量方便除錯與 TensorBoard 記錄。"""

    config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_REWARD))

    def __post_init__(self) -> None:
        merged = dict(DEFAULT_REWARD)
        merged.update(self.config or {})
        self.config = merged

    def set_progress(self, progress: float) -> None:
        """設定課程進度（0..1），用來在 PPO 前後段調整權重。"""

        self.progress = float(np.clip(progress, 0.0, 1.0))

    progress: float = 0.0

    # ------------------------------------------------------------------ 主計算
    def compute(
        self,
        events: StepEvents,
        before: GameSnapshot,
        after: GameSnapshot,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config
        schedule = cfg.get("schedule", {})
        phase = float(schedule.get("survival_phase_fraction", 0.2))
        in_survival_phase = self.progress < phase
        survival_scale = float(schedule.get("survival_multiplier", 1.5)) if in_survival_phase else 1.0
        attack_scale = 1.0 if in_survival_phase else float(schedule.get("attack_multiplier_late", 1.5))

        components: dict[str, float] = {}
        components["survival"] = float(cfg["survival_per_step"]) * survival_scale

        clear_value = 0.0
        if events.clear_type in _LINE_CLEAR_KEYS:
            clear_value = float(cfg[_LINE_CLEAR_KEYS[events.clear_type]])
        elif events.clear_type in _TSPIN_KEYS:
            clear_value = float(cfg[_TSPIN_KEYS[events.clear_type]])
        components["line_clear"] = clear_value

        if clear_value > 0.0 and events.b2b_active:
            components["b2b"] = clear_value * (float(cfg["b2b_multiplier"]) - 1.0)
        else:
            components["b2b"] = 0.0

        combo_level = min(events.combo, int(cfg["combo_bonus_cap"]))
        components["combo"] = float(cfg["combo_bonus_per_level"]) * max(0, combo_level - 1)
        components["perfect_clear"] = float(cfg["perfect_clear"]) if events.perfect_clear else 0.0
        components["top_out"] = float(cfg["top_out"]) if events.top_out else 0.0

        components["garbage_sent"] = (
            float(cfg["garbage_sent_per_line"]) * events.attack_sent * attack_scale
        )
        components["garbage_received"] = float(cfg["garbage_received_per_line"]) * events.garbage_applied
        components["garbage_efficiency"] = (
            float(cfg["garbage_efficiency"]) * (events.attack_sent / max(1, events.lines_cleared))
            if events.attack_sent
            else 0.0
        )

        holes_before = holes_from_grid(before.board)
        holes_after = holes_from_grid(after.board)
        height_after = max_height_from_grid(after.board)
        bumpiness_after = bumpiness_from_grid(after.board)

        components["new_hole"] = float(cfg["new_hole"]) * max(0, events.holes_created)
        components["hole_state"] = float(cfg["hole_state"]) * (holes_after - holes_before)
        components["height_state"] = (
            float(cfg["height_state"]) * max(0.0, height_after - float(cfg["height_threshold"]))
        )
        components["bumpiness_state"] = float(cfg["bumpiness_state"]) * bumpiness_after

        total = float(sum(components.values()))
        components["total"] = total
        return total, components
