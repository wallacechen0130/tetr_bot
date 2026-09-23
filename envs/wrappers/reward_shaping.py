"""依課程進度動態調整 reward 權重。"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class RewardScheduleWrapper(gym.Wrapper):
    """把環境進度餵給 RewardCalculator，實現 survival → attack 的課程。"""

    def __init__(self, env: gym.Env, phase_fraction: float | None = None) -> None:
        super().__init__(env)
        self.phase_fraction = phase_fraction

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)  # type: ignore[attr-defined]
        self._sync_progress(info)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)  # type: ignore[attr-defined]
        self._sync_progress(info)
        return observation, reward, terminated, truncated, info

    def _sync_progress(self, info: dict[str, Any]) -> None:
        env = self.env.unwrapped  # type: ignore[attr-defined]
        progress = getattr(env, "_progress", None)
        if callable(progress) and hasattr(env, "reward"):
            env.reward.set_progress(progress())
        if self.phase_fraction is not None and hasattr(env, "reward"):
            env.reward.config.setdefault("schedule", {})["survival_phase_fraction"] = self.phase_fraction
