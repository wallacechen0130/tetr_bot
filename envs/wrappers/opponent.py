"""腳本對手 wrapper：以固定 APM 對玩家注入垃圾行。"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class ScriptedOpponentWrapper(gym.Wrapper):
    """調整內建腳本對手的攻擊率（APM）。"""

    def __init__(self, env: gym.Env, apm: float) -> None:
        super().__init__(env)
        self.set_apm(apm)

    def set_apm(self, apm: float) -> None:
        unwrapped = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(unwrapped, "set_opponent_apm"):
            unwrapped.set_opponent_apm(apm)

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)  # type: ignore[attr-defined]
        return observation, info
