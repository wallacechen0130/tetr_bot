"""envs 套件：Tetris 規則引擎、Gymnasium 環境與註冊。

匯入本套件即完成 Gymnasium 環境註冊：
- ``TetrisSurvival-v0``：存活模式（Blitz 風格）
- ``Tetris40L-v0``：40 行競速
- ``TetrisVersus-v0``：對上腳本對手（固定 APM 垃圾行）
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.gym.tetris_env import TetrisEnv

ENV_IDS: dict[str, dict[str, Any]] = {
    "TetrisSurvival-v0": {"mode": "survival"},
    "Tetris40L-v0": {"mode": "40l"},
    "TetrisVersus-v0": {"mode": "versus"},
}

N_ENVS = 4


def register_envs(force: bool = False) -> None:
    """註冊所有環境 id（重複註冊時可傳 force=True）。"""

    for env_id, kwargs in ENV_IDS.items():
        try:
            register(id=env_id, entry_point="envs.gym.tetris_env:TetrisEnv", kwargs=kwargs)
        except Exception:  # pragma: no cover - 已註冊時忽略
            if force:
                raise


register_envs()


def make_env(env_id: str = "TetrisSurvival-v0", **kwargs: Any) -> gym.Env:
    """建立環境（先確保已註冊）。"""

    register_envs()
    return gym.make(env_id, **kwargs)


def make_masked_env(env_id: str = "TetrisSurvival-v0", **kwargs: Any) -> gym.Env:
    """建立可回傳 action mask 的環境（供 MaskablePPO 使用）。"""

    return make_env(env_id, **kwargs)


__all__ = ["TetrisEnv", "ENV_IDS", "N_ENVS", "make_env", "make_masked_env", "register_envs"]
