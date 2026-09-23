"""Gymnasium 環境：高階放置動作空間的 Tetris 環境。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.config import load_config
from envs.engine.placement import Placement
from envs.engine.rules import Ruleset, load_ruleset
from envs.engine.simulator import GameConfig, TetrisSimulator
from envs.gym.obs_encoder import ObservationEncoder
from envs.gym.placement import (
    N_ACTIONS,
    action_mask,
    build_action_map,
    mask_to_bool,
)
from envs.render.rgb_array import board_to_rgb
from envs.reward import RewardCalculator

MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "survival": {"time_limit": 120.0, "target_lines": None},
    "40l": {"time_limit": None, "target_lines": 40},
    "versus": {"time_limit": 120.0, "target_lines": None},
}


@lru_cache(maxsize=4)
def _cached_config(path: str) -> dict[str, Any]:
    return load_config(path)


class AttackPacer:
    """把目標 APM 轉成「每顆方塊要送幾行」的整數攻擊。"""

    def __init__(self, apm: float = 0.0, seed: int | None = None) -> None:
        self.apm = float(apm)
        self.rng = np.random.default_rng(seed)
        self._carry = 0.0

    def reset(self) -> None:
        self._carry = 0.0

    def attack_for(self, dt: float) -> int:
        if self.apm <= 0 or dt <= 0:
            return 0
        self._carry += self.apm / 60.0 * dt
        whole = int(np.floor(self._carry))
        self._carry -= whole
        return whole


class TetrisEnv(gym.Env):
    """TETR.IO 風格 Tetris 環境。

    動作空間：``Discrete(80)``（見 envs/gym/placement.py）
    Observation：Dict，包含棋盤影像、方塊資訊、垃圾行資訊與 action mask
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        mode: str = "survival",
        seed: int | None = None,
        render_mode: str | None = None,
        opponent_apm: float | None = None,
        piece_time: float | None = None,
        reward_config: dict[str, Any] | None = None,
        config_path: str = "configs/default.yaml",
        rules_path: str = "configs/rules_tetrio.yaml",
        max_pieces: int = 5000,
    ) -> None:
        super().__init__()
        self.config_data = config if config is not None else _cached_config(config_path)
        rules = self.config_data.get("rules")
        self.rules: Ruleset = rules if isinstance(rules, Ruleset) else load_ruleset(rules_path)

        mode_overrides = dict(MODE_DEFAULTS.get(mode, MODE_DEFAULTS["survival"]))
        if mode == "40l":
            mode_overrides["target_lines"] = int(self.config_data.get("env", {}).get("target_lines", 40))
        overrides: dict[str, Any] = {"mode": mode, **mode_overrides}
        if opponent_apm is not None:
            overrides["opponent_apm"] = opponent_apm

        self.game_config = GameConfig.from_dict(self.config_data, rules=self.rules, overrides=overrides)
        self._seed = seed if seed is not None else int(self.config_data.get("seed", 0))
        self.game_config.seed = self._seed
        if piece_time is not None:
            self.game_config.default_piece_time = float(piece_time)

        self.sim = TetrisSimulator(self.game_config, seed=self._seed)
        self.encoder = ObservationEncoder(
            visible_rows=self.rules.visible_rows,
            cols=self.rules.cols,
            queue_rows=max(self.game_config.next_count, 7),
        )
        self.observation_space = self.encoder.observation_space()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.reward = RewardCalculator(reward_config or self.config_data.get("reward", {}))
        self.opponent = AttackPacer(self.game_config.opponent_apm, seed=self._seed)
        self.max_pieces = int(max_pieces)
        self.render_mode = render_mode

        self._action_map: dict[int, Placement] = {}
        self._mask: np.ndarray = np.zeros(N_ACTIONS, dtype=np.int8)
        self._refresh_turn()

    # ------------------------------------------------------------------ 內部工具
    def _refresh_turn(self) -> None:
        placements = self.sim.legal_placements()
        self._action_map = build_action_map(placements)
        self._mask = action_mask(self._action_map)

    def _piece_time(self) -> float:
        return float(self.game_config.default_piece_time)

    def _progress(self) -> float:
        if self.game_config.time_limit:
            return float(np.clip(self.sim.elapsed / self.game_config.time_limit, 0.0, 1.0))
        if self.game_config.target_lines:
            return float(np.clip(self.sim.lines_total / self.game_config.target_lines, 0.0, 1.0))
        return float(np.clip(self.sim.pieces_placed / 300.0, 0.0, 1.0))

    def _info(self, *, invalid_action: bool = False, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = max(self.sim.elapsed, 1e-6)
        info: dict[str, Any] = {
            "snapshot": self.sim.snapshot(),
            "placements": dict(self._action_map),
            "action_mask": self._mask.copy(),
            "lines_total": self.sim.lines_total,
            "lines_remaining": self.sim.lines_remaining(),
            "pieces": self.sim.pieces_placed,
            "elapsed": self.sim.elapsed,
            "pps": self.sim.pieces_placed / elapsed,
            "apm": self.sim.attack_sent / elapsed * 60.0,
            "attack_sent": self.sim.attack_sent,
            "garbage_received": self.sim.garbage_received,
            "top_out": self.sim.top_out,
            "invalid_action": invalid_action,
            "mode": self.game_config.mode,
        }
        if extra:
            info.update(extra)
        return info

    # ------------------------------------------------------------------ API
    def set_piece_time(self, seconds: float) -> None:
        """設定每顆方塊耗時（用來模擬目標 PPS）。"""

        self.game_config.default_piece_time = float(max(seconds, 0.0))

    def set_opponent_apm(self, apm: float) -> None:
        self.opponent.apm = float(max(apm, 0.0))

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO 需要的 bool mask。"""

        return mask_to_bool(self._mask)

    @property
    def current_placements(self) -> dict[int, Placement]:
        return self._action_map

    def legal_action_indices(self) -> list[int]:
        return sorted(self._action_map)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.sim.reset(seed if seed is not None else self._seed)
        self.opponent.reset()
        self.reward.set_progress(0.0)
        self._refresh_turn()
        observation = self.encoder.encode(self.sim.snapshot(), self._mask)
        return observation, self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = int(action)
        placement = self._action_map.get(action)
        invalid_action = placement is None
        if placement is None:
            if not self._action_map:
                observation = self.encoder.encode(self.sim.snapshot(), self._mask)
                reward, _ = self.reward.compute(
                    _topout_events(),
                    self.sim.snapshot(),
                    self.sim.snapshot(),
                )
                info = self._info(invalid_action=True, extra={"events": _topout_events().to_dict()})
                return observation, float(reward), True, False, info
            placement = self._action_map[min(self._action_map)]

        before = self.sim.snapshot()
        dt = self._piece_time()
        result = self.sim.step_placement(placement, time_cost=dt)

        if self.opponent.apm > 0 and not result.top_out:
            injected = self.opponent.attack_for(dt)
            if injected:
                self.sim.receive_garbage(injected)

        self.reward.set_progress(self._progress())
        reward, components = self.reward.compute(result.events, before, result.snapshot)

        self._refresh_turn()
        observation = self.encoder.encode(self.sim.snapshot(), self._mask)

        terminated = bool(result.top_out or result.completed)
        truncated = bool((not terminated) and (self.sim.timed_out() or self.sim.pieces_placed >= self.max_pieces))
        info = self._info(
            invalid_action=invalid_action,
            extra={
                "events": result.events.to_dict(),
                "reward_components": components,
                "reward": reward,
                "cleared_rows": result.info.get("cleared_rows", []),
            },
        )
        return observation, float(reward), terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        from envs.render.ascii import render_board_ascii

        if self.render_mode == "rgb_array":
            return board_to_rgb(self.sim.board)
        if self.render_mode == "ansi" or self.render_mode is None:
            return render_board_ascii(self.sim.board)
        return None

    def close(self) -> None:  # pragma: no cover - 無需釋放資源
        return None


def _topout_events():  # pragma: no cover - 少見的極端分支
    from envs.engine.events import StepEvents

    return StepEvents(top_out=True)
