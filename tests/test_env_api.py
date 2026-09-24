"""Gymnasium 環境 API、observation 空間與決定性測試。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401 - 匯入即完成環境註冊

ENV_IDS = ("TetrisSurvival-v0", "Tetris40L-v0", "TetrisVersus-v0")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_observation_space_and_reset(env_id: str) -> None:
    env = gym.make(env_id)
    observation, info = env.reset(seed=0)
    space = env.observation_space
    assert space.contains(observation)
    assert observation["board"].shape == (2, 20, 10)
    assert observation["board"].dtype == np.float32
    assert observation["piece_onehot"].shape == (7,)
    assert observation["hold_onehot"].shape == (8,)
    assert observation["hold_available"].shape == (1,)
    assert observation["next_queue"].shape == (7, 7)
    assert observation["combo_b2b"].shape == (4,)
    assert observation["garbage"].shape == (3,)
    assert observation["stats"].shape == (8,)
    assert observation["action_mask"].shape == (80,)
    assert observation["action_mask"].dtype == np.int8
    assert info["placements"]
    assert info["snapshot"].current in "IJLOSTZ"
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_passes_gymnasium_checker(env_id: str) -> None:
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_same_seed_is_reproducible(env_id: str) -> None:
    env = gym.make(env_id)
    first, info_first = env.reset(seed=42)
    second, info_second = env.reset(seed=42)
    assert np.array_equal(first["board"], second["board"])
    assert np.array_equal(first["action_mask"], second["action_mask"])
    assert sorted(info_first["placements"]) == sorted(info_second["placements"])
    assert info_first["snapshot"].current == info_second["snapshot"].current
    env.close()


def test_action_mask_matches_placements() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=3)
    mask = observation["action_mask"]
    assert sorted(int(i) for i in np.nonzero(mask)[0]) == sorted(info["placements"])
    assert bool(env.unwrapped.action_masks().any())
    env.close()


def test_illegal_action_falls_back_gracefully() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=5)
    legal = set(info["placements"])
    illegal = next(index for index in range(80) if index not in legal)
    observation, reward, terminated, truncated, info = env.step(illegal)
    assert info["invalid_action"] is True
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    env.close()


def test_step_signature_and_info_keys() -> None:
    env = gym.make("Tetris40L-v0")
    observation, info = env.reset(seed=1)
    action = next(iter(info["placements"]))
    observation, reward, terminated, truncated, info = env.step(action)
    for key in ("events", "snapshot", "placements", "lines_total", "pps", "apm", "mode"):
        assert key in info, f"info 缺少 {key}"
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["mode"] == "40l"
    env.close()


def test_render_returns_string_and_rgb() -> None:
    env = gym.make("TetrisSurvival-v0", render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert isinstance(text, str) and "+" in text
    env.close()

    rgb_env = gym.make("TetrisSurvival-v0", render_mode="rgb_array")
    rgb_env.reset(seed=0)
    image = rgb_env.render()
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    rgb_env.close()


def test_mode_limits_are_not_leaked_between_modes() -> None:
    """survival / versus 不可以繼承 40L 的 target_lines。"""

    survival = gym.make("TetrisSurvival-v0")
    assert survival.unwrapped.game_config.target_lines is None
    assert survival.unwrapped.game_config.time_limit == 120.0
    survival.close()

    versus = gym.make("TetrisVersus-v0")
    assert versus.unwrapped.game_config.target_lines is None
    versus.close()

    forty = gym.make("Tetris40L-v0")
    assert forty.unwrapped.game_config.target_lines == 40
    assert forty.unwrapped.game_config.time_limit is None
    forty.close()


def test_survival_does_not_end_at_forty_lines() -> None:
    env = gym.make("TetrisSurvival-v0")
    env.reset(seed=0)
    env.unwrapped.sim.lines_total = 40
    assert env.unwrapped.sim.completed() is False
    assert env.unwrapped.sim.lines_remaining() is None
    env.close()
