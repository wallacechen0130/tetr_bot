"""Action mask 與高階動作編碼的正確性。"""

from __future__ import annotations

import gymnasium as gym

import envs  # noqa: F401
from envs.gym.placement import N_ACTIONS, action_index, decode_action


def test_action_index_roundtrip() -> None:
    env = gym.make("TetrisSurvival-v0")
    _observation, info = env.reset(seed=11)
    for index, placement in info["placements"].items():
        column, rotation, hold_used = decode_action(index)
        assert column == placement.left_column
        assert rotation == placement.rotation % 4
        assert hold_used == placement.hold_used
        assert action_index(placement) == index
    env.close()


def test_masked_actions_are_executable() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=13)
    board = env.unwrapped.sim.board
    for action in info["placements"]:
        placement = info["placements"][action]
        assert not board.collides(placement.cells)
        assert 0 <= action < N_ACTIONS
    env.close()


def test_mask_shrinks_as_board_fills() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=17)
    first_count = len(info["placements"])
    board = env.unwrapped.sim.board
    # 擋住出生列左側（rows 18-19, cols 0-2）→ 可達欄位變少
    board.cells[18:20, 0:3] = 1
    env.unwrapped._refresh_turn()
    second_count = len(env.unwrapped.current_placements)
    assert second_count < first_count
    env.close()


def test_hold_placements_present_when_hold_enabled() -> None:
    env = gym.make("TetrisSurvival-v0")
    _observation, info = env.reset(seed=19)
    assert any(placement.hold_used for placement in info["placements"].values())
    env.close()


def test_low_level_plan_compiles_all_placements() -> None:
    from envs.gym.low_level import compile_placement_plan

    env = gym.make("TetrisSurvival-v0")
    _observation, info = env.reset(seed=23)
    board = env.unwrapped.sim.board
    for placement in list(info["placements"].values())[:10]:
        plan = compile_placement_plan(board, placement.kind, placement)
        assert plan.keys[-1] == "hard_drop"
        if placement.hold_used:
            assert plan.keys[0] == "hold"
        assert set(plan.keys) <= set(
            ["left", "right", "rotate_cw", "rotate_ccw", "rotate_180", "soft_drop", "hard_drop", "hold"]
        )
    env.close()
