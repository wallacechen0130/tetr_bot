"""Gymnasium 介面層：observation 編碼、action 映射、環境本體。"""

from envs.gym.low_level import compile_placement_plan
from envs.gym.obs_encoder import ObservationEncoder
from envs.gym.placement import (
    N_ACTIONS,
    N_COLUMNS,
    N_ROTATIONS,
    action_index,
    action_mask,
    build_action_map,
    decode_action,
)
from envs.gym.tetris_env import TetrisEnv

__all__ = [
    "TetrisEnv",
    "ObservationEncoder",
    "N_ACTIONS",
    "N_COLUMNS",
    "N_ROTATIONS",
    "action_index",
    "action_mask",
    "build_action_map",
    "decode_action",
    "compile_placement_plan",
]
