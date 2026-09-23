"""高階放置動作空間的編碼與解碼。

動作索引：``index = ((column * 4) + rotation) * 2 + hold``
- ``column``：落點最左側佔用格的欄位（0..9）
- ``rotation``：方塊最終旋轉狀態（0..3）
- ``hold``：是否先使用 hold（0/1）

共 80 個離散動作；不合法的 (column, rotation, hold) 組合由 action mask 遮蔽。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from envs.engine.placement import Placement

N_COLUMNS = 10
N_ROTATIONS = 4
N_HOLD = 2
N_ACTIONS = N_COLUMNS * N_ROTATIONS * N_HOLD  # 80


def action_index(placement: Placement) -> int:
    """把落點轉成動作索引。"""

    column = placement.left_column
    if not 0 <= column < N_COLUMNS:
        raise ValueError(f"落點欄位超出範圍：{column}")
    return ((column * N_ROTATIONS) + (placement.rotation % N_ROTATIONS)) * N_HOLD + int(placement.hold_used)


def decode_action(action: int) -> tuple[int, int, bool]:
    """把動作索引拆成 (column, rotation, hold_used)。"""

    if not 0 <= int(action) < N_ACTIONS:
        raise ValueError(f"動作索引超出範圍：{action}")
    value = int(action)
    hold_used = bool(value % N_HOLD)
    value //= N_HOLD
    rotation = value % N_ROTATIONS
    column = value // N_ROTATIONS
    return column, rotation, hold_used


def build_action_map(placements: Iterable[Placement]) -> dict[int, Placement]:
    """把落點集合轉成 {action_index: Placement}。

    同一個 (column, rotation, hold) 可能對應多個落點（例如原地踢牆後再落下），
    此時優先保留「直線落下」的落點，kick 落點只用來填補尚未被佔用的索引。
    """

    ordered = sorted(placements, key=lambda p: (p.used_kick, p.rotation, p.left_column, p.cells))
    action_map: dict[int, Placement] = {}
    for placement in ordered:
        index = action_index(placement)
        action_map.setdefault(index, placement)
    return action_map


def action_mask(action_map: dict[int, Placement]) -> np.ndarray:
    """由 action map 產生 0/1 mask（長度 80）。"""

    mask = np.zeros(N_ACTIONS, dtype=np.int8)
    for index in action_map:
        mask[index] = 1
    return mask


def mask_to_bool(mask: np.ndarray) -> np.ndarray:
    """轉成 sb3-contrib 需要的 bool mask。"""

    return np.asarray(mask, dtype=bool).reshape(-1)
