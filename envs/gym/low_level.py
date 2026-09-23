"""低階按鍵計畫：把高階落點編譯成鍵盤操作序列。

MVP 不執行按鍵，只產生「意圖序列」，供未來的 TETR.IO 接入
（integrations/）與人類化操作模擬使用。
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.engine.board import Board
from envs.engine.piece import box_offsets, spawn_origin
from envs.engine.placement import Placement

LOW_LEVEL_ACTIONS: tuple[str, ...] = (
    "left",
    "right",
    "rotate_cw",
    "rotate_ccw",
    "rotate_180",
    "soft_drop",
    "hard_drop",
    "hold",
)


@dataclass(slots=True)
class KeyPlan:
    """一次落子所需的按鍵序列。"""

    keys: tuple[str, ...]

    @property
    def key_count(self) -> int:
        return len(self.keys)

    def describe(self) -> str:
        return " -> ".join(self.keys)


def compile_placement_plan(
    board: Board,
    kind: str,
    placement: Placement,
    *,
    allow_180: bool = True,
) -> KeyPlan:
    """產生「先 hold、再旋轉、再橫移、最後 Hard Drop」的按鍵計畫。

    已知限制：此序列假設方塊在出生列即可完成旋轉與橫移（多數情況成立），
    真正需要 tuck 的落點在低階執行時必須改走 BFS 路徑，
    詳見 docs/06_action_space.md。
    """

    keys: list[str] = []
    if placement.hold_used:
        keys.append("hold")

    delta = (placement.rotation % 4) % 4
    if delta == 1:
        keys.append("rotate_cw")
    elif delta == 2:
        keys.append("rotate_180" if allow_180 else "rotate_cw")
        if not allow_180:
            keys.append("rotate_cw")
    elif delta == 3:
        keys.append("rotate_ccw")

    _, origin_col = spawn_origin(kind, board.cols, board.buffer_rows)
    _, offset_col = box_offsets(kind, placement.rotation)
    target_col = placement.left_column - offset_col
    shift = target_col - origin_col
    keys.extend("right" if shift > 0 else "left" for _ in range(abs(shift)))

    keys.append("hard_drop")
    return KeyPlan(tuple(keys))


def compile_placement_keys(
    board: Board,
    kind: str,
    placement: Placement,
    *,
    allow_180: bool = True,
) -> tuple[str, ...]:
    """便利函式：直接回傳按鍵字串 tuple。"""

    return compile_placement_plan(board, kind, placement, allow_180=allow_180).keys
