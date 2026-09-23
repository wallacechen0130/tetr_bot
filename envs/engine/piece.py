"""方塊幾何與 SRS 旋轉狀態定義。

座標系：``row`` 由上往下遞增、``col`` 由左往右遞增；
每個方塊都放在正方形的 bounding box 中（I/O 為 4x4，其餘為 3x3），
如此 SRS 的 kick 表才能直接套用。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Literal

PieceKind = Literal["I", "J", "L", "O", "S", "T", "Z"]

PIECE_KINDS: tuple[str, ...] = ("I", "J", "L", "O", "S", "T", "Z")
PIECE_INDEX: dict[str, int] = {kind: idx for idx, kind in enumerate(PIECE_KINDS)}

BOX_SIZE: dict[str, int] = {"I": 4, "J": 3, "L": 3, "O": 4, "S": 3, "T": 3, "Z": 3}

# 顏色編碼（board 上使用的整數值）：1..7 對應 PIECE_KINDS，8 代表垃圾行
PIECE_VALUE: dict[str, int] = {kind: idx + 1 for idx, kind in enumerate(PIECE_KINDS)}
GARBAGE_VALUE: int = 8
EMPTY: int = 0

# rotation 0（spawn）佔用的格子，座標是 bounding box 內的 (row, col)
_BASE_CELLS: dict[str, tuple[tuple[int, int], ...]] = {
    "I": ((1, 0), (1, 1), (1, 2), (1, 3)),
    "J": ((0, 0), (1, 0), (1, 1), (1, 2)),
    "L": ((0, 2), (1, 0), (1, 1), (1, 2)),
    "O": ((1, 1), (1, 2), (2, 1), (2, 2)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "T": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
}


def _rotate_cw(cells: Iterable[tuple[int, int]], size: int) -> tuple[tuple[int, int], ...]:
    """矩陣順時針旋轉：(r, c) -> (c, size - 1 - r)。"""

    return tuple(sorted((c, size - 1 - r) for r, c in cells))


@cache
def piece_rotations(kind: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    """回傳某方塊四種旋轉狀態的格子集合（O 的四種狀態相同）。"""

    if kind not in _BASE_CELLS:
        raise KeyError(f"未知方塊：{kind}")
    size = BOX_SIZE[kind]
    base = _BASE_CELLS[kind]
    if kind == "O":
        return (base, base, base, base)
    states = [tuple(sorted(base))]
    for _ in range(3):
        states.append(_rotate_cw(states[-1], size))
    return tuple(states)


def piece_cells(
    kind: str,
    rotation: int,
    row: int,
    col: int,
) -> tuple[tuple[int, int], ...]:
    """把 bounding box 座標轉成棋盤上的絕對座標。"""

    state = piece_rotations(kind)[rotation % 4]
    return tuple((row + r, col + c) for r, c in state)


def spawn_origin(kind: str, cols: int, buffer_rows: int) -> tuple[int, int]:
    """回傳出生時 bounding box 左上角座標。

    3x3 方塊佔用 buffer 最下方兩列，4x4（I/O）佔用最下方一列，
    相當於 Guideline 的「出生於可見區上方」行為。
    """

    size = BOX_SIZE[kind]
    row = buffer_rows - 2
    col = (cols - size) // 2
    return row, col


def normalize_rotation(kind: str, rotation: int) -> int:
    """把任意旋轉索引正規化到 0..3。"""

    return rotation % 4


@cache
def box_offsets(kind: str, rotation: int) -> tuple[int, int]:
    """回傳該旋轉狀態中，最左上佔用格相對 bounding box 的位移。"""

    cells = piece_rotations(kind)[rotation % 4]
    return min(r for r, _ in cells), min(c for _, c in cells)


def piece_from_cells(
    kind: str,
    rotation: int,
    cells: tuple[tuple[int, int], ...],
) -> Piece:
    """由絕對座標反推 bounding box 原點，還原出 Piece 物件。"""

    offset_row, offset_col = box_offsets(kind, rotation)
    row = min(r for r, _ in cells) - offset_row
    col = min(c for _, c in cells) - offset_col
    return Piece(kind, rotation % 4, row, col)


@dataclass(frozen=True, slots=True)
class Piece:
    """單一 falling piece 的狀態。"""

    kind: str
    rotation: int = 0
    row: int = 0
    col: int = 0

    def cells(self) -> tuple[tuple[int, int], ...]:
        return piece_cells(self.kind, self.rotation, self.row, self.col)

    def moved(self, d_row: int = 0, d_col: int = 0) -> Piece:
        return Piece(self.kind, self.rotation, self.row + d_row, self.col + d_col)

    def rotated(self, rotation: int, row: int | None = None, col: int | None = None) -> Piece:
        return Piece(
            self.kind,
            rotation % 4,
            self.row if row is None else row,
            self.col if col is None else col,
        )

    def left_column(self) -> int:
        return min(c for _, c in self.cells())

    def right_column(self) -> int:
        return max(c for _, c in self.cells())

    def bottom_row(self) -> int:
        return max(r for r, _ in self.cells())

    def as_tuple(self) -> tuple[str, int, int, int]:
        return (self.kind, self.rotation, self.row, self.col)
