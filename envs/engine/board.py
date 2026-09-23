"""棋盤資料結構與形狀特徵計算。"""

from __future__ import annotations

import numpy as np

from envs.engine.piece import EMPTY, GARBAGE_VALUE, PIECE_VALUE


def heights_from_grid(grid: np.ndarray) -> np.ndarray:
    """由 numpy 棋盤計算每欄高度（以底部為 0 起算，含 buffer 區）。"""

    total_rows, cols = grid.shape
    heights = np.zeros(cols, dtype=np.int32)
    for col in range(cols):
        occupied = np.nonzero(grid[:, col])[0]
        heights[col] = 0 if occupied.size == 0 else total_rows - int(occupied[0])
    return heights


def max_height_from_grid(grid: np.ndarray) -> int:
    return int(heights_from_grid(grid).max(initial=0))


def aggregate_height_from_grid(grid: np.ndarray) -> int:
    return int(heights_from_grid(grid).sum())


def bumpiness_from_grid(grid: np.ndarray) -> int:
    heights = heights_from_grid(grid)
    if heights.size < 2:
        return 0
    return int(np.abs(np.diff(heights)).sum())


def holes_from_grid(grid: np.ndarray) -> int:
    """被佔用格擋在上方、且本身為空的格子數量。"""

    total_rows, cols = grid.shape
    heights = heights_from_grid(grid)
    total = 0
    for col in range(cols):
        if heights[col] == 0:
            continue
        top = total_rows - heights[col]
        total += int((grid[top:, col] == EMPTY).sum())
    return total


class Board:
    """Tetris 棋盤。

    ``cells`` 形狀為 ``(buffer_rows + visible_rows, cols)``，row 0 在最上方。
    可見區是最後 ``visible_rows`` 列，上方 buffer 用來讓方塊出生與 top-out 判定。
    """

    __slots__ = ("visible_rows", "cols", "buffer_rows", "total_rows", "cells")

    def __init__(self, visible_rows: int = 20, cols: int = 10, buffer_rows: int = 20) -> None:
        self.visible_rows = int(visible_rows)
        self.cols = int(cols)
        self.buffer_rows = int(buffer_rows)
        self.total_rows = self.buffer_rows + self.visible_rows
        self.cells = np.zeros((self.total_rows, self.cols), dtype=np.int8)

    # ------------------------------------------------------------------ 基本操作
    def reset(self) -> None:
        self.cells.fill(EMPTY)

    def copy(self) -> Board:
        clone = Board(self.visible_rows, self.cols, self.buffer_rows)
        clone.cells = self.cells.copy()
        return clone

    @property
    def visible(self) -> np.ndarray:
        return self.cells[self.buffer_rows :]

    @property
    def garbage_mask(self) -> np.ndarray:
        return self.cells[self.buffer_rows :] == GARBAGE_VALUE

    def is_empty(self) -> bool:
        return not bool(self.cells.any())

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.total_rows and 0 <= col < self.cols

    def occupied(self, row: int, col: int) -> bool:
        """格子是否被佔用；左右越界視為佔用、上下越界依規則處理。"""

        if col < 0 or col >= self.cols:
            return True
        if row < 0:
            return False
        if row >= self.total_rows:
            return True
        return bool(self.cells[row, col] != EMPTY)

    def collides(self, cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
        """方塊是否與牆壁、地板或其他方塊衝突。"""

        for row, col in cells:
            if col < 0 or col >= self.cols:
                return True
            if row >= self.total_rows:
                return True
            if row < 0:
                continue
            if self.cells[row, col] != EMPTY:
                return True
        return False

    def lock(self, cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], value: int) -> None:
        for row, col in cells:
            if 0 <= row < self.total_rows and 0 <= col < self.cols:
                self.cells[row, col] = value

    def lock_piece(self, kind: str, cells: tuple[tuple[int, int], ...]) -> None:
        self.lock(cells, PIECE_VALUE[kind])

    def add_garbage_row(self, hole_col: int, value: int = GARBAGE_VALUE) -> None:
        """把整個棋盤上移一列並在底部加入一列垃圾（洞在 hole_col）。"""

        self.cells[:-1] = self.cells[1:]
        self.cells[-1] = value
        self.cells[-1, hole_col] = EMPTY

    # ------------------------------------------------------------------ 消行
    def full_rows(self) -> list[int]:
        return [row for row in range(self.total_rows) if bool((self.cells[row] != EMPTY).all())]

    def clear_lines(self) -> list[int]:
        """消除滿列並回傳被消除的 row 索引（由小到大）。"""

        cleared = self.full_rows()
        if not cleared:
            return cleared
        keep = np.array([r for r in range(self.total_rows) if r not in set(cleared)], dtype=np.int64)
        remaining = self.cells[keep]
        self.cells.fill(EMPTY)
        self.cells[self.total_rows - remaining.shape[0] :] = remaining
        return cleared

    def is_perfect_clear(self) -> bool:
        return not bool(self.cells.any())

    # ------------------------------------------------------------------ 形狀特徵
    def column_heights(self) -> np.ndarray:
        """每欄高度（以底部為 0 起算，含 buffer 區）。"""

        return heights_from_grid(self.cells)

    def max_height(self) -> int:
        return max_height_from_grid(self.cells)

    def aggregate_height(self) -> int:
        return aggregate_height_from_grid(self.cells)

    def bumpiness(self) -> int:
        return bumpiness_from_grid(self.cells)

    def holes(self) -> int:
        """被佔用格擋在上方、且本身為空的格子數量。"""

        return holes_from_grid(self.cells)

    def row_transitions(self) -> int:
        """水平方向的佔用狀態轉換數（左右牆視為佔用）。"""

        filled = self.cells != EMPTY
        padded = np.ones((self.total_rows, self.cols + 2), dtype=bool)
        padded[:, 1:-1] = filled
        return int((padded[:, 1:] != padded[:, :-1]).sum())

    def column_transitions(self) -> int:
        """垂直方向的佔用狀態轉換數（地板視為佔用、天花板視為空）。"""

        filled = self.cells != EMPTY
        padded = np.zeros((self.total_rows + 1, self.cols), dtype=bool)
        padded[: self.total_rows] = filled
        padded[self.total_rows] = True
        return int((padded[1:] != padded[:-1]).sum())

    def well_sums(self) -> int:
        """well（兩側較高且自身為空）深度累加值。"""

        filled = self.cells != EMPTY
        total = 0
        for col in range(self.cols):
            depth = 0
            for row in range(self.total_rows):
                if filled[row, col]:
                    depth = 0
                    continue
                left_blocked = col == 0 or filled[row, col - 1]
                right_blocked = col == self.cols - 1 or filled[row, col + 1]
                if left_blocked and right_blocked:
                    depth += 1
                    total += depth
                else:
                    depth = 0
        return total

    def hole_columns(self) -> list[int]:
        """每個洞所在的欄位（用於形狀分析）。"""

        result: list[int] = []
        heights = self.column_heights()
        for col in range(self.cols):
            if heights[col] == 0:
                continue
            top = self.total_rows - heights[col]
            for row in range(top, self.total_rows):
                if self.cells[row, col] == EMPTY:
                    result.append(col)
        return result

    def feature_vector(self) -> np.ndarray:
        """回傳固定長度 8 維形狀特徵：高度、洞、bumpiness 等。"""

        heights = self.column_heights()
        return np.array(
            [
                float(heights.max(initial=0)),
                float(heights.sum()),
                float(self.bumpiness()),
                float(self.holes()),
                float(self.well_sums()),
                float(self.row_transitions()),
                float(self.column_transitions()),
                float(len(self.hole_columns())),
            ],
            dtype=np.float32,
        )

    def __eq__(self, other: object) -> bool:  # pragma: no cover - 方便測試比對
        if not isinstance(other, Board):
            return NotImplemented
        return np.array_equal(self.cells, other.cells)

    def __repr__(self) -> str:  # pragma: no cover - 便於除錯輸出
        return f"Board(visible_rows={self.visible_rows}, cols={self.cols}, filled={int((self.cells != 0).sum())})"


def render_board_ascii(board: Board, show_buffer: bool = False) -> str:
    """把棋盤畫成 ASCII（除錯與 play_ascii.py 用）。"""

    glyphs = {EMPTY: ".", GARBAGE_VALUE: "#", **{v: k for k, v in PIECE_VALUE.items()}}
    cells = board.cells if show_buffer else board.visible
    lines = ["+" + "-" * board.cols + "+"]
    for row in cells:
        lines.append("|" + "".join(glyphs.get(int(v), "?") for v in row) + "|")
    lines.append("+" + "-" * board.cols + "+")
    return "\n".join(lines)
