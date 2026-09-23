"""把棋盤轉成 numpy RGB 影像（不需要 pygame）。"""

from __future__ import annotations

import numpy as np

from envs.engine.board import Board
from envs.engine.piece import GARBAGE_VALUE, PIECE_KINDS

# 0 空、1..7 對應 I,J,L,O,S,T,Z（TETR.IO 風格近似色）
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (12, 12, 16),
    GARBAGE_VALUE: (110, 110, 120),
    1: (0, 199, 214),   # I
    2: (40, 80, 220),   # J
    3: (230, 140, 20),  # L
    4: (230, 200, 40),  # O
    5: (60, 200, 60),   # S
    6: (170, 60, 200),  # T
    7: (220, 50, 60),   # Z
}


def board_to_rgb(board: Board, cell_size: int = 16, grid_lines: bool = True) -> np.ndarray:
    """把可見區轉成 (H, W, 3) uint8 影像。"""

    visible = board.visible
    rows, cols = visible.shape
    height = rows * cell_size
    width = cols * cell_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            color = PALETTE.get(int(visible[row, col]), (255, 255, 255))
            image[row * cell_size : (row + 1) * cell_size, col * cell_size : (col + 1) * cell_size] = color
    if grid_lines and cell_size >= 4:
        image[::cell_size, :, :] = np.maximum(image[::cell_size, :, :], 40)
        image[:, ::cell_size, :] = np.maximum(image[:, ::cell_size, :], 40)
    return image


def piece_color(kind: str) -> tuple[int, int, int]:
    """取得方塊顏色（給未來的 HUD 使用）。"""

    return PALETTE[PIECE_KINDS.index(kind) + 1]
