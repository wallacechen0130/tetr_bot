"""棋盤資料結構測試。"""

from __future__ import annotations

from envs.engine.board import Board, holes_from_grid, max_height_from_grid
from envs.engine.piece import PIECE_VALUE, Piece


def test_clear_lines_shifts_rows_down():
    board = Board()
    board.cells[39, :] = PIECE_VALUE["J"]
    board.cells[38, 0] = PIECE_VALUE["T"]
    cleared = board.clear_lines()
    assert cleared == [39]
    assert board.cells[39, 0] == PIECE_VALUE["T"]
    assert int(board.cells[38].sum()) == 0


def test_heights_holes_and_bumpiness():
    board = Board()
    board.cells[39, 0] = 1
    board.cells[38, 0] = 1
    board.cells[39, 2] = 1
    heights = board.column_heights()
    assert heights[0] == 2
    assert heights[1] == 0
    assert heights[2] == 1
    assert board.bumpiness() == 4  # |2-0| + |0-1| + |1-0|
    assert board.holes() == 0


def test_hole_detection_counts_blocked_cells():
    board = Board()
    board.cells[37, 0] = 1
    board.cells[38, 0] = 1
    # (39, 0) 被上方擋住 → 一個洞
    assert board.holes() == 1
    assert holes_from_grid(board.cells) == 1
    assert max_height_from_grid(board.cells) == 3


def test_perfect_clear_detection():
    board = Board()
    board.cells[39, 0:6] = PIECE_VALUE["J"]
    piece = Piece("I", 0, 38, 6)
    board.lock_piece("I", piece.cells())
    assert board.full_rows() == [39]
    assert len(board.clear_lines()) == 1
    assert board.is_perfect_clear()


def test_garbage_row_adds_hole():
    board = Board()
    board.add_garbage_row(3)
    assert int((board.cells[-1] == 0).sum()) == 1
    assert board.cells[-1, 3] == 0


def test_well_sums_counts_nested_wells():
    board = Board()
    for row in (38, 39):
        board.cells[row, 0] = 1
        board.cells[row, 2] = 1
    # col 1 是深度 2 的 well：1 + 2 = 3
    assert board.well_sums() == 3


def test_copy_is_independent():
    board = Board()
    board.cells[39, 0] = 1
    clone = board.copy()
    clone.cells[39, 1] = 1
    assert board.cells[39, 1] == 0
    assert clone.cells[39, 1] == 1
