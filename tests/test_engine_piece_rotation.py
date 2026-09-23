"""方塊幾何與 SRS / SRS+ kick 表測試。"""

from __future__ import annotations

import pytest

from envs.engine.board import Board
from envs.engine.piece import BOX_SIZE, PIECE_KINDS, Piece, piece_cells, piece_rotations, spawn_origin
from envs.engine.srs import KICKS_180, kick_table, try_rotate


def test_every_piece_has_four_cells():
    for kind in PIECE_KINDS:
        for rotation in range(4):
            assert len(piece_rotations(kind)[rotation]) == 4, f"{kind} rot{rotation} 格數不是 4"


def test_o_piece_rotations_are_identical():
    rotations = piece_rotations("O")
    assert rotations[0] == rotations[1] == rotations[2] == rotations[3]


def test_piece_rotation_states_match_srs():
    # J 的 R 狀態：.XX / .X. / .X.
    assert piece_rotations("J")[1] == ((0, 1), (0, 2), (1, 1), (2, 1))
    # T 的 R 狀態：.X. / .XX / .X.
    assert piece_rotations("T")[1] == ((0, 1), (1, 1), (1, 2), (2, 1))
    # I 的 R 狀態：第 2 欄直立
    assert piece_rotations("I")[1] == ((0, 2), (1, 2), (2, 2), (3, 2))
    # I 的 180 狀態：第 2 列
    assert piece_rotations("I")[2] == ((2, 0), (2, 1), (2, 2), (2, 3))


def test_spawn_origin_is_centered():
    row, col = spawn_origin("T", cols=10, buffer_rows=20)
    assert (row, col) == (18, 3)
    row_i, col_i = spawn_origin("I", cols=10, buffer_rows=20)
    assert (row_i, col_i) == (18, 3)
    assert BOX_SIZE["I"] == 4 and BOX_SIZE["T"] == 3


def test_kick_tables_have_srs_shape():
    assert kick_table("T", 0, 1)[0] == (0, 0)
    assert len(kick_table("T", 0, 1)) == 5
    assert len(kick_table("I", 0, 1)) == 5
    assert kick_table("O", 0, 1) == ((0, 0),)
    assert kick_table("T", 0, 2) == KICKS_180


def test_rotation_on_empty_board_uses_first_kick():
    board = Board()
    piece = Piece("T", 0, 18, 3)
    rotated, kick_index = try_rotate(board, piece, 1)
    assert rotated is not None
    assert rotated.rotation == 1
    assert kick_index == 0


def test_wall_kick_is_used_when_needed():
    board = Board()
    for row in (21, 22, 23):
        board.cells[row, 4] = 1
    piece = Piece("T", 0, 19, 3)
    rotated, kick_index = try_rotate(board, piece, 1)
    assert rotated is not None
    assert kick_index >= 1
    assert not board.collides(rotated.cells())


def test_180_rotation_is_supported_when_allowed():
    board = Board()
    piece = Piece("T", 0, 18, 3)
    rotated, _ = try_rotate(board, piece, 2, allow_180=True)
    assert rotated is not None and rotated.rotation == 2
    blocked, _ = try_rotate(board, piece, 2, allow_180=False)
    # 關閉 180 時只剩原地 kick，仍可能成功，但旋轉結果必須是 2
    if blocked is not None:
        assert blocked.rotation == 2


@pytest.mark.parametrize("kind", PIECE_KINDS)
def test_piece_cells_stay_in_bounds(kind: str):
    cells = piece_cells(kind, 0, 18, 3)
    assert all(0 <= col < 10 for _, col in cells)
