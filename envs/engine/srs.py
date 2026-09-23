"""SRS / SRS+ 旋轉與 kick 表。

座標系為 (d_row, d_col)，``d_row`` 為負代表往上。
180 度 kick 表在 TETR.IO 並未完整公開，本檔採用社群常見近似值，
並由 ``configs/rules_tetrio.yaml`` 的 ``rotation.kicks_180_approximate`` 標記。
"""

from __future__ import annotations

from collections.abc import Sequence

from envs.engine.board import Board
from envs.engine.piece import BOX_SIZE, Piece

Kick = tuple[int, int]

# JLSTZ：{ (from_rot, to_rot): kicks }
JLSTZ_KICKS: dict[tuple[int, int], tuple[Kick, ...]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (1, 0): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (1, 2): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (2, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (2, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (3, 2): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (3, 0): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (0, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
}

I_KICKS: dict[tuple[int, int], tuple[Kick, ...]] = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (1, -2), (-2, 1)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (1, -2), (-2, 1)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
}

# O 方塊不移動
O_KICKS: tuple[Kick, ...] = ((0, 0),)

# 180 度近似 kick（依序嘗試：原地、上、下、左右、斜角）
KICKS_180: tuple[Kick, ...] = (
    (0, 0),
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
    (1, -1),
    (-1, -1),
)


def kick_table(kind: str, from_rot: int, to_rot: int, *, allow_180: bool = True) -> tuple[Kick, ...]:
    """取得指定旋轉轉換的 kick 序列（第一個一定是原地）。"""

    from_rot %= 4
    to_rot %= 4
    if kind == "O":
        return O_KICKS
    if (from_rot - to_rot) % 4 == 2:  # 180 度
        return KICKS_180 if allow_180 else ((0, 0),)
    table = I_KICKS if kind == "I" else JLSTZ_KICKS
    return table[(from_rot, to_rot)]


def try_rotate(
    board: Board,
    piece: Piece,
    direction: int,
    *,
    allow_180: bool = True,
) -> tuple[Piece | None, int]:
    """嘗試旋轉，成功回傳新 piece 與使用的 kick 索引，失敗回傳 (None, -1)。

    ``direction``：``1`` = 順時針、``-1`` = 逆時針、``2`` = 180 度。
    """

    to_rot = (piece.rotation + direction) % 4 if direction != 2 else (piece.rotation + 2) % 4
    for index, (d_row, d_col) in enumerate(kick_table(piece.kind, piece.rotation, to_rot, allow_180=allow_180)):
        candidate = piece.rotated(to_rot, piece.row + d_row, piece.col + d_col)
        if not board.collides(candidate.cells()):
            return candidate, index
    return None, -1


def hard_drop_row(board: Board, piece: Piece, *, max_drop: int = 64) -> Piece:
    """把 piece 直線下落到最低合法位置。"""

    dropped = piece
    for _ in range(max_drop):
        candidate = dropped.moved(1, 0)
        if board.collides(candidate.cells()):
            break
        dropped = candidate
    return dropped


def tspin_corners(board: Board, piece: Piece) -> tuple[int, int]:
    """回傳 (已填角數, 面向方向已填角數)，僅對 T 方塊有意義。

    使用 3-corner 規則：三個角以上被佔用即為 T-Spin；
    面向方向的兩個角都填滿為完整 T-Spin，否則為 Mini。
    """

    if piece.kind != "T":
        return 0, 0
    row, col = piece.row, piece.col
    corners = {
        "TL": (row, col),
        "TR": (row, col + 2),
        "BL": (row + 2, col),
        "BR": (row + 2, col + 2),
    }
    filled = {name: board.occupied(r, c) for name, (r, c) in corners.items()}
    front_by_rotation: dict[int, Sequence[str]] = {
        0: ("TL", "TR"),
        1: ("TR", "BR"),
        2: ("BL", "BR"),
        3: ("TL", "BL"),
    }
    front = front_by_rotation[piece.rotation % 4]
    return sum(filled.values()), sum(1 for name in front if filled[name])


def box_center(piece: Piece) -> tuple[int, int]:
    """回傳 bounding box 中心格（T-Spin 判定與特徵用）。"""

    size = BOX_SIZE[piece.kind]
    return piece.row + size // 2, piece.col + size // 2
