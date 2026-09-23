"""落點（placement）列舉。

採兩階段策略：
1. 在出生列做 BFS（左右移動 + 旋轉含 kick），再直線 hard drop。
2. 對每個落點嘗試「落地後原地旋轉」的另一組候選，用來涵蓋 T-Spin 與部分踢牆。

此近似會漏掉真正需要「先下移再旋轉」的 tuck 型落點，
詳見 docs/06_action_space.md 的已知限制說明。
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.engine.board import Board
from envs.engine.piece import Piece, piece_from_cells, spawn_origin
from envs.engine.srs import hard_drop_row, try_rotate


@dataclass(frozen=True, slots=True)
class Placement:
    """一次完整的落子結果（含是否使用 hold、是否為旋轉落點）。"""

    kind: str
    rotation: int
    cells: tuple[tuple[int, int], ...]
    hold_used: bool = False
    rotated: bool = False
    kick_index: int = 0

    @property
    def used_kick(self) -> bool:
        return self.kick_index > 0

    @property
    def left_column(self) -> int:
        return min(col for _, col in self.cells)

    @property
    def right_column(self) -> int:
        return max(col for _, col in self.cells)

    @property
    def bottom_row(self) -> int:
        return max(row for row, _ in self.cells)

    @property
    def top_row(self) -> int:
        return min(row for row, _ in self.cells)

    def landing_height(self, total_rows: int) -> int:
        """落點底部距離地板的高度（Dellacherie 定義的近似）。"""

        return total_rows - self.bottom_row

    def piece(self) -> Piece:
        return piece_from_cells(self.kind, self.rotation, self.cells)

    def dedup_key(self) -> tuple[str, tuple[tuple[int, int], ...], bool]:
        return (self.kind, self.cells, self.hold_used)


def _reachable_states(
    board: Board,
    kind: str,
    *,
    allow_180: bool,
) -> tuple[int, set[tuple[int, int]]]:
    """在出生列以 BFS 找出所有可達的 (rotation, col) 狀態。"""

    origin_row, origin_col = spawn_origin(kind, board.cols, board.buffer_rows)
    start = (0, origin_col)
    if board.collides(Piece(kind, 0, origin_row, origin_col).cells()):
        return origin_row, set()
    seen: set[tuple[int, int]] = {start}
    stack: list[tuple[int, int]] = [start]
    while stack:
        rotation, col = stack.pop()
        piece = Piece(kind, rotation, origin_row, col)
        for d_col in (-1, 1):
            candidate = piece.moved(0, d_col)
            if board.collides(candidate.cells()):
                continue
            state = (rotation, col + d_col)
            if state not in seen:
                seen.add(state)
                stack.append(state)
        for direction in (1, -1, 2):
            if direction == 2 and not allow_180:
                continue
            rotated, _ = try_rotate(board, piece, direction, allow_180=allow_180)
            if rotated is None:
                continue
            state = (rotated.rotation, rotated.col)
            if state not in seen:
                seen.add(state)
                stack.append(state)
    return origin_row, seen


def enumerate_placements(
    board: Board,
    kind: str,
    *,
    hold_used: bool = False,
    allow_180: bool = True,
    allow_kicked_placements: bool = True,
) -> list[Placement]:
    """列出指定方塊在目前棋盤上的所有候選落點。"""

    origin_row, states = _reachable_states(board, kind, allow_180=allow_180)
    placements: dict[tuple[str, tuple[tuple[int, int], ...], bool], Placement] = {}
    landing_pieces: list[Piece] = []

    for rotation, col in states:
        piece = hard_drop_row(board, Piece(kind, rotation, origin_row, col))
        if board.collides(piece.cells()):
            continue
        landing_pieces.append(piece)
        placement = Placement(
            kind=kind,
            rotation=piece.rotation,
            cells=piece.cells(),
            hold_used=hold_used,
            rotated=piece.rotation != 0,
            kick_index=0,
        )
        placements.setdefault(placement.dedup_key(), placement)

    if not allow_kicked_placements:
        return list(placements.values())

    kicked: dict[tuple[str, tuple[tuple[int, int], ...], bool], Placement] = {}
    for piece in landing_pieces:
        for direction in (1, -1, 2):
            if direction == 2 and not allow_180:
                continue
            rotated, kick_index = try_rotate(board, piece, direction, allow_180=allow_180)
            if rotated is None:
                continue
            dropped = hard_drop_row(board, rotated)
            if board.collides(dropped.cells()):
                continue
            candidate = Placement(
                kind=kind,
                rotation=dropped.rotation,
                cells=dropped.cells(),
                hold_used=hold_used,
                rotated=True,
                kick_index=kick_index,
            )
            key = candidate.dedup_key()
            if key in placements:
                continue
            kicked.setdefault(key, candidate)

    placements.update(kicked)
    return list(placements.values())
