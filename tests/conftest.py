"""共用 fixture。"""

from __future__ import annotations

import pytest

import envs  # noqa: F401 - 匯入即完成環境註冊
from agents.heuristic_agent import Candidate
from envs.engine.piece import piece_cells, spawn_origin
from envs.engine.placement import Placement
from envs.engine.rules import load_ruleset


@pytest.fixture(scope="session")
def ruleset():
    return load_ruleset()


def make_placement(kind: str = "T", *, column: int = 4, rotation: int = 0, hold_used: bool = False) -> Placement:
    """建立一個合法的假落點（放在第 39 列附近）。"""

    from envs.engine.board import Board

    board = Board()
    spawn_row, spawn_col = spawn_origin(kind, board.cols, board.buffer_rows)
    cells = piece_cells(kind, rotation, 36, column)
    return Placement(kind=kind, rotation=rotation, cells=cells, hold_used=hold_used, rotated=rotation != 0)


@pytest.fixture
def candidate_factory():
    """產生測試用候選落點清單。"""

    def _factory(count: int = 5, *, base_score: float = 1.0, **kwargs) -> list[Candidate]:
        candidates = []
        for index in range(count):
            placement = make_placement(column=min(9, index), **kwargs)
            candidates.append(
                Candidate(
                    action=index,
                    placement=placement,
                    score=base_score - index * 0.1,
                    features={
                        "combo_bonus": 0.0,
                        "attack_bonus": float(index),
                        "tspin_bonus": 0.0,
                    },
                    lines_cleared=0,
                    attack=index,
                )
            )
        return candidates

    return _factory
