"""啟發式 AI：特徵評估 + 可調深度的落點搜尋。

特徵沿用 Dellacherie 經典權重為骨架，另外加上現代 Tetris 偏好項
（Tetris / T-Spin / Perfect Clear / Combo / B2B），權重全部放在
``configs/heuristic.yaml``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.base import BaseAgent
from envs.config import load_yaml
from envs.engine.board import Board
from envs.engine.placement import Placement, enumerate_placements
from envs.engine.rules import ClearType, Ruleset, classify_clear, load_ruleset
from envs.engine.simulator import GameSnapshot
from envs.engine.srs import tspin_corners
from envs.gym.placement import action_index

FEATURE_ORDER: tuple[str, ...] = (
    "landing_height",
    "eroded_piece_cells",
    "row_transitions",
    "column_transitions",
    "holes",
    "well_sums",
    "aggregate_height",
    "max_height",
    "bumpiness",
    "new_holes",
    "lines_cleared",
    "tetris_bonus",
    "tspin_bonus",
    "perfect_clear_bonus",
    "combo_bonus",
    "b2b_bonus",
    "hold_used_penalty",
    "garbage_pressure",
    "attack_bonus",
)


@dataclass(slots=True)
class Candidate:
    """單一候選落點的評估結果。"""

    action: int
    placement: Placement
    score: float
    features: dict[str, float]
    lines_cleared: int = 0
    attack: int = 0
    is_tspin: bool = False
    tspin_mini: bool = False
    perfect_clear: bool = False
    uses_hold: bool = False
    clear_type: ClearType = ClearType.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "attack": self.attack,
            "is_tspin": self.is_tspin,
            "perfect_clear": self.perfect_clear,
            "uses_hold": self.uses_hold,
            "clear_type": self.clear_type.value,
            "left_column": self.placement.left_column,
            "rotation": self.placement.rotation,
        }


class HeuristicAgent(BaseAgent):
    """以特徵加權評分挑選最佳落點。"""

    name = "heuristic"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        depth: int = 1,
        beam_width: int = 8,
        depth3_top_k: int = 3,
        next_weight: float = 0.6,
        seed: int | None = None,
        config_path: str = "configs/heuristic.yaml",
        rules: Ruleset | None = None,
        rules_path: str = "configs/rules_tetrio.yaml",
    ) -> None:
        super().__init__(seed)
        config = load_yaml(config_path)
        merged = dict(config.get("weights", {}))
        if weights:
            merged.update(weights)
        self.weights: dict[str, float] = {name: float(merged.get(name, 0.0)) for name in FEATURE_ORDER}
        search = config.get("search", {})
        self.depth = int(depth if depth is not None else search.get("default_depth", 1))
        self.beam_width = int(beam_width or search.get("beam_width", 8))
        self.depth3_top_k = int(depth3_top_k or search.get("depth3_top_k", 3))
        self.next_weight = float(next_weight)
        self.rules: Ruleset = rules or load_ruleset(rules_path)

    # ------------------------------------------------------------------ 對外 API
    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        candidates = self.rank(info["snapshot"], info["placements"], depth=self.depth)
        if not candidates:
            return 0
        return int(candidates[0].action)

    def rank(
        self,
        snapshot: GameSnapshot,
        placements: dict[int, Placement] | list[Placement],
        *,
        depth: int | None = None,
        board: Board | None = None,
    ) -> list[Candidate]:
        """回傳依分數排序（高到低）的候選落點。"""

        search_depth = int(self.depth if depth is None else depth)
        board = board if board is not None else self._board_from_snapshot(snapshot)
        if isinstance(placements, dict):
            mapped = {int(key): value for key, value in placements.items()}
        else:
            mapped = {action_index(p): p for p in placements}
        pieces = [snapshot.current, *snapshot.next_queue]
        candidates = self._search(board, snapshot, pieces, mapped, search_depth)
        candidates.sort(key=lambda cand: (-cand.score, cand.action))
        return candidates

    # ------------------------------------------------------------------ 內部
    @staticmethod
    def _board_from_snapshot(snapshot: GameSnapshot) -> Board:
        total_rows, cols = snapshot.board.shape
        board = Board(visible_rows=total_rows - 0, cols=cols, buffer_rows=0)
        board.cells = snapshot.board.copy()
        return board

    def _search(
        self,
        board: Board,
        snapshot: GameSnapshot,
        pieces: list[str],
        placements: dict[int, Placement],
        depth: int,
    ) -> list[Candidate]:
        candidates = [
            self.evaluate(board, snapshot, placement, action=action)
            for action, placement in placements.items()
        ]
        if depth <= 1 or not candidates or len(pieces) < 2:
            return candidates

        top_k = self.beam_width if depth == 2 else self.depth3_top_k
        candidates.sort(key=lambda cand: (-cand.score, cand.action))
        for candidate in candidates[:top_k]:
            child_board = board.copy()
            child_board.lock_piece(candidate.placement.kind, candidate.placement.cells)
            child_board.clear_lines()
            child_placements = enumerate_placements(
                child_board,
                pieces[1],
                allow_180=self.rules.allow_180,
            )
            if not child_placements:
                continue
            child_map = {action_index(p): p for p in child_placements}
            child_candidates = self._search(
                child_board,
                snapshot,
                pieces[1:],
                child_map,
                depth - 1,
            )
            best_child = max((child.score for child in child_candidates), default=0.0)
            candidate.score += self.next_weight * best_child
        return candidates

    def evaluate(
        self,
        board: Board,
        snapshot: GameSnapshot,
        placement: Placement,
        *,
        action: int | None = None,
    ) -> Candidate:
        """評估單一落點（同時回傳完整特徵，供 controller 的風格重排序使用）。"""

        sim = board.copy()
        holes_before = sim.holes()
        sim.lock_piece(placement.kind, placement.cells)

        tspin = False
        tspin_mini = False
        if placement.kind == "T" and placement.rotated:
            filled, front = tspin_corners(sim, placement.piece())
            if filled >= 3:
                tspin = True
                tspin_mini = front < 2 and placement.kick_index != 4

        cleared_rows = sim.clear_lines()
        lines = len(cleared_rows)
        cleared_set = set(cleared_rows)
        eroded_piece_cells = float(lines * sum(1 for row, _ in placement.cells if row in cleared_set))
        perfect_clear = bool(lines > 0 and sim.is_perfect_clear())
        clear_type = classify_clear(lines, tspin=tspin, mini=tspin_mini)

        holes = float(sim.holes())
        combo = snapshot.combo + 1 if lines > 0 else 0
        b2b_chain = snapshot.b2b_chain + 1 if clear_type.is_difficult else 0
        attack = self.rules.attack_for(
            clear_type,
            combo=combo,
            b2b_chain=b2b_chain,
            perfect_clear=perfect_clear,
        )

        features = {
            "landing_height": float(placement.landing_height(sim.total_rows)),
            "eroded_piece_cells": eroded_piece_cells,
            "row_transitions": float(sim.row_transitions()),
            "column_transitions": float(sim.column_transitions()),
            "holes": holes,
            "well_sums": float(sim.well_sums()),
            "aggregate_height": float(sim.aggregate_height()),
            "max_height": float(sim.max_height()),
            "bumpiness": float(sim.bumpiness()),
            "new_holes": float(max(0.0, holes - holes_before)),
            "lines_cleared": float(lines),
            "tetris_bonus": 1.0 if clear_type is ClearType.TETRIS else 0.0,
            "tspin_bonus": 1.0 if tspin else 0.0,
            "perfect_clear_bonus": 1.0 if perfect_clear else 0.0,
            "combo_bonus": 1.0 if (lines > 0 and combo >= 2) else 0.0,
            "b2b_bonus": 1.0 if (clear_type.is_difficult and b2b_chain >= 2) else 0.0,
            "hold_used_penalty": 1.0 if placement.hold_used else 0.0,
            "garbage_pressure": float(snapshot.garbage_pending),
            "attack_bonus": float(attack),
        }
        score = float(sum(self.weights[name] * features[name] for name in FEATURE_ORDER))

        return Candidate(
            action=int(action if action is not None else action_index(placement)),
            placement=placement,
            score=score,
            features=features,
            lines_cleared=lines,
            attack=attack,
            is_tspin=tspin,
            tspin_mini=tspin_mini,
            perfect_clear=perfect_clear,
            uses_hold=placement.hold_used,
            clear_type=clear_type,
        )
