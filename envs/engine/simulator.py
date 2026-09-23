"""Tetris 遊戲模擬器：套用落點、計算事件、維護 B2B / Combo / 垃圾行。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.engine.bag import SevenBag
from envs.engine.board import Board
from envs.engine.events import StepEvents
from envs.engine.garbage import GarbageManager
from envs.engine.piece import Piece, spawn_origin
from envs.engine.placement import Placement, enumerate_placements
from envs.engine.rules import Ruleset, classify_clear
from envs.engine.srs import tspin_corners

__all__ = ["GameConfig", "GameSnapshot", "StepResult", "TetrisSimulator"]


@dataclass
class GameConfig:
    """環境層級的遊戲設定（規則、模式與目標）。"""

    rules: Ruleset = field(default_factory=Ruleset)
    mode: str = "survival"  # survival | 40l | versus
    hold_enabled: bool = True
    next_count: int = 7
    time_limit: float | None = 120.0
    target_lines: int | None = None
    opponent_apm: float = 0.0
    allow_kicked_placements: bool = True
    default_piece_time: float = 0.0
    seed: int | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        rules: Ruleset | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> GameConfig:
        env = dict(data.get("env", {}))
        timing = dict(data.get("timing", {}))
        if overrides:
            env.update({key: value for key, value in overrides.items() if value is not None})
        return cls(
            rules=rules or Ruleset(),
            mode=str(env.get("mode", "survival")),
            hold_enabled=bool(env.get("hold_enabled", True)),
            next_count=int(env.get("next_count", 7)),
            time_limit=None if env.get("time_limit") is None else float(env["time_limit"]),
            target_lines=None if env.get("target_lines") is None else int(env["target_lines"]),
            opponent_apm=float(env.get("opponent_apm", 0.0)),
            allow_kicked_placements=bool(env.get("allow_kicked_placements", True)),
            default_piece_time=float(timing.get("piece_time_default", 0.0)),
        )


@dataclass(frozen=True, eq=False)
class GameSnapshot:
    """提供給 agent / controller 的唯讀狀態快照。"""

    board: np.ndarray
    visible: np.ndarray
    current: str
    rotation: int
    hold: str | None
    hold_available: bool
    next_queue: tuple[str, ...]
    combo: int
    b2b_chain: int
    lines_total: int
    pieces_placed: int
    attack_sent: int
    garbage_received: int
    garbage_pending: int
    elapsed: float
    top_out: bool
    lines_remaining: int | None

    def copy(self) -> GameSnapshot:
        return GameSnapshot(
            board=self.board.copy(),
            visible=self.visible.copy(),
            current=self.current,
            rotation=self.rotation,
            hold=self.hold,
            hold_available=self.hold_available,
            next_queue=self.next_queue,
            combo=self.combo,
            b2b_chain=self.b2b_chain,
            lines_total=self.lines_total,
            pieces_placed=self.pieces_placed,
            attack_sent=self.attack_sent,
            garbage_received=self.garbage_received,
            garbage_pending=self.garbage_pending,
            elapsed=self.elapsed,
            top_out=self.top_out,
            lines_remaining=self.lines_remaining,
        )

    def __eq__(self, other: object) -> bool:
        """結構化比較（供 gymnasium env checker 的決定性檢查使用）。"""

        if not isinstance(other, GameSnapshot):
            return NotImplemented
        return (
            np.array_equal(self.board, other.board)
            and np.array_equal(self.visible, other.visible)
            and self.current == other.current
            and self.rotation == other.rotation
            and self.hold == other.hold
            and self.hold_available == other.hold_available
            and self.next_queue == other.next_queue
            and self.combo == other.combo
            and self.b2b_chain == other.b2b_chain
            and self.lines_total == other.lines_total
            and self.pieces_placed == other.pieces_placed
            and self.attack_sent == other.attack_sent
            and self.garbage_received == other.garbage_received
            and self.garbage_pending == other.garbage_pending
            and float(self.elapsed) == float(other.elapsed)
            and self.top_out == other.top_out
            and self.lines_remaining == other.lines_remaining
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    """一次落子的完整結果。"""

    events: StepEvents
    snapshot: GameSnapshot
    top_out: bool
    completed: bool
    info: dict[str, Any] = field(default_factory=dict)


class TetrisSimulator:
    """不綁定 Gymnasium 的核心遊戲邏輯。"""

    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GameConfig()
        self._seed = seed if seed is not None else self.config.seed
        self.reset(self._seed)

    # ------------------------------------------------------------------ 生命週期
    def reset(self, seed: int | None = None) -> None:
        self._seed = seed if seed is not None else self._seed
        self.rng = np.random.default_rng(self._seed)
        self.board = Board(
            visible_rows=self.config.rules.visible_rows,
            cols=self.config.rules.cols,
            buffer_rows=self.config.rules.buffer_rows,
        )
        self.bag = SevenBag(self.rng, queue_length=max(14, self.config.next_count + 7))
        self.garbage = GarbageManager(self.config.rules.garbage, self.rng)
        self.current: str = self.bag.next()
        self.hold: str | None = None
        self.hold_available = True
        self.combo = 0
        self.b2b_chain = 0
        self.lines_total = 0
        self.pieces_placed = 0
        self.attack_sent = 0
        self.garbage_received = 0
        self.elapsed = 0.0
        self.top_out = False

    # ------------------------------------------------------------------ 查詢
    @property
    def rules(self) -> Ruleset:
        return self.config.rules

    def lines_remaining(self) -> int | None:
        if self.config.target_lines is None:
            return None
        return max(0, self.config.target_lines - self.lines_total)

    def completed(self) -> bool:
        remaining = self.lines_remaining()
        return remaining == 0 if remaining is not None else False

    def timed_out(self) -> bool:
        limit = self.config.time_limit
        return limit is not None and self.elapsed >= limit

    def current_piece(self) -> Piece:
        row, col = spawn_origin(self.current, self.board.cols, self.board.buffer_rows)
        return Piece(self.current, 0, row, col)

    def incoming_hold_kind(self) -> str | None:
        """使用 hold 時實際會被放置的方塊種類。"""

        if not self.config.hold_enabled:
            return None
        if self.hold is not None:
            return self.hold
        return self.bag.peek(1)[0]

    def legal_placements(self, *, include_hold: bool = True) -> list[Placement]:
        """列出目前所有合法落點（含 hold 變體）。"""

        placements = enumerate_placements(
            self.board,
            self.current,
            hold_used=False,
            allow_180=self.rules.allow_180,
            allow_kicked_placements=self.config.allow_kicked_placements,
        )
        if include_hold and self.config.hold_enabled:
            hold_kind = self.incoming_hold_kind()
            if hold_kind is not None:
                placements.extend(
                    enumerate_placements(
                        self.board,
                        hold_kind,
                        hold_used=True,
                        allow_180=self.rules.allow_180,
                        allow_kicked_placements=self.config.allow_kicked_placements,
                    )
                )
        return placements

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            board=self.board.cells.copy(),
            visible=self.board.visible.copy(),
            current=self.current,
            rotation=0,
            hold=self.hold,
            hold_available=self.hold_available,
            next_queue=tuple(self.bag.peek(self.config.next_count)),
            combo=self.combo,
            b2b_chain=self.b2b_chain,
            lines_total=self.lines_total,
            pieces_placed=self.pieces_placed,
            attack_sent=self.attack_sent,
            garbage_received=self.garbage_received,
            garbage_pending=self.garbage.pending,
            elapsed=self.elapsed,
            top_out=self.top_out,
            lines_remaining=self.lines_remaining(),
        )

    # ------------------------------------------------------------------ 落子
    def _apply_hold(self, placement: Placement) -> None:
        """先完成 hold 交換，再讓呼叫端放置 placement.kind。"""

        if not placement.hold_used:
            return
        if self.hold is None:
            self.hold = self.current
            self.current = self.bag.next()
        else:
            self.hold, self.current = self.current, self.hold
        self.hold_available = False

    def step_placement(self, placement: Placement, *, time_cost: float | None = None) -> StepResult:
        """套用一個落點，回傳事件與新快照。"""

        if self.top_out:
            raise RuntimeError("遊戲已結束（top out），請先 reset()")

        self._apply_hold(placement)
        if placement.kind != self.current:
            raise ValueError(f"落點方塊 {placement.kind} 與目前方塊 {self.current} 不符")

        prev_holes = self.board.holes()
        prev_height = self.board.max_height()

        self.board.lock_piece(placement.kind, placement.cells)

        tspin = False
        tspin_mini = False
        if placement.kind == "T" and placement.rotated:
            filled, front = tspin_corners(self.board, placement.piece())
            if filled >= 3:
                tspin = True
                tspin_mini = front < 2 and placement.kick_index != 4

        cleared_rows = self.board.clear_lines()
        lines = len(cleared_rows)
        perfect_clear = lines > 0 and self.board.is_perfect_clear()
        clear_type = classify_clear(lines, tspin=tspin, mini=tspin_mini)

        if lines > 0:
            self.combo += 1
            self.b2b_chain = self.b2b_chain + 1 if clear_type.is_difficult else 0
        else:
            self.combo = 0

        attack = self.rules.attack_for(
            clear_type,
            combo=self.combo,
            b2b_chain=self.b2b_chain,
            perfect_clear=perfect_clear,
        )
        attack_offset, attack_out = self.garbage.offset(attack)
        self.attack_sent += attack_out

        self.garbage.advance(1)
        garbage_applied = self.garbage.apply(self.board)

        self.lines_total += lines
        self.pieces_placed += 1
        self.elapsed += self.config.default_piece_time if time_cost is None else float(time_cost)

        new_holes = self.board.holes()
        new_height = self.board.max_height()

        # 產生下一顆方塊並檢查 top out
        self.current = self.bag.next()
        self.hold_available = True
        spawn = self.current_piece()
        self.top_out = bool(self.board.collides(spawn.cells()))

        events = StepEvents(
            clear_type=clear_type,
            lines_cleared=lines,
            tspin=tspin,
            tspin_mini=tspin_mini,
            combo=self.combo if lines > 0 else 0,
            b2b_chain=self.b2b_chain,
            b2b_active=self.b2b_chain >= 2,
            perfect_clear=perfect_clear,
            attack_sent=attack_out,
            attack_offset=attack_offset,
            garbage_received=garbage_applied,
            garbage_applied=garbage_applied,
            used_hold=placement.hold_used,
            top_out=self.top_out,
            rows_increased=max(0, new_height - prev_height),
            holes_created=max(0, new_holes - prev_holes),
        )
        return StepResult(
            events=events,
            snapshot=self.snapshot(),
            top_out=self.top_out,
            completed=self.completed(),
            info={"cleared_rows": cleared_rows},
        )

    # ------------------------------------------------------------------ 對手
    def receive_garbage(self, amount: int) -> int:
        """讓外部（腳本對手）注入垃圾行。"""

        received = self.garbage.receive(amount)
        self.garbage_received += received
        return received
