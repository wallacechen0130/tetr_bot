"""TETR.IO 風格規則表：消行分類、攻擊表、B2B/Combo/Perfect Clear。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from envs.config import load_yaml
from envs.engine.garbage import GarbageConfig


class ClearType(str, Enum):
    """一次落子造成的消行結果。"""

    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    TETRIS = "tetris"
    TSPIN_MINI = "tspin_mini"
    TSPIN_MINI_SINGLE = "tspin_mini_single"
    TSPIN_MINI_DOUBLE = "tspin_mini_double"
    TSPIN_MINI_TRIPLE = "tspin_mini_triple"
    TSPIN_SINGLE = "tspin_single"
    TSPIN_DOUBLE = "tspin_double"
    TSPIN_TRIPLE = "tspin_triple"

    @property
    def lines(self) -> int:
        return {
            ClearType.NONE: 0,
            ClearType.SINGLE: 1,
            ClearType.DOUBLE: 2,
            ClearType.TRIPLE: 3,
            ClearType.TETRIS: 4,
            ClearType.TSPIN_MINI: 0,
            ClearType.TSPIN_MINI_SINGLE: 1,
            ClearType.TSPIN_MINI_DOUBLE: 2,
            ClearType.TSPIN_MINI_TRIPLE: 3,
            ClearType.TSPIN_SINGLE: 1,
            ClearType.TSPIN_DOUBLE: 2,
            ClearType.TSPIN_TRIPLE: 3,
        }[self]

    @property
    def is_tspin(self) -> bool:
        return self.value.startswith("tspin")

    @property
    def is_mini(self) -> bool:
        return "mini" in self.value

    @property
    def is_difficult(self) -> bool:
        """是否計入 Back-To-Back（Tetris 與有消行的 T-Spin）。"""

        return self is ClearType.TETRIS or (self.is_tspin and self.lines > 0)


BASE_LINE_CLEAR: dict[int, ClearType] = {
    0: ClearType.NONE,
    1: ClearType.SINGLE,
    2: ClearType.DOUBLE,
    3: ClearType.TRIPLE,
    4: ClearType.TETRIS,
}


def classify_clear(lines: int, *, tspin: bool, mini: bool) -> ClearType:
    """依消行數與 T-Spin 狀態分類。"""

    if not tspin:
        return BASE_LINE_CLEAR.get(lines, ClearType.TETRIS)
    if lines <= 0:
        return ClearType.TSPIN_MINI if mini else ClearType.TSPIN_MINI
    if mini:
        return {
            1: ClearType.TSPIN_MINI_SINGLE,
            2: ClearType.TSPIN_MINI_DOUBLE,
            3: ClearType.TSPIN_MINI_TRIPLE,
        }.get(lines, ClearType.TSPIN_MINI_TRIPLE)
    return {
        1: ClearType.TSPIN_SINGLE,
        2: ClearType.TSPIN_DOUBLE,
        3: ClearType.TSPIN_TRIPLE,
    }.get(lines, ClearType.TSPIN_TRIPLE)


@dataclass(frozen=True)
class Ruleset:
    """棋盤尺寸、攻擊表與 TETR.IO 風格規則設定。"""

    visible_rows: int = 20
    cols: int = 10
    buffer_rows: int = 20
    attack: dict[str, int] = field(default_factory=dict)
    b2b_bonus: int = 1
    perfect_clear: int = 10
    combo_table: tuple[int, ...] = (0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5)
    hold_once_per_piece: bool = True
    allow_180: bool = True
    kicks_180_approximate: bool = True
    lock_delay_seconds: float = 0.5
    lock_delay_max_resets: int = 15
    gravity_cells_per_second: float = 20.0
    soft_drop_multiplier: float = 20.0
    block_out: bool = True
    lock_out: bool = False
    garbage: GarbageConfig = field(default_factory=GarbageConfig)

    def base_attack(self, clear_type: ClearType) -> int:
        return int(self.attack.get(clear_type.value, 0))

    def combo_bonus(self, combo: int) -> int:
        """``combo`` 為連續消行次數（1 代表第一次消行）。"""

        if combo <= 0:
            return 0
        index = min(combo - 1, len(self.combo_table) - 1)
        return int(self.combo_table[index])

    def attack_for(
        self,
        clear_type: ClearType,
        *,
        combo: int = 0,
        b2b_chain: int = 0,
        perfect_clear: bool = False,
    ) -> int:
        """計算一次落子送出的垃圾行數。"""

        total = self.base_attack(clear_type)
        if clear_type.is_difficult and b2b_chain >= 2:
            total += self.b2b_bonus
        total += self.combo_bonus(combo)
        if perfect_clear:
            total += self.perfect_clear
        return total


def ruleset_from_dict(data: dict[str, Any]) -> Ruleset:
    """由 YAML dict 建立 Ruleset。"""

    board = data.get("board", {})
    attack = dict(data.get("attack", {}))
    rotation = data.get("rotation", {})
    lock_delay = data.get("lock_delay", {})
    gravity = data.get("gravity", {})
    spawn = data.get("spawn", {})
    topout = data.get("topout", {})
    garbage = data.get("garbage", {})
    return Ruleset(
        visible_rows=int(board.get("visible_rows", 20)),
        cols=int(board.get("cols", 10)),
        buffer_rows=int(board.get("buffer_rows", 20)),
        attack={key: int(value) for key, value in attack.items() if key not in {"b2b_bonus", "perfect_clear", "combo_table"}},
        b2b_bonus=int(attack.get("b2b_bonus", 1)),
        perfect_clear=int(attack.get("perfect_clear", 10)),
        combo_table=tuple(int(v) for v in attack.get("combo_table", ())),
        hold_once_per_piece=bool(spawn.get("hold_once_per_piece", True)),
        allow_180=bool(rotation.get("kicks_180_enabled", spawn.get("allow_180", True))),
        kicks_180_approximate=bool(rotation.get("kicks_180_approximate", True)),
        lock_delay_seconds=float(lock_delay.get("seconds", 0.5)),
        lock_delay_max_resets=int(lock_delay.get("max_resets", 15)),
        gravity_cells_per_second=float(gravity.get("cells_per_second", 20.0)),
        soft_drop_multiplier=float(gravity.get("soft_drop_multiplier", 20.0)),
        block_out=bool(topout.get("block_out", True)),
        lock_out=bool(topout.get("lock_out", False)),
        garbage=GarbageConfig(
            rows_per_batch=int(garbage.get("rows_per_batch", 1)),
            delay_pieces=int(garbage.get("delay_pieces", 0)),
            cheese_ratio=float(garbage.get("cheese_ratio", 0.0)),
            hole_pattern=str(garbage.get("hole_pattern", "random")),
            max_pending=int(garbage.get("max_pending", 20)),
        ),
    )


def load_ruleset(path: str | Path = "configs/rules_tetrio.yaml") -> Ruleset:
    """從 YAML 載入規則表（預設為 TETR.IO 風格）。"""

    return ruleset_from_dict(load_yaml(path))
