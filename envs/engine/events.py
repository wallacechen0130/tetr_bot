"""落子事件的結構化描述（reward 與評估指標的唯一輸入）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from envs.engine.rules import ClearType


@dataclass(frozen=True, slots=True)
class StepEvents:
    """單次落子造成的所有事件。"""

    clear_type: ClearType = ClearType.NONE
    lines_cleared: int = 0
    tspin: bool = False
    tspin_mini: bool = False
    combo: int = 0
    b2b_chain: int = 0
    b2b_active: bool = False
    perfect_clear: bool = False
    attack_sent: int = 0
    attack_offset: int = 0
    garbage_received: int = 0
    garbage_applied: int = 0
    used_hold: bool = False
    top_out: bool = False
    rows_increased: int = 0
    holes_created: int = 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["clear_type"] = self.clear_type.value
        return data
