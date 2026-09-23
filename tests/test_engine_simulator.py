"""模擬器：消行、B2B、Combo、Perfect Clear、垃圾行與 top out。"""

from __future__ import annotations

import numpy as np

from envs.engine.piece import PIECE_VALUE
from envs.engine.placement import Placement
from envs.engine.rules import ClearType, load_ruleset
from envs.engine.simulator import GameConfig, TetrisSimulator


def make_sim(**kwargs) -> TetrisSimulator:
    config = GameConfig(rules=load_ruleset(), target_lines=40, time_limit=None, **kwargs)
    return TetrisSimulator(config, seed=1234)


def test_reset_is_deterministic():
    first = make_sim().snapshot()
    second = make_sim().snapshot()
    assert first.current == second.current
    assert first.next_queue == second.next_queue


def test_legal_placements_are_valid():
    sim = make_sim()
    placements = sim.legal_placements()
    assert placements
    for placement in placements:
        assert not sim.board.collides(placement.cells)
        assert all(0 <= col < 10 for _, col in placement.cells)
        assert all(row < sim.board.total_rows for row, _ in placement.cells)


def test_step_placement_produces_events_and_advances():
    sim = make_sim()
    placements = sim.legal_placements()
    result = sim.step_placement(placements[0])
    assert result.events.lines_cleared >= 0
    assert result.snapshot.pieces_placed == 1
    assert sim.pieces_placed == 1


def test_hold_swaps_current_piece():
    sim = make_sim()
    original = sim.current
    hold_placement = next(p for p in sim.legal_placements() if p.hold_used)
    sim.step_placement(hold_placement)
    assert sim.hold == original


def test_single_line_clear_updates_combo_and_attack():
    sim = make_sim()
    # 先鋪滿第 39 列，只留 col 9
    sim.board.cells[39, 0:9] = PIECE_VALUE["J"]
    # 直接把方塊鎖在 (39, 9)
    sim.board.cells[39, 9] = PIECE_VALUE["T"]
    result = sim.step_placement(Placement(kind=sim.current, rotation=0, cells=((39, 9), (38, 9), (38, 8), (37, 9))))
    assert result.events.lines_cleared >= 1
    assert result.events.combo >= 1
    assert result.events.clear_type is not ClearType.NONE


def test_top_out_when_spawn_blocked():
    sim = make_sim()
    sim.board.cells[18:22, 0:9] = PIECE_VALUE["J"]
    placement = Placement(kind=sim.current, rotation=0, cells=((30, 0), (30, 1), (30, 2), (30, 3)))
    result = sim.step_placement(placement)
    assert result.top_out
    assert result.events.top_out


def test_garbage_received_then_offset():
    sim = make_sim()
    sim.receive_garbage(4)
    assert sim.garbage.pending == 4
    placements = sim.legal_placements()
    sim.step_placement(placements[0])
    # 沒有攻擊時垃圾應該落到棋盤底部
    assert sim.garbage_received == 4
    assert int((sim.board.cells[-1] == 0).sum()) >= 1


def test_40l_completion_flag():
    sim = make_sim()
    sim.lines_total = 40
    assert sim.completed()
    assert sim.lines_remaining() == 0


def test_time_cost_accumulates():
    sim = make_sim()
    placements = sim.legal_placements()
    sim.step_placement(placements[0], time_cost=0.25)
    assert sim.elapsed == 0.25


def test_bag_distribution_is_uniform_across_seven_pieces():
    from envs.engine.bag import SevenBag

    bag = SevenBag(np.random.default_rng(7))
    first = [bag.next() for _ in range(7)]
    second = [bag.next() for _ in range(7)]
    assert sorted(first) == sorted(second)
    assert len(set(first)) == 7
    assert bag.peek(5) == bag.peek(5)
