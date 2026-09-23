"""Reward 計算測試。"""

from __future__ import annotations

import pytest

from envs.engine.events import StepEvents
from envs.engine.rules import ClearType, load_ruleset
from envs.engine.simulator import GameConfig, TetrisSimulator
from envs.reward import DEFAULT_REWARD, RewardCalculator


def make_snapshot():
    sim = TetrisSimulator(GameConfig(rules=load_ruleset(), target_lines=None, time_limit=None), seed=0)
    return sim.snapshot()


def test_default_config_matches_yaml_keys() -> None:
    from envs.config import load_yaml

    yaml_config = load_yaml("configs/reward.yaml")
    for key in DEFAULT_REWARD:
        assert key in yaml_config, f"reward.yaml 缺少 {key}"


def test_tetris_reward_and_components() -> None:
    calculator = RewardCalculator()
    snapshot = make_snapshot()
    events = StepEvents(clear_type=ClearType.TETRIS, lines_cleared=4, attack_sent=2, combo=1)
    total, components = calculator.compute(events, snapshot, snapshot)
    assert components["line_clear"] == 12.0
    assert components["garbage_sent"] == pytest.approx(0.8)
    assert components["b2b"] == 0.0
    assert total == pytest.approx(sum(value for key, value in components.items() if key != "total"))


def test_b2b_and_combo_and_perfect_clear_bonuses() -> None:
    calculator = RewardCalculator()
    snapshot = make_snapshot()
    events = StepEvents(
        clear_type=ClearType.TSPIN_DOUBLE,
        lines_cleared=2,
        tspin=True,
        b2b_chain=2,
        b2b_active=True,
        combo=3,
        perfect_clear=True,
        attack_sent=6,
    )
    _total, components = calculator.compute(events, snapshot, snapshot)
    assert components["line_clear"] == 14.0
    assert components["b2b"] == pytest.approx(14.0 * 0.5)
    assert components["combo"] == pytest.approx(1.0)
    assert components["perfect_clear"] == 30.0


def test_top_out_and_hole_penalties() -> None:
    calculator = RewardCalculator()
    snapshot = make_snapshot()
    events = StepEvents(top_out=True, holes_created=2, garbage_applied=3)
    total, components = calculator.compute(events, snapshot, snapshot)
    assert components["top_out"] == -20.0
    assert components["new_hole"] == pytest.approx(-1.0)
    assert components["garbage_received"] == pytest.approx(-0.9)
    assert total < 0


def test_curriculum_progress_scales_weights() -> None:
    early = RewardCalculator()
    early.set_progress(0.0)
    late = RewardCalculator()
    late.set_progress(0.9)
    snapshot = make_snapshot()
    events = StepEvents(clear_type=ClearType.SINGLE, lines_cleared=1, attack_sent=1)
    _t1, early_components = early.compute(events, snapshot, snapshot)
    _t2, late_components = late.compute(events, snapshot, snapshot)
    assert early_components["survival"] > late_components["survival"]
    assert late_components["garbage_sent"] > early_components["garbage_sent"]
