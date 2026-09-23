"""消行分類與攻擊表測試。"""

from __future__ import annotations

from envs.engine.rules import ClearType, classify_clear, load_ruleset


def test_clear_type_classification():
    assert classify_clear(0, tspin=False, mini=False) is ClearType.NONE
    assert classify_clear(1, tspin=False, mini=False) is ClearType.SINGLE
    assert classify_clear(4, tspin=False, mini=False) is ClearType.TETRIS
    assert classify_clear(2, tspin=True, mini=False) is ClearType.TSPIN_DOUBLE
    assert classify_clear(2, tspin=True, mini=True) is ClearType.TSPIN_MINI_DOUBLE
    assert classify_clear(3, tspin=True, mini=False) is ClearType.TSPIN_TRIPLE


def test_clear_type_flags():
    assert ClearType.TETRIS.is_difficult
    assert ClearType.TSPIN_DOUBLE.is_difficult
    assert not ClearType.DOUBLE.is_difficult
    assert ClearType.TSPIN_MINI_DOUBLE.is_mini
    assert ClearType.TSPIN_TRIPLE.lines == 3


def test_attack_table_loaded_from_yaml():
    rules = load_ruleset()
    assert rules.attack["single"] == 0
    assert rules.attack["double"] == 1
    assert rules.attack["triple"] == 2
    assert rules.attack["tetris"] == 4
    assert rules.attack["tspin_single"] == 2
    assert rules.attack["tspin_double"] == 4
    assert rules.attack["tspin_triple"] == 6
    assert rules.b2b_bonus == 1
    assert rules.perfect_clear == 10


def test_attack_for_clear_types():
    rules = load_ruleset()
    assert rules.attack_for(ClearType.SINGLE) == 0
    assert rules.attack_for(ClearType.DOUBLE) == 1
    assert rules.attack_for(ClearType.TETRIS) == 4
    assert rules.attack_for(ClearType.TSPIN_DOUBLE) == 4


def test_b2b_charges_only_from_second_difficult_clear():
    rules = load_ruleset()
    assert rules.attack_for(ClearType.TETRIS, b2b_chain=1) == 4
    assert rules.attack_for(ClearType.TETRIS, b2b_chain=2) == 5
    # 非困難消行不吃 B2B 加成
    assert rules.attack_for(ClearType.DOUBLE, b2b_chain=3) == 1


def test_combo_table_and_perfect_clear_bonus():
    rules = load_ruleset()
    assert rules.combo_bonus(0) == 0
    assert rules.combo_bonus(1) == 0
    assert rules.combo_bonus(3) == 1
    assert rules.attack_for(ClearType.TETRIS, perfect_clear=True) == 4 + 10
    assert rules.attack_for(ClearType.DOUBLE, combo=3) == 1 + 1
