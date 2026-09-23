"""DifficultyController：PPS/APM 保真度、反應時間、失誤與風格測試。"""

from __future__ import annotations

import numpy as np
import pytest

from controllers.difficulty import DIFFICULTY_TIERS, DifficultyController, load_profiles
from controllers.governor import APMGovernor
from controllers.misplay import MisplayModel
from controllers.style import StyleBias, StyleWeights
from controllers.timing import FakeClock, ReactionModel, TokenBucket


def test_all_tiers_are_loaded() -> None:
    profiles = load_profiles()
    assert tuple(profiles) == DIFFICULTY_TIERS
    assert len(profiles) == 7


def test_tier_parameters_are_monotonic() -> None:
    profiles = load_profiles()
    pps = [profiles[tier].target_pps for tier in DIFFICULTY_TIERS]
    apm = [profiles[tier].target_apm for tier in DIFFICULTY_TIERS]
    misplay = [profiles[tier].misplay_chance for tier in DIFFICULTY_TIERS]
    reaction = [profiles[tier].reaction_ms for tier in DIFFICULTY_TIERS]
    assert pps == sorted(pps)
    assert apm == sorted(apm)
    assert misplay == sorted(misplay, reverse=True)
    assert reaction == sorted(reaction, reverse=True)
    assert profiles["Godlike"].lookahead_depth >= profiles["Beginner"].lookahead_depth


def test_token_bucket_limits_rate() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=2.0, start=clock.now())
    assert bucket.wait_time(clock.now()) == 0.0
    bucket.consume(clock.now())
    wait = bucket.wait_time(clock.now())
    assert wait == pytest.approx(0.5, abs=1e-6)


def test_pps_fidelity_within_five_percent() -> None:
    for tier in ("Beginner", "Casual", "Intermediate", "Advanced", "Expert", "Professional", "Godlike"):
        clock = FakeClock()
        controller = DifficultyController(tier, clock=clock, seed=1)
        candidates = _fake_candidates()
        for _ in range(300):
            decision = controller.decide(candidates, now=clock.now())
            clock.advance(decision.delay_s)
        observed = controller.pieces / clock.now()
        target = controller.profile.target_pps
        assert abs(observed - target) / target <= 0.05, f"{tier}: {observed:.3f} vs {target}"


def test_first_decision_respects_reaction_time() -> None:
    clock = FakeClock()
    controller = DifficultyController("Advanced", clock=clock, seed=2)
    decision = controller.decide(_fake_candidates(), now=clock.now())
    minimum = controller.profile.reaction_ms / 1000.0
    assert decision.delay_s >= minimum


def test_reaction_model_jitter_is_non_negative() -> None:
    model = ReactionModel(200, 80, 25, seed=3)
    delays = [model.delay_seconds() for _ in range(50)]
    assert all(delay >= 0.28 - 1e-9 for delay in delays)
    assert float(np.std(delays)) > 0.0


def test_misplay_probabilities() -> None:
    scores = np.array([3.0, 2.5, 2.0, 1.0])
    always = MisplayModel(1.0, seed=4)
    ranks = [always.choose(scores) for _ in range(200)]
    assert min(ranks) >= 1
    assert always.mistakes == 200

    never = MisplayModel(0.0, seed=4)
    assert all(never.choose(scores) == 0 for _ in range(50))


def test_governor_reacts_to_over_budget_attack() -> None:
    governor = APMGovernor(target_apm=30.0, window_s=6.0)
    for second in range(12):
        governor.note_attack(20, second * 0.5)
    assert governor.current_apm(6.0) > 30.0
    assert governor.over_budget(6.0)
    assert governor.prefer_low_attack(6.0)
    assert governor.extra_delay(6.0) > 0.0


def test_style_bias_promotes_tspin() -> None:
    candidates = _fake_candidates()
    candidates[1].is_tspin = True
    candidates[1].score = candidates[0].score - 0.5
    style = StyleBias(StyleWeights(tspin=1.0))
    _ordered, adjusted = style.rerank(candidates)
    assert adjusted[1] > candidates[1].score
    plain = StyleBias(StyleWeights())
    _ordered2, adjusted2 = plain.rerank(candidates)
    assert adjusted2[1] == pytest.approx(candidates[1].score)


def test_controller_switches_tier() -> None:
    controller = DifficultyController("Intermediate", clock=FakeClock(), seed=5)
    controller.set_tier("Godlike")
    assert controller.profile.target_pps == pytest.approx(3.6)
    assert controller.bucket.rate == pytest.approx(3.6)
    assert controller.governor.target_apm == pytest.approx(150.0)
    with pytest.raises(KeyError):
        controller.set_tier("NotATier")


def test_controller_stats_and_note_action() -> None:
    clock = FakeClock()
    controller = DifficultyController("Intermediate", clock=clock, seed=6)
    candidates = _fake_candidates()
    for _ in range(20):
        decision = controller.decide(candidates, now=clock.now())
        clock.advance(decision.delay_s)
        controller.note_action(attack=2)
    stats = controller.stats()
    assert stats["pieces"] == 20
    assert stats["observed_pps"] > 0
    assert stats["tier"] == "Intermediate"


def _fake_candidates():
    from agents.heuristic_agent import Candidate
    from tests.conftest import make_placement

    candidates = []
    for index in range(6):
        placement = make_placement(column=index)
        candidates.append(
            Candidate(
                action=index,
                placement=placement,
                score=10.0 - index,
                features={"combo_bonus": 0.0, "attack_bonus": 0.0},
                attack=index % 3,
                lines_cleared=1 if index % 2 == 0 else 0,
            )
        )
    return candidates
