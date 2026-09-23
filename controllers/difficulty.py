"""DifficultyController：單一模型模擬七種等級玩家。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from controllers.governor import APMGovernor
from controllers.misplay import MisplayModel
from controllers.style import StyleBias, StyleWeights
from controllers.timing import Clock, FakeClock, ReactionModel, SystemClock, TokenBucket
from envs.config import load_yaml

if TYPE_CHECKING:  # pragma: no cover - 只為型別檢查
    from agents.heuristic_agent import Candidate
    from envs.engine.placement import Placement

DIFFICULTY_TIERS: tuple[str, ...] = (
    "Beginner",
    "Casual",
    "Intermediate",
    "Advanced",
    "Expert",
    "Professional",
    "Godlike",
)


@dataclass(frozen=True)
class DifficultyProfile:
    """單一等級的完整參數。"""

    name: str
    reaction_ms: float
    think_ms: float
    target_pps: float
    target_apm: float
    lookahead_depth: int
    misplay_chance: float
    hold_usage: float
    tspin_preference: float
    pc_preference: float
    combo_preference: float
    jitter_ms: float = 25.0
    placement_noise_std: float = 0.0
    soft_drop_misuse: float = 0.0

    @property
    def search_depth(self) -> int:
        """把 lookahead_depth 轉成啟發式搜尋層數。"""

        if self.lookahead_depth <= 1:
            return 1
        if self.lookahead_depth == 2:
            return 2
        if self.lookahead_depth == 3:
            return 3
        return 3

    @property
    def beam_width(self) -> int:
        return 8 if self.lookahead_depth < 4 else 16


def load_profiles(path: str = "configs/difficulty.yaml") -> dict[str, DifficultyProfile]:
    """從 YAML 載入七級難度表。"""

    data = load_yaml(path)
    defaults = data.get("defaults", {})
    tiers = data.get("tiers", {})
    profiles: dict[str, DifficultyProfile] = {}
    for name in DIFFICULTY_TIERS:
        values = dict(tiers.get(name, {}))
        profiles[name] = DifficultyProfile(
            name=name,
            reaction_ms=float(values.get("reaction_ms", 250.0)),
            think_ms=float(values.get("think_ms", 100.0)),
            target_pps=float(values.get("target_pps", 1.0)),
            target_apm=float(values.get("target_apm", 30.0)),
            lookahead_depth=int(values.get("lookahead_depth", 1)),
            misplay_chance=float(values.get("misplay_chance", 0.1)),
            hold_usage=float(values.get("hold_usage", 0.3)),
            tspin_preference=float(values.get("tspin_preference", 0.0)),
            pc_preference=float(values.get("pc_preference", 0.0)),
            combo_preference=float(values.get("combo_preference", 0.0)),
            jitter_ms=float(defaults.get("jitter_ms", 25.0)),
            placement_noise_std=float(values.get("placement_noise_std", defaults.get("placement_noise_std", 0.0))),
            soft_drop_misuse=float(values.get("soft_drop_misuse", defaults.get("soft_drop_misuse", 0.0))),
        )
    return profiles


@dataclass(slots=True)
class Decision:
    """一次決策的輸出。"""

    action: int
    delay_s: float
    rank: int
    used_hold: bool
    reason: str
    attack: int = 0
    lines: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "delay_s": self.delay_s,
            "rank": self.rank,
            "used_hold": self.used_hold,
            "reason": self.reason,
            "attack": self.attack,
            "lines": self.lines,
            "score": self.score,
        }


class DifficultyController:
    """以 PPS / APM / 反應時間 / 失誤率 / 風格偏好控制 AI 強度。"""

    def __init__(
        self,
        tier: str = "Intermediate",
        *,
        profiles: dict[str, DifficultyProfile] | None = None,
        clock: Clock | None = None,
        seed: int | None = None,
        ranker: Any = None,
        config_path: str = "configs/difficulty.yaml",
        style_scale: float = 1.0,
    ) -> None:
        self.profiles = profiles or load_profiles(config_path)
        if tier not in self.profiles:
            raise KeyError(f"未知難度：{tier}（可用：{list(self.profiles)}）")
        self.clock: Clock = clock or SystemClock()
        self.rng = np.random.default_rng(seed)
        self.tier = tier
        self.profile = self.profiles[tier]
        self.ranker = ranker
        self.style = StyleBias(StyleWeights.from_profile(self.profile), scale=style_scale)
        self.misplay = MisplayModel(self.profile.misplay_chance, seed=seed)
        self.reaction = ReactionModel(
            self.profile.reaction_ms,
            self.profile.think_ms,
            self.profile.jitter_ms,
            seed=seed,
        )
        self.bucket = TokenBucket(self.profile.target_pps, start=self.clock.now())
        self.governor = APMGovernor(self.profile.target_apm)
        self.pieces = 0
        self.total_delay = 0.0
        self.first_piece = True

    # ------------------------------------------------------------------ 設定
    def set_tier(self, tier: str) -> None:
        """切換難度並同步所有子模型。"""

        if tier not in self.profiles:
            raise KeyError(f"未知難度：{tier}")
        self.tier = tier
        self.profile = self.profiles[tier]
        self.reaction.set_profile(self.profile.reaction_ms, self.profile.think_ms, self.profile.jitter_ms)
        self.bucket.set_rate(self.profile.target_pps)
        self.governor.set_target(self.profile.target_apm)
        self.misplay.set_chance(self.profile.misplay_chance)
        self.style.set_weights(StyleWeights.from_profile(self.profile))

    def with_profile(self, **changes: Any) -> DifficultyProfile:
        """在現有等級上覆寫單一參數（例如只改 target_apm）。"""

        self.profile = replace(self.profile, **changes)
        self.set_tier_values(self.profile)
        return self.profile

    def set_tier_values(self, profile: DifficultyProfile) -> None:
        self.reaction.set_profile(profile.reaction_ms, profile.think_ms, profile.jitter_ms)
        self.bucket.set_rate(profile.target_pps)
        self.governor.set_target(profile.target_apm)
        self.misplay.set_chance(profile.misplay_chance)
        self.style.set_weights(StyleWeights.from_profile(profile))

    def reset(self) -> None:
        """重置計時與統計（換局時呼叫）。"""

        self.bucket.reset()
        self.governor.reset()
        self.pieces = 0
        self.total_delay = 0.0
        self.first_piece = True

    # ------------------------------------------------------------------ 排名
    def rank(self, snapshot: Any, placements: dict[int, Placement]) -> list[Candidate]:
        """用內建啟發式（或外部 ranker）對落點評分。"""

        if self.ranker is None:
            from agents.heuristic_agent import HeuristicAgent

            self.ranker = HeuristicAgent(
                depth=self.profile.search_depth,
                beam_width=self.profile.beam_width,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
        else:
            self.ranker.depth = self.profile.search_depth
            self.ranker.beam_width = self.profile.beam_width
        self._apply_style_weights()
        return self.ranker.rank(snapshot, placements, depth=self.profile.search_depth)

    def _apply_style_weights(self) -> None:
        """把等級的風格偏好轉成啟發式權重覆寫（讓不同等級的打法真的不同）。"""

        if self.ranker is None or not hasattr(self.ranker, "weights"):
            return
        profile = self.profile
        weights = self.ranker.weights
        weights["attack_bonus"] = 20.0 * (
            0.5 * profile.combo_preference + 0.5 * profile.tspin_preference
        )
        weights["tspin_bonus"] = 25.0 * profile.tspin_preference
        weights["perfect_clear_bonus"] = 30.0 * profile.pc_preference
        weights["combo_bonus"] = 8.0 * profile.combo_preference
        weights["hold_used_penalty"] = -3.0 * (1.0 - profile.hold_usage)

    # ------------------------------------------------------------------ 決策
    def begin_piece(self, state: Any = None) -> None:
        """在每顆方塊開始時呼叫（保留給未來加入 per-piece 狀態的擴充）。"""

        return None

    def decide(
        self,
        candidates: list[Candidate],
        *,
        probs: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        now: float | None = None,
    ) -> Decision:
        """從候選落點中選出一個動作，並回傳需要的延遲。"""

        rng = rng or self.rng
        now = self.clock.now() if now is None else float(now)
        profile = self.profile

        if not candidates:
            delay = self.reaction.delay_seconds(first_piece=self.first_piece)
            self.first_piece = False
            self.total_delay += delay
            return Decision(action=0, delay_s=delay, rank=-1, used_hold=False, reason="no_candidates")

        _, adjusted = self.style.rerank(candidates)
        adjusted = np.asarray(adjusted, dtype=np.float64)
        if profile.placement_noise_std > 0:
            adjusted = adjusted + rng.normal(0.0, profile.placement_noise_std, size=adjusted.shape)

        if probs is not None and action_mask is not None:
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
            valid = np.asarray(action_mask).reshape(-1) > 0
            probs = np.where(valid, np.clip(probs, 1e-12, None), 1e-12)
            policy_scores = np.log(np.array([probs[c.action] for c in candidates], dtype=np.float64))
            scale = float(np.std(adjusted)) or 1.0
            combined = policy_scores + adjusted / scale
            reason_prefix = "policy"
        else:
            combined = adjusted
            reason_prefix = "heuristic"

        order = np.argsort(-combined, kind="stable")
        ranked = combined[order]
        position = self.misplay.choose(ranked, rng=rng)
        chosen = candidates[int(order[position])]
        reason = f"{reason_prefix}:rank{position}"

        if self.governor.prefer_low_attack(now) and position == 0:
            top_slice = [candidates[int(index)] for index in order[: min(3, order.size)]]
            cheapest = min(top_slice, key=lambda cand: (cand.attack, -cand.score))
            chosen = cheapest
            reason = f"{reason_prefix}:low_attack"

        base_delay = self.reaction.delay_seconds(first_piece=self.first_piece)
        self.first_piece = False
        wait = self.bucket.wait_time(now, 1.0)
        delay = max(base_delay, wait)
        delay += self.governor.extra_delay(now)
        self.bucket.consume(now + delay, 1.0)
        self.pieces += 1
        self.total_delay += delay

        return Decision(
            action=int(chosen.action),
            delay_s=float(delay),
            rank=int(position),
            used_hold=bool(chosen.uses_hold),
            reason=reason,
            attack=int(chosen.attack),
            lines=int(chosen.lines_cleared),
            score=float(chosen.score),
        )

    def decide_from_observation(
        self,
        observation: dict[str, np.ndarray],
        info: dict[str, Any],
        *,
        probs: np.ndarray | None = None,
    ) -> Decision:
        """便利函式：直接吃 env 的 observation/info。"""

        candidates = self.rank(info["snapshot"], info["placements"])
        return self.decide(
            candidates,
            probs=probs,
            action_mask=np.asarray(observation.get("action_mask", info.get("action_mask"))),
        )

    def note_action(self, *, attack: int = 0, lines: int = 0, now: float | None = None) -> None:
        """回報上一動的攻擊量（APM 治理用）。"""

        now = self.clock.now() if now is None else float(now)
        self.governor.note_attack(attack, now)

    # ------------------------------------------------------------------ 統計
    def stats(self) -> dict[str, float]:
        mean_delay = self.total_delay / max(1, self.pieces)
        return {
            "pieces": float(self.pieces),
            "mean_delay_s": mean_delay,
            "observed_pps": 1.0 / mean_delay if mean_delay > 0 else 0.0,
            "observed_apm": self.governor.current_apm(self.clock.now()),
            "target_pps": self.profile.target_pps,
            "target_apm": self.profile.target_apm,
            "tier": self.tier,  # type: ignore[dict-item]
        }


def make_controller(tier: str = "Intermediate", *, fake_clock: bool = False, seed: int | None = None) -> DifficultyController:
    """便利建構子（測試常需要假時鐘）。"""

    clock: Clock = FakeClock() if fake_clock else SystemClock()
    return DifficultyController(tier, clock=clock, seed=seed)
