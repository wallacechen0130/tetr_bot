# 4. 強度控制系統設計（DifficultyController）

目標：**同一個模型**可以模擬從新手到神級的不同玩家。控制點分五層：

| 子模組 | 檔案 | 控制什麼 |
|---|---|---|
| `ReactionModel` | `controllers/timing.py` | 反應時間、思考時間、抖動 |
| `TokenBucket` | `controllers/timing.py` | PPS（保護節奏上限） |
| `MisplayModel` | `controllers/misplay.py` | 操作失誤（依排名加權挑次佳落點） |
| `StyleBias` | `controllers/style.py` | T-Spin / PC / Combo / Hold / 攻擊偏好 |
| `APMGovernor` | `controllers/governor.py` | APM 上限（超標時放慢或改選低攻擊落點） |

## 4.1 七級參數表（`configs/difficulty.yaml`）

| 等級 | reaction_ms | think_ms | target_pps | target_apm | lookahead | misplay | hold | T-Spin | PC | Combo |
|---|---|---|---|---|---|---|---|---|---|---|
| Beginner | 700 | 300 | 0.5 | 10 | 0 | 0.35 | 0.05 | 0.00 | 0.00 | 0.00 |
| Casual | 450 | 250 | 0.9 | 20 | 1 | 0.22 | 0.15 | 0.05 | 0.05 | 0.10 |
| Intermediate | 320 | 180 | 1.4 | 35 | 1 | 0.12 | 0.30 | 0.15 | 0.10 | 0.25 |
| Advanced | 240 | 130 | 1.9 | 55 | 2 | 0.06 | 0.50 | 0.30 | 0.20 | 0.40 |
| Expert | 180 | 90 | 2.4 | 80 | 2 | 0.03 | 0.65 | 0.50 | 0.35 | 0.55 |
| Professional | 140 | 60 | 3.0 | 110 | 3 | 0.015 | 0.80 | 0.70 | 0.50 | 0.70 |
| Godlike | 100 | 40 | 3.6 | 150 | 4 | 0.005 | 0.90 | 0.85 | 0.65 | 0.85 |

* `lookahead` 0–1 對映 1 層搜尋、2 → 2 層、3 → 3 層、4 → 3 層但 beam 加倍。
* 風格偏好會同時改寫啟發式權重（`DifficultyController._apply_style_weights`），
  所以高等級真的會去做 T-Spin、追 Combo、保 hold，而不是只改一個標籤。

## 4.2 決策流程

```text
decide(candidates)
  1. StyleBias.rerank()          # 依等級重排（T-Spin / PC / Combo / Hold / 攻擊）
  2. 若提供 policy probs         # log p + 風格分數 → 融合排序
  3. MisplayModel.choose()       # 以 misplay_chance 挑次佳落點
  4. APMGovernor.prefer_low_attack()  # 超標時改選低攻擊落點
  5. delay = max(reaction+think+jitter, token bucket wait) + governor delay
  6. 回傳 Decision(action, delay_s, rank, used_hold, reason)
```

## 4.3 完整類別範例

以下與 `controllers/difficulty.py` 相同（節錄常數以保持可讀，實際檔案含型別註解與 docstring）。

```python
from dataclasses import dataclass, replace
import numpy as np

DIFFICULTY_TIERS = (
    "Beginner", "Casual", "Intermediate", "Advanced",
    "Expert", "Professional", "Godlike",
)


@dataclass(frozen=True)
class DifficultyProfile:
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
        return 1 if self.lookahead_depth <= 1 else min(self.lookahead_depth, 3)

    @property
    def beam_width(self) -> int:
        return 8 if self.lookahead_depth < 4 else 16


@dataclass(slots=True)
class Decision:
    action: int
    delay_s: float
    rank: int
    used_hold: bool
    reason: str
    attack: int = 0
    lines: int = 0
    score: float = 0.0


class DifficultyController:
    """以 PPS / APM / 反應時間 / 失誤率 / 風格偏好控制 AI 強度。"""

    def __init__(self, tier="Intermediate", *, profiles=None, clock=None, seed=None,
                 ranker=None, config_path="configs/difficulty.yaml", style_scale=1.0):
        self.profiles = profiles or load_profiles(config_path)
        self.clock = clock or SystemClock()
        self.rng = np.random.default_rng(seed)
        self.tier = tier
        self.profile = self.profiles[tier]
        self.ranker = ranker                       # 通常是 HeuristicAgent
        self.style = StyleBias(StyleWeights.from_profile(self.profile), scale=style_scale)
        self.misplay = MisplayModel(self.profile.misplay_chance, seed=seed)
        self.reaction = ReactionModel(self.profile.reaction_ms, self.profile.think_ms,
                                      self.profile.jitter_ms, seed=seed)
        self.bucket = TokenBucket(self.profile.target_pps, start=self.clock.now())
        self.governor = APMGovernor(self.profile.target_apm)
        self.pieces = 0
        self.total_delay = 0.0
        self.first_piece = True

    # ---- 設定 -------------------------------------------------------------
    def set_tier(self, tier: str) -> None:
        self.tier = tier
        self.profile = self.profiles[tier]
        self.set_tier_values(self.profile)

    def set_tier_values(self, profile: DifficultyProfile) -> None:
        self.reaction.set_profile(profile.reaction_ms, profile.think_ms, profile.jitter_ms)
        self.bucket.set_rate(profile.target_pps)
        self.governor.set_target(profile.target_apm)
        self.misplay.set_chance(profile.misplay_chance)
        self.style.set_weights(StyleWeights.from_profile(profile))

    def with_profile(self, **changes):
        """在現有等級上覆寫單一參數，例如 with_profile(target_apm=45)."""
        self.profile = replace(self.profile, **changes)
        self.set_tier_values(self.profile)
        return self.profile

    def reset(self) -> None:
        self.bucket.reset()
        self.governor.reset()
        self.pieces = 0
        self.total_delay = 0.0
        self.first_piece = True

    # ---- 決策 -------------------------------------------------------------
    def decide(self, candidates, *, probs=None, action_mask=None, rng=None, now=None) -> Decision:
        rng = rng or self.rng
        now = self.clock.now() if now is None else float(now)
        if not candidates:
            delay = self.reaction.delay_seconds(first_piece=self.first_piece)
            self.first_piece = False
            self.total_delay += delay
            return Decision(action=0, delay_s=delay, rank=-1, used_hold=False,
                            reason="no_candidates")

        _, adjusted = self.style.rerank(candidates)          # 風格重排序
        adjusted = np.asarray(adjusted, dtype=np.float64)
        if self.profile.placement_noise_std > 0:
            adjusted += rng.normal(0.0, self.profile.placement_noise_std, size=adjusted.shape)

        if probs is not None and action_mask is not None:
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
            valid = np.asarray(action_mask).reshape(-1) > 0
            probs = np.where(valid, np.clip(probs, 1e-12, None), 1e-12)
            policy_scores = np.log(np.array([probs[c.action] for c in candidates]))
            scale = float(np.std(adjusted)) or 1.0
            combined = policy_scores + adjusted / scale
            prefix = "policy"
        else:
            combined = adjusted
            prefix = "heuristic"

        order = np.argsort(-combined, kind="stable")
        position = self.misplay.choose(combined[order], rng=rng)     # 操作失誤
        chosen = candidates[int(order[position])]
        reason = f"{prefix}:rank{position}"

        if self.governor.prefer_low_attack(now) and position == 0:    # APM 治理
            top = [candidates[int(i)] for i in order[: min(3, order.size)]]
            chosen = min(top, key=lambda c: (c.attack, -c.score))
            reason = f"{prefix}:low_attack"

        base_delay = self.reaction.delay_seconds(first_piece=self.first_piece)
        self.first_piece = False
        delay = max(base_delay, self.bucket.wait_time(now, 1.0))
        delay += self.governor.extra_delay(now)
        self.bucket.consume(now + delay, 1.0)
        self.pieces += 1
        self.total_delay += delay
        return Decision(action=int(chosen.action), delay_s=float(delay), rank=int(position),
                        used_hold=bool(chosen.uses_hold), reason=reason,
                        attack=int(chosen.attack), lines=int(chosen.lines_cleared),
                        score=float(chosen.score))

    def note_action(self, *, attack=0, lines=0, now=None) -> None:
        self.governor.note_attack(attack, self.clock.now() if now is None else float(now))

    def stats(self) -> dict:
        mean_delay = self.total_delay / max(1, self.pieces)
        return {"pieces": self.pieces, "observed_pps": 1.0 / mean_delay if mean_delay else 0.0,
                "observed_apm": self.governor.current_apm(self.clock.now()),
                "target_pps": self.profile.target_pps, "target_apm": self.profile.target_apm,
                "tier": self.tier}
```

## 4.4 七級實測（本機，20 步短局）

| 等級 | 目標 PPS | 實測 PPS | 誤差 | 目標 APM（上限） | 實測 APM | 上限符合 |
|---|---|---|---|---|---|---|
| Beginner | 0.50 | 0.50 | 0.6% | 10 | 0.5 | ✅ |
| Casual | 0.90 | 0.90 | 0.1% | 20 | 3.0 | ✅ |
| Intermediate | 1.40 | 1.40 | 0.1% | 35 | 4.8 | ✅ |
| Advanced | 1.90 | 1.88 | 0.8% | 55 | 0.0 | ✅ |
| Expert | 2.40 | 2.39 | 0.4% | 80 | 0.0 | ✅ |
| Professional | 3.00 | 2.99 | 0.3% | 110 | 0.0 | ✅ |
| Godlike | 3.60 | 3.60 | 0.0% | 150 | 4.7 | ✅ |

**PPS 是可精確控制的（誤差 < 1%）**；APM 的部分要說清楚：

* `target_apm` 由 `APMGovernor` 當作**上限**治理，七級全部符合「不超過目標」。
* 實際達成值偏低是因為目前的策略主體是「存活優先」的啟發式教師 —— 它不堆高做 Tetris，
  所以攻擊輸出天生就少（≤ 5 APM）。要讓實際 APM 追到 50–150，必須靠 Phase 5 的 PPO
  搭配攻擊導向 reward（`reward_config.garbage_sent_per_line` + 課程後段 `attack_multiplier_late`）。
* 這是設計上的取捨：控制器負責「不超過上限」與「出手節奏」，攻擊企圖心交給模型學習。
