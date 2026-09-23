"""評估指標：強度、效率、攻擊與風格。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    """單局指標。"""

    tier: str = "N/A"
    mode: str = "survival"
    agent: str = "unknown"
    reward: float = 0.0
    pieces: int = 0
    lines: int = 0
    elapsed: float = 0.0
    pps: float = 0.0
    apm: float = 0.0
    attack_sent: int = 0
    garbage_received: int = 0
    max_combo: int = 0
    b2b_clears: int = 0
    tspins: int = 0
    perfect_clears: int = 0
    holds: int = 0
    invalid_actions: int = 0
    holes_final: int = 0
    max_height_final: int = 0
    top_out: bool = False
    completed: bool = False
    lines_remaining: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    """把多局指標彙總成報告用的 dict。"""

    if not episodes:
        return {}
    pps = np.array([ep.pps for ep in episodes], dtype=np.float64)
    apm = np.array([ep.apm for ep in episodes], dtype=np.float64)
    lines = np.array([ep.lines for ep in episodes], dtype=np.float64)
    rewards = np.array([ep.reward for ep in episodes], dtype=np.float64)
    pieces = np.array([ep.pieces for ep in episodes], dtype=np.float64)
    elapsed = np.array([ep.elapsed for ep in episodes], dtype=np.float64)
    style_pieces = float(np.maximum(pieces, 1).sum())

    summary: dict[str, Any] = {
        "episodes": len(episodes),
        "agent": episodes[0].agent,
        "mode": episodes[0].mode,
        "tier": episodes[0].tier,
        "mean_reward": float(rewards.mean()),
        "mean_pps": float(pps.mean()),
        "mean_apm": float(apm.mean()),
        "mean_lines": float(lines.mean()),
        "mean_pieces": float(pieces.mean()),
        "mean_elapsed": float(elapsed.mean()),
        "lines_per_minute": float(lines.sum() / max(float(elapsed.sum()), 1e-6) * 60.0),
        "attack_per_minute": float(
            sum(ep.attack_sent for ep in episodes) / max(float(elapsed.sum()), 1e-6) * 60.0
        ),
        "max_combo": int(max(ep.max_combo for ep in episodes)),
        "b2b_clears": int(sum(ep.b2b_clears for ep in episodes)),
        "top_out_rate": float(np.mean([1.0 if ep.top_out else 0.0 for ep in episodes])),
        "completion_rate": float(np.mean([1.0 if ep.completed else 0.0 for ep in episodes])),
        "mean_holes_final": float(np.mean([ep.holes_final for ep in episodes])),
        "mean_max_height_final": float(np.mean([ep.max_height_final for ep in episodes])),
        "tspin_rate": float(sum(ep.tspins for ep in episodes) / style_pieces),
        "pc_rate": float(sum(ep.perfect_clears for ep in episodes) / style_pieces),
        "hold_rate": float(sum(ep.holds for ep in episodes) / style_pieces),
        "invalid_actions": int(sum(ep.invalid_actions for ep in episodes)),
    }
    times = [ep.elapsed for ep in episodes if ep.completed]
    if times:
        summary["mean_completion_time"] = float(np.mean(times))
        summary["best_completion_time"] = float(np.min(times))
    return summary


def difficulty_fidelity(
    *,
    observed_pps: float,
    target_pps: float,
    observed_apm: float,
    target_apm: float,
) -> dict[str, float]:
    """計算實際 PPS/APM 與目標值的相對誤差。"""

    def rel(observed: float, target: float) -> float:
        if target <= 0:
            return 0.0
        return float(abs(observed - target) / target)

    return {
        "pps_error": rel(observed_pps, target_pps),
        "apm_error": rel(observed_apm, target_apm),
        "apm_ceiling_ok": float(observed_apm <= target_apm * 1.05),
        "observed_pps": float(observed_pps),
        "target_pps": float(target_pps),
        "observed_apm": float(observed_apm),
        "target_apm": float(target_apm),
    }
