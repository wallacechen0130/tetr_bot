"""難度保真度評估：同一模型在七個等級下的 PPS / APM / 風格差異。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from controllers.difficulty import DIFFICULTY_TIERS, DifficultyController
from controllers.timing import FakeClock
from evaluators.benchmark import BenchmarkRunner
from evaluators.metrics import aggregate_metrics, difficulty_fidelity


def run_difficulty_suite(
    env_factory: Callable[[], Any],
    *,
    tiers: tuple[str, ...] = DIFFICULTY_TIERS,
    episodes: int = 1,
    max_steps: int = 120,
    seed: int = 0,
    ranker: Any = None,
) -> dict[str, Any]:
    """對每個等級跑短局，檢查 PPS/APM 是否落在目標附近。"""

    results: dict[str, Any] = {"tiers": {}, "fidelity": {}}
    for offset, tier in enumerate(tiers):
        clock = FakeClock()
        controller = DifficultyController(tier, clock=clock, seed=seed + offset, ranker=ranker)
        runner = BenchmarkRunner(env_factory, agent=controller, controller=controller, seed=seed + offset)
        episodes_metrics = runner.run(episodes=episodes)
        summary = aggregate_metrics(episodes_metrics)
        profile = controller.profile
        observed_pps = float(summary.get("mean_pps", 0.0))
        observed_apm = float(summary.get("mean_apm", 0.0))
        fidelity = difficulty_fidelity(
            observed_pps=observed_pps,
            target_pps=profile.target_pps,
            observed_apm=observed_apm,
            target_apm=profile.target_apm,
        )
        fidelity["tier"] = tier  # type: ignore[assignment]
        fidelity["mean_lines"] = float(summary.get("mean_lines", 0.0))
        fidelity["tspin_rate"] = float(summary.get("tspin_rate", 0.0))
        results["tiers"][tier] = summary
        results["fidelity"][tier] = fidelity
    results["monotonic_pps"] = bool(
        all(
            results["fidelity"][tiers[index]]["target_pps"]
            < results["fidelity"][tiers[index + 1]]["target_pps"]
            for index in range(len(tiers) - 1)
        )
    )
    return results


def summarize_fidelity(results: dict[str, Any]) -> dict[str, float]:
    """計算整體 PPS/APM 誤差統計。"""

    pps_errors = np.array([value["pps_error"] for value in results["fidelity"].values()], dtype=np.float64)
    apm_errors = np.array([value["apm_error"] for value in results["fidelity"].values()], dtype=np.float64)
    return {
        "mean_pps_error": float(pps_errors.mean()) if pps_errors.size else 0.0,
        "max_pps_error": float(pps_errors.max()) if pps_errors.size else 0.0,
        "mean_apm_error": float(apm_errors.mean()) if apm_errors.size else 0.0,
        "max_apm_error": float(apm_errors.max()) if apm_errors.size else 0.0,
    }
