"""評估流程與報告輸出測試。"""

from __future__ import annotations

import gymnasium as gym
import pytest

import envs  # noqa: F401
from agents.heuristic_agent import HeuristicAgent
from evaluators.benchmark import BenchmarkRunner
from evaluators.metrics import EpisodeMetrics, aggregate_metrics, difficulty_fidelity
from evaluators.report import write_report


def test_benchmark_runner_produces_metrics() -> None:
    runner = BenchmarkRunner(
        lambda: gym.make("Tetris40L-v0"),
        HeuristicAgent(depth=1, seed=5),
        seed=5,
        max_steps=400,
    )
    metrics = runner.run(episodes=1)
    assert len(metrics) == 1
    episode = metrics[0]
    assert episode.lines >= 0
    assert episode.pieces > 0
    assert episode.pps > 0
    assert episode.agent == "heuristic"


def test_aggregate_metrics_summary() -> None:
    episodes = [
        EpisodeMetrics(agent="a", lines=10, pieces=20, elapsed=10.0, pps=2.0, apm=40.0, attack_sent=7),
        EpisodeMetrics(agent="a", lines=0, pieces=15, elapsed=8.0, pps=1.9, apm=30.0, top_out=True),
    ]
    summary = aggregate_metrics(episodes)
    assert summary["episodes"] == 2
    assert summary["mean_lines"] == 5.0
    assert summary["top_out_rate"] == 0.5
    assert summary["lines_per_minute"] > 0
    assert summary["agent"] == "a"


def test_difficulty_fidelity_math() -> None:
    result = difficulty_fidelity(observed_pps=2.0, target_pps=2.0, observed_apm=45.0, target_apm=50.0)
    assert result["pps_error"] == 0.0
    assert result["apm_error"] == pytest.approx(0.1)


def test_write_report_files(tmp_path) -> None:
    episodes = [EpisodeMetrics(agent="heuristic", lines=3, pieces=10, elapsed=5.0, pps=2.0)]
    write_report(
        episodes,
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
    )
    assert (tmp_path / "report.json").exists()
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "彙總" in text
    assert "heuristic" in text or "mean_lines" in text


def test_difficulty_suite_runs_two_tiers() -> None:
    from evaluators.difficulty_eval import run_difficulty_suite, summarize_fidelity

    suite = run_difficulty_suite(
        lambda: gym.make("TetrisSurvival-v0"),
        tiers=("Beginner", "Advanced"),
        episodes=1,
        max_steps=25,
        seed=3,
    )
    assert set(suite["fidelity"]) == {"Beginner", "Advanced"}
    fidelity = summarize_fidelity(suite)
    assert fidelity["max_pps_error"] < 0.35
