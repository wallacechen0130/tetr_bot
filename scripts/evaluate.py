"""Phase 6 前的評估：跑基準局並輸出 JSON / Markdown 報告。"""

from __future__ import annotations

import argparse
import json

import gymnasium as gym

import envs  # noqa: F401 - 匯入即完成環境註冊
from agents.heuristic_agent import HeuristicAgent
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent
from agents.scripted_agent import ScriptedPlayer
from controllers.difficulty import DifficultyController
from controllers.timing import FakeClock
from evaluators.benchmark import BenchmarkRunner
from evaluators.difficulty_eval import run_difficulty_suite, summarize_fidelity
from evaluators.metrics import aggregate_metrics
from evaluators.report import write_report


def build_agent(kind: str, *, depth: int, model_path: str | None, seed: int):
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "scripted":
        return ScriptedPlayer(seed=seed)
    if kind == "policy":
        return PolicyAgent(model_path, seed=seed)
    return HeuristicAgent(depth=depth, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tetris AI 評估")
    parser.add_argument("--agent", default="heuristic", choices=["heuristic", "random", "policy", "scripted"])
    parser.add_argument("--model", default=None, help="policy agent 的 .pt 路徑")
    parser.add_argument("--env-id", default="TetrisSurvival-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--tier", default=None, help="若指定則套用 DifficultyController")
    parser.add_argument("--difficulty-suite", action="store_true", help="跑全部七級並檢查保真度")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--out-json", default="logs/report.json")
    parser.add_argument("--out-md", default="logs/report.md")
    args = parser.parse_args()

    if args.difficulty_suite:
        suite = run_difficulty_suite(
            lambda: gym.make(args.env_id),
            episodes=1,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        payload = {"suite": suite, "fidelity": summarize_fidelity(suite)}
        write_report(payload["fidelity"], json_path=args.out_json, markdown_path=args.out_md, title="難度保真度報告")
        print(json.dumps(payload["fidelity"], indent=2, ensure_ascii=False))
        return

    agent = build_agent(args.agent, depth=args.depth, model_path=args.model, seed=args.seed)
    controller = None
    if args.tier:
        controller = DifficultyController(args.tier, clock=FakeClock(), seed=args.seed)
    runner = BenchmarkRunner(
        lambda: gym.make(args.env_id),
        agent,
        controller=controller,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    episodes = runner.run(episodes=args.episodes)
    summary = aggregate_metrics(episodes)
    write_report(episodes, json_path=args.out_json, markdown_path=args.out_md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
