"""端到端 smoke test：引擎 → 教師 → 資料集 → IL → PPO → 評估。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _log(step: str, message: str) -> None:
    print(f"[smoke:{step}] {message}", flush=True)


def step_engine() -> dict[str, float]:
    """1. 引擎自我檢查：隨機落點玩 N 顆方塊。"""

    from envs.engine.rules import load_ruleset
    from envs.engine.simulator import GameConfig, TetrisSimulator

    sim = TetrisSimulator(GameConfig(rules=load_ruleset(), target_lines=40, time_limit=None), seed=3)
    rng = np.random.default_rng(0)
    for _ in range(40):
        placements = sim.legal_placements()
        if not placements:
            break
        sim.step_placement(placements[int(rng.integers(0, len(placements)))])
    snapshot = sim.snapshot()
    _log("engine", f"pieces={snapshot.pieces_placed} lines={snapshot.lines_total} top_out={snapshot.top_out}")
    return {"pieces": float(snapshot.pieces_placed), "lines": float(snapshot.lines_total)}


def step_heuristic_40l(*, max_steps: int = 400, seed: int = 11) -> dict[str, float]:
    """2. 啟發式教師打 40L，量測完成時間（以 2 PPS 的時間模型換算）。"""

    import gymnasium as gym

    import envs  # noqa: F401
    from agents.heuristic_agent import HeuristicAgent

    env = gym.make("Tetris40L-v0", seed=seed)
    env.unwrapped.set_piece_time(0.5)
    agent = HeuristicAgent(depth=1, seed=seed)
    observation, info = env.reset(seed=seed)
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.act(observation, info)
        observation, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    env.close()
    completed = info.get("lines_remaining") == 0
    _log(
        "heuristic",
        f"40L completed={completed} lines={info['lines_total']} pieces={info['pieces']} "
        f"elapsed={info['elapsed']:.1f}s pps={info['pps']:.2f}",
    )
    return {
        "completed": float(bool(completed)),
        "lines": float(info["lines_total"]),
        "pieces": float(info["pieces"]),
        "elapsed": float(info["elapsed"]),
        "pps": float(info["pps"]),
    }


def step_dataset(*, episodes: int = 2, workers: int = 2, max_pieces: int = 40, root: str = "datasets/smoke-v1") -> dict[str, float]:
    """3. 產生小資料集（npz + parquet）。"""

    from scripts.generate_dataset import generate

    summary = generate(
        out=root,
        episodes=episodes,
        workers=workers,
        depth=1,
        max_pieces=max_pieces,
        name="smoke-v1",
    )
    _log("dataset", f"samples={summary['num_samples']} shards={summary['num_shards']} root={summary['root']}")
    return {
        "num_samples": float(summary["num_samples"]),
        "num_shards": float(summary["num_shards"]),
        "elapsed_s": float(summary["elapsed_s"]),
    }


def step_il(*, data_root: str = "datasets/smoke-v1", epochs: int = 1, limit: int | None = 512) -> dict[str, float]:
    """4. 用極小資料集跑 1 個 epoch 的模仿學習。"""

    try:
        from trainers.il_trainer import ILConfig, ILTrainer
    except ImportError as exc:  # pragma: no cover - 缺 torch 時跳過
        _log("il", f"跳過（{exc}）")
        return {"skipped": 1.0}

    config = ILConfig(data_root=data_root, epochs=epochs, limit=limit, checkpoint_dir="checkpoints/smoke-il")
    trainer = ILTrainer(config)
    result = trainer.fit(epochs=epochs)
    _log(
        "il",
        f"best_top1={result['best_top1']:.3f} params={result['parameters']} "
        f"train={result['train_samples']} val={result['val_samples']}",
    )
    return {
        "best_top1": float(result["best_top1"]),
        "parameters": float(result["parameters"]),
        "train_samples": float(result["train_samples"]),
    }


def step_ppo(*, timesteps: int = 2048, n_envs: int = 2, il_weights: str | None = None) -> dict[str, float]:
    """5. 小規模 PPO 訓練（驗證 MaskablePPO + 自訂抽取器可跑通）。"""

    try:
        from envs.config import load_yaml
        from trainers.ppo_trainer import PPOTrainer
    except ImportError as exc:  # pragma: no cover
        _log("ppo", f"跳過（{exc}）")
        return {"skipped": 1.0}

    config = load_yaml("configs/ppo.yaml")
    config["env"]["n_envs"] = n_envs
    config["env"]["vec"] = "dummy"
    config["env"]["mode"] = "survival"
    config["env"]["opponent_apm"] = 20.0
    config["model"]["n_steps"] = 128
    config["model"]["batch_size"] = 128
    config["model"]["n_epochs"] = 2
    config["checkpoint"]["dir"] = "checkpoints/smoke-ppo"
    trainer = PPOTrainer(config, seed=0)
    result = trainer.run(total_timesteps=timesteps, il_weights=il_weights)
    evaluation = trainer.evaluate(episodes=2)
    _log(
        "ppo",
        f"timesteps={result['total_timesteps']} transfer={result['transfer']['loaded']}/"
        f"{result['transfer']['total']} mean_reward={evaluation['mean_reward']:.2f}",
    )
    return {
        "timesteps": float(result["total_timesteps"]),
        "il_transfer_loaded": float(result["transfer"]["loaded"]),
        "mean_reward": float(evaluation["mean_reward"]),
        "mean_lines": float(evaluation["mean_lines"]),
    }


def step_evaluate(*, episodes: int = 2) -> dict[str, float]:
    """6. 用 BenchmarkRunner 產出報告。"""

    import gymnasium as gym

    import envs  # noqa: F401
    from agents.heuristic_agent import HeuristicAgent
    from evaluators.benchmark import BenchmarkRunner
    from evaluators.metrics import aggregate_metrics
    from evaluators.report import write_report

    runner = BenchmarkRunner(lambda: gym.make("Tetris40L-v0"), HeuristicAgent(depth=1), seed=5, max_steps=400)
    episodes_metrics = runner.run(episodes=episodes)
    summary = aggregate_metrics(episodes_metrics)
    write_report(episodes_metrics, json_path="logs/smoke_report.json", markdown_path="logs/smoke_report.md")
    _log(
        "evaluate",
        f"episodes={summary['episodes']} completion_rate={summary['completion_rate']:.2f} "
        f"mean_elapsed={summary['mean_elapsed']:.1f}s mean_pps={summary['mean_pps']:.2f}",
    )
    return {
        "completion_rate": float(summary.get("completion_rate", 0.0)),
        "mean_elapsed": float(summary.get("mean_elapsed", 0.0)),
        "mean_pps": float(summary.get("mean_pps", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端 smoke test")
    parser.add_argument("--timesteps", type=int, default=2048, help="PPO 訓練步數（0 = 跳過）")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--out", default="logs/smoke_summary.json")
    parser.add_argument("--dataset-root", default="datasets/smoke-v1")
    args = parser.parse_args()

    started = time.time()
    result: dict[str, object] = {}
    result["engine"] = step_engine()
    result["heuristic_40l"] = step_heuristic_40l()
    result["dataset"] = step_dataset(root=args.dataset_root)
    result["il"] = step_il(data_root=args.dataset_root)
    il_checkpoint = Path("checkpoints/smoke-il/best.pt")
    if args.skip_ppo or args.timesteps <= 0:
        result["ppo"] = {"skipped": 1.0}
        _log("ppo", "依參數跳過")
    else:
        result["ppo"] = step_ppo(
            timesteps=args.timesteps,
            il_weights=str(il_checkpoint) if il_checkpoint.exists() else None,
        )
    result["evaluate"] = step_evaluate()
    result["total_seconds"] = time.time() - started

    Path("logs").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    _log("done", f"全部完成，耗時 {result['total_seconds']:.1f} 秒")


if __name__ == "__main__":
    main()
