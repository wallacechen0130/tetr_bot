"""Phase 5：PPO 微調（SB3 + sb3-contrib MaskablePPO）。"""

from __future__ import annotations

import argparse
import json

from envs.config import load_yaml
from trainers.common import write_json

try:
    from trainers.ppo_trainer import PPOTrainer
except ModuleNotFoundError as exc:  # pragma: no cover - 缺依賴時給出可執行的指示
    raise SystemExit(
        f"PPO 需要 stable-baselines3 與 sb3-contrib（{exc.name} 找不到）。\n"
        "請先安裝依賴：pip install -r requirements-colab.txt（或 pip install -r requirements.txt）"
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 微調")
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec", default=None, choices=["subproc", "dummy"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--il-weights", default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--out", default="logs/ppo_result.json")
    args = parser.parse_args()

    config = load_yaml(args.config)
    trainer = PPOTrainer(config)
    if args.n_envs:
        trainer.n_envs = int(args.n_envs)
    if args.vec:
        trainer.vec_type = args.vec
    if args.no_warm_start:
        # 注意：只把 il_weights 設成 None 沒用，run() 會回頭吃 config 的 il_warm_start
        config.setdefault("train", {})["il_warm_start"] = None
    il_weights = None if args.no_warm_start else args.il_weights
    result = trainer.run(
        total_timesteps=args.timesteps,
        resume_from=args.resume,
        il_weights=il_weights,
        use_wandb=args.wandb,
    )
    result["eval"] = trainer.evaluate(episodes=3)
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
