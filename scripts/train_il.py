"""Phase 4：模仿學習訓練（Board → Action）。"""

from __future__ import annotations

import argparse
import json

from envs.config import load_yaml
from trainers.common import write_json
from trainers.il_trainer import ILConfig, ILTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="模仿學習訓練")
    parser.add_argument("--config", default="configs/il.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 筆樣本（除錯用）")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--network", default=None, choices=["small_cnn", "cnn", "resnet", "cnn_attention", "transformer"])
    parser.add_argument("--out", default="logs/il_result.json")
    args = parser.parse_args()

    config = ILConfig.from_dict(load_yaml(args.config))
    if args.data_root:
        config.data_root = args.data_root
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.network:
        config.network = args.network
    config.limit = args.limit

    trainer = ILTrainer(config, device=args.device)
    result = trainer.fit(epochs=args.epochs)
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
