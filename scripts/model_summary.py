"""列出所有網路架構的層結構與參數量（docs/08_networks.md 的數據來源）。"""

from __future__ import annotations

import argparse
import json

from envs.gym.obs_encoder import VECTOR_DIM
from policies.extractors import BOARD_ENCODERS, build_board_encoder
from policies.factory import build_network, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="網路參數量統計")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out", default=None, help="輸出 JSON 路徑（可選）")
    args = parser.parse_args()

    report: dict[str, dict[str, float | int]] = {}
    for name in sorted(BOARD_ENCODERS):
        encoder = build_board_encoder(name)
        network = build_network(name, vector_dim=VECTOR_DIM, hidden_dim=args.hidden_dim)
        report[name] = {
            "encoder_params": count_parameters(encoder),
            "total_params": count_parameters(network),
            "vector_dim": VECTOR_DIM,
            "hidden_dim": args.hidden_dim,
        }
        print(f"{name:>14} | encoder={report[name]['encoder_params']:>9,} | total={report[name]['total_params']:>9,}")
    if args.out:
        from trainers.common import write_json

        write_json(args.out, report)
        print(f"已寫出 {args.out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
