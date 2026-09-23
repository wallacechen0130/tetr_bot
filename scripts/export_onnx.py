"""把訓練好的模型匯出成 ONNX（供未來 C++ / 推論伺服器使用）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from envs.gym.obs_encoder import VECTOR_DIM
from policies.factory import build_network
from trainers.common import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="匯出 ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/il/best.pt")
    parser.add_argument("--network", default="resnet")
    parser.add_argument("--out", default="models/tetris_policy.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    import torch
    from torch import nn

    network = build_network(args.network, vector_dim=VECTOR_DIM)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        network.load_state_dict(state, strict=False)
        print(f"[export_onnx] 已載入 {checkpoint}")
    else:
        print(f"[export_onnx] 找不到 {checkpoint}，改為匯出隨機初始化權重")
    network.eval()

    class Wrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, board: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
            logits, _ = self.inner(board, vector, None)
            return logits

    wrapper = Wrapper(network)
    dummy_board = torch.zeros(args.batch, 2, 20, 10, dtype=torch.float32)
    dummy_vector = torch.zeros(args.batch, VECTOR_DIM, dtype=torch.float32)
    target = Path(args.out)
    ensure_dir(target.parent)
    try:
        torch.onnx.export(
            wrapper,
            (dummy_board, dummy_vector),
            str(target),
            input_names=["board", "vector"],
            output_names=["logits"],
            dynamic_axes={"board": {0: "batch"}, "vector": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # 使用 legacy exporter，避免額外需要 onnxscript
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - 缺少 onnxscript 時
        raise SystemExit(
            f"ONNX 匯出失敗（{exc}）。請先執行：pip install onnxscript，或改用 torch 的 legacy exporter。"
        ) from exc
    try:
        import onnx

        model = onnx.load(str(target))
        onnx.checker.check_model(model)
        print(f"[export_onnx] 已通過 onnx.checker：{target}")
    except ImportError:  # pragma: no cover - onnx 為可選依賴
        print("[export_onnx] 未安裝 onnx，略過格式驗證")
    except Exception as exc:  # pragma: no cover
        print(f"[export_onnx] onnx 驗證失敗：{exc}")
    size_kb = target.stat().st_size / 1024
    print(f"[export_onnx] 完成：{target} ({size_kb:.1f} KB)")
    print("提示：推論時請自行對 logits 套用 action mask（-1e9）。")


if __name__ == "__main__":
    main()
