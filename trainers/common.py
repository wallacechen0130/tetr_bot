"""訓練共用工具：種子、裝置、checkpoint 與 observation 轉 tensor。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from envs.config import load_yaml
from policies.factory import VECTOR_KEYS


def set_seed(seed: int) -> None:
    """設定所有隨機來源的種子。"""

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - 本機無 GPU
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch 一定存在於訓練流程
        pass


def resolve_device(device: str | None = None) -> Any:
    """決定運算裝置（auto 會優先選 CUDA，其次 MPS，最後 CPU）。"""

    import torch

    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():  # pragma: no cover - Colab 才會走到
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def observation_to_tensors(observation: dict[str, np.ndarray]) -> tuple[Any, Any, Any]:
    """把單一 observation 轉成 (board, vector, mask) 的 1-batch tensor。"""

    import torch

    board = torch.as_tensor(np.asarray(observation["board"])[None], dtype=torch.float32)
    parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in VECTOR_KEYS]
    vector = torch.as_tensor(np.concatenate(parts)[None], dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(observation["action_mask"], dtype=np.float32)[None])
    return board, vector, mask


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """儲存完整 checkpoint（支援 resume）。"""

    import torch

    target = Path(path)
    ensure_dir(target.parent)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """載入 checkpoint，回傳附帶的 metadata。"""

    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return {key: value for key, value in payload.items() if key not in {"model", "optimizer"}} if isinstance(payload, dict) else {}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target


def load_training_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)
