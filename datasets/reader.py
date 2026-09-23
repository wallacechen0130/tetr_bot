"""資料集讀取器：npz shards（主要）與 parquet（交換格式）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from datasets.manifest import DatasetManifest, load_manifest
from datasets.writer import SAMPLE_KEYS


class DatasetReader:
    """載入資料集並提供批次迭代。"""

    def __init__(self, root: str | Path, *, limit: int | None = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"找不到資料集：{self.root}")
        self.manifest: DatasetManifest = load_manifest(self.root)
        self.limit = limit

    # ------------------------------------------------------------------ 資料
    def shard_paths(self) -> list[Path]:
        paths = [self.root / shard for shard in self.manifest.shards]
        if not paths:
            paths = sorted((self.root / "shards").glob("*.npz"))
        return paths

    def arrays(self) -> dict[str, np.ndarray]:
        """把所有 shard 串成單一大陣列（MVP 規模足夠）。"""

        data: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_KEYS}
        for path in self.shard_paths():
            with np.load(path) as shard:
                for key in SAMPLE_KEYS:
                    data[key].append(shard[key])
        if not data["action"]:
            raise ValueError(f"資料集沒有任何 shard：{self.root}")
        merged = {key: np.concatenate(values, axis=0) for key, values in data.items()}
        if self.limit is not None:
            merged = {key: value[: self.limit] for key, value in merged.items()}
        return merged

    def parquet_arrays(self) -> dict[str, np.ndarray]:
        """從 parquet 讀回（驗證交換格式用）。"""

        import pandas as pd

        table = pd.read_parquet(self.root / "data.parquet")
        result: dict[str, np.ndarray] = {}
        for key in SAMPLE_KEYS:
            column = table[key].to_numpy()
            result[key] = np.stack([np.asarray(value) for value in column])
        if self.limit is not None:
            result = {key: value[: self.limit] for key, value in result.items()}
        return result

    def iter_batches(self, batch_size: int = 256, *, shuffle: bool = False, seed: int = 0) -> Iterator[dict[str, np.ndarray]]:
        arrays = self.arrays()
        total = arrays["action"].shape[0]
        indices = np.arange(total)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, total, batch_size):
            chunk = indices[start : start + batch_size]
            yield {key: value[chunk] for key, value in arrays.items()}

    def __len__(self) -> int:
        return int(self.manifest.num_samples or 0)


class TensorDataset:
    """PyTorch Dataset（torch 延遲匯入）。"""

    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> None:
        self.arrays = arrays
        self.indices = np.arange(arrays["action"].shape[0]) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch

        index = int(self.indices[item])
        board = torch.as_tensor(self.arrays["board"][index], dtype=torch.float32)
        vector = torch.as_tensor(self.arrays["vector"][index], dtype=torch.float32)
        mask = torch.as_tensor(self.arrays["mask"][index], dtype=torch.float32)
        action = torch.as_tensor(int(self.arrays["action"][index]), dtype=torch.long)
        topk_actions = torch.as_tensor(self.arrays["topk_actions"][index], dtype=torch.long)
        topk_scores = torch.as_tensor(self.arrays["topk_scores"][index], dtype=torch.float32)
        return {
            "board": board,
            "vector": vector,
            "mask": mask,
            "action": action,
            "topk_actions": topk_actions,
            "topk_scores": topk_scores,
        }


def split_indices(total: int, *, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """依固定種子切分 train/val。"""

    indices = np.arange(total)
    np.random.default_rng(seed).shuffle(indices)
    val_size = max(1, int(total * val_fraction)) if total > 1 else 0
    val = np.sort(indices[:val_size])
    train = np.sort(indices[val_size:])
    return train, val
