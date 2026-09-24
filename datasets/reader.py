"""資料集讀取器：npz shards（主要）與 parquet（交換格式）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from datasets.manifest import DatasetManifest, load_manifest
from datasets.writer import SAMPLE_KEYS


def resolve_relative(root: str | Path, relative: str) -> Path:
    """把 manifest 內的相對路徑轉成實際路徑（跨平台）。

    manifest 可能是在 Windows 上產生的，裡面會是
    ``shards\\worker00_00000.npz``；到了 Colab / Linux，反斜線不是分隔符，
    會被當成檔名的一部分，於是出現 ``.../shards\\worker00_00000.npz``
    找不到檔案的錯誤。這裡統一轉成 POSIX 分隔符再組合，並忽略開頭的磁碟代號。
    """

    root = Path(root)
    parts = [part for part in str(relative).replace("\\", "/").split("/") if part not in ("", ".", "..")]
    if parts and parts[0].endswith(":"):  # 去掉 Windows 磁碟代號（例如 C:）
        parts = parts[1:]
    if not parts:
        return root
    return root.joinpath(*parts)


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
        """回傳實際存在的 shard 路徑。

        跨平台與部分同步的容錯：

        * manifest 內可能是 Windows 反斜線路徑 → 用 ``resolve_relative`` 正規化。
        * Drive 只同步了一部分、或 manifest 與檔案對不上時，改用 ``shards/*.npz``
          實際掃到的檔案，並印出明顯警告（不讓訓練直接爆掉）。
        """

        declared = [resolve_relative(self.root, shard) for shard in self.manifest.shards]
        existing = [path for path in declared if path.exists()]
        if declared and len(existing) == len(declared):
            return declared

        fallback = sorted((self.root / "shards").glob("*.npz"))
        if fallback:
            if declared:
                print(
                    f"[DatasetReader] manifest 列出 {len(declared)} 個 shard，"
                    f"實際只找到 {len(existing)} 個（例：{declared[0].name}）。"
                    f"改用 {self.root / 'shards'} 下的 {len(fallback)} 個 npz。"
                    "若數字不符，請重新同步資料集。"
                )
            return fallback
        if declared:
            raise FileNotFoundError(
                f"manifest 指向 {len(declared)} 個不存在的 shard，例如：{declared[0]}\n"
                f"請確認 {self.root / 'shards'} 內有 .npz 檔案（可能是 Drive 尚未同步完成）。"
            )
        return []

    def arrays(self) -> dict[str, np.ndarray]:
        """把所有 shard 串成單一大陣列。

        ``limit`` 會在讀滿指定筆數後就停止讀檔，不會先把整個資料集載進記憶體
        （147 萬筆的資料集約 1.2 GB，用 ``--limit`` 做小樣本測試時不需要全載）。
        """

        data: dict[str, list[np.ndarray]] = {key: [] for key in SAMPLE_KEYS}
        collected = 0
        for path in self.shard_paths():
            if self.limit is not None and collected >= self.limit:
                break
            with np.load(path) as shard:
                available = int(shard["action"].shape[0])
                take = available if self.limit is None else min(available, self.limit - collected)
                for key in SAMPLE_KEYS:
                    data[key].append(shard[key][:take])
                collected += take
        if not data["action"]:
            raise ValueError(f"資料集沒有任何 shard：{self.root}")
        merged = {key: np.concatenate(values, axis=0) for key, values in data.items()}
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
