"""資料集寫入器：支援 .npz 分片與 parquet。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from datasets.manifest import DatasetManifest, write_manifest

SAMPLE_KEYS: tuple[str, ...] = (
    "board",
    "vector",
    "mask",
    "action",
    "topk_actions",
    "topk_scores",
    "context",
)

_DTYPES: dict[str, Any] = {
    "board": np.uint8,
    "vector": np.float16,
    "mask": np.uint8,
    "action": np.int16,
    "topk_actions": np.int16,
    "topk_scores": np.float32,
    "context": np.int32,
}


def _as_stacked(samples: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([np.asarray(sample[key]) for sample in samples]).astype(_DTYPES[key], copy=False)


def _sample_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """把 numpy 陣列字典轉成規定的 dtype。"""

    return {key: np.asarray(arrays[key]).astype(_DTYPES[key], copy=False) for key in SAMPLE_KEYS}


class DatasetWriter:
    """累積樣本並輸出 shards/ + manifest.json（可選 parquet）。"""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "heuristic-v1",
        formats: Iterable[str] = ("npz", "parquet"),
        shard_size: int = 5000,
        config: dict[str, Any] | None = None,
        generator: str = "heuristic",
        shard_prefix: str = "shard",
        write_manifest_file: bool = True,
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.formats = tuple(formats)
        self.shard_size = int(shard_size)
        self.config = config or {}
        self.generator = generator
        self.shard_prefix = shard_prefix
        self.write_manifest_file = write_manifest_file
        self.shard_dir = self.root / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list[dict[str, np.ndarray]] = []
        self._shards: list[str] = []
        self._files: list[str] = []
        self._num_samples = 0
        self._parquet_writer: Any = None
        self._parquet_path: Path | None = None

    # ------------------------------------------------------------------ 寫入
    def add(self, sample: dict[str, np.ndarray]) -> None:
        missing = [key for key in SAMPLE_KEYS if key not in sample]
        if missing:
            raise KeyError(f"樣本缺少欄位：{missing}")
        self._buffer.append(sample)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def add_many(self, samples: Iterable[dict[str, np.ndarray]]) -> None:
        for sample in samples:
            self.add(sample)

    def flush(self) -> None:
        if not self._buffer:
            return
        index = len(self._shards)
        arrays = {key: _as_stacked(self._buffer, key) for key in SAMPLE_KEYS}
        if "npz" in self.formats:
            shard_path = self.shard_dir / f"{self.shard_prefix}_{index:05d}.npz"
            np.savez_compressed(shard_path, **arrays)
            self._shards.append(str(shard_path.relative_to(self.root)))
        if "parquet" in self.formats:
            self._write_parquet(arrays)
        self._num_samples += len(self._buffer)
        self._buffer.clear()

    def _write_parquet(self, arrays: dict[str, np.ndarray]) -> None:
        import pyarrow.parquet as pq

        if self._parquet_writer is None:
            self._parquet_path = self.root / "data.parquet"
            self._parquet_writer = pq.ParquetWriter(self._parquet_path, parquet_schema(), compression="zstd")
        self._parquet_writer.write_table(to_parquet_table(arrays))

    def close(self) -> DatasetManifest:
        """收尾並回傳 manifest。"""

        self.flush()
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            assert self._parquet_path is not None
            self._files.append(str(self._parquet_path.relative_to(self.root)))
        manifest = DatasetManifest(
            name=self.name,
            num_samples=self._num_samples,
            num_shards=len(self._shards),
            shards=list(self._shards),
            files=list(self._files),
            format=list(self.formats),
            config=self.config,
            generator=self.generator,
        )
        if self.write_manifest_file:
            write_manifest(self.root / "manifest.json", manifest)
        return manifest

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parquet_schema() -> Any:
    """資料集的 parquet schema（npz 與 parquet 共用欄位定義）。"""

    import pyarrow as pa

    return pa.schema(
        [
            ("board", pa.list_(pa.uint8())),
            ("vector", pa.list_(pa.float32())),
            ("mask", pa.list_(pa.uint8())),
            ("action", pa.int16()),
            ("topk_actions", pa.list_(pa.int16())),
            ("topk_scores", pa.list_(pa.float32())),
            ("context", pa.list_(pa.int32())),
        ]
    )


def to_parquet_table(arrays: dict[str, np.ndarray]) -> Any:
    """把一批樣本轉成 pyarrow Table。"""

    import pyarrow as pa

    return pa.table(
        {
            "board": pa.array([row.reshape(-1).tolist() for row in arrays["board"]], type=pa.list_(pa.uint8())),
            "vector": pa.array(
                [row.reshape(-1).astype(np.float32).tolist() for row in arrays["vector"]],
                type=pa.list_(pa.float32()),
            ),
            "mask": pa.array([row.reshape(-1).tolist() for row in arrays["mask"]], type=pa.list_(pa.uint8())),
            "action": pa.array(arrays["action"], type=pa.int16()),
            "topk_actions": pa.array([row.tolist() for row in arrays["topk_actions"]], type=pa.list_(pa.int16())),
            "topk_scores": pa.array([row.tolist() for row in arrays["topk_scores"]], type=pa.list_(pa.float32())),
            "context": pa.array([row.tolist() for row in arrays["context"]], type=pa.list_(pa.int32())),
        }
    )


def write_parquet_from_shards(root: str | Path, shard_paths: Iterable[Path], *, batch_size: int = 4000) -> Path:
    """把多個 npz shard 合併成單一 parquet（逐 shard 串流寫入）。"""

    import pyarrow.parquet as pq

    root = Path(root)
    target = root / "data.parquet"
    writer = pq.ParquetWriter(target, parquet_schema(), compression="zstd")
    try:
        for path in shard_paths:
            with np.load(path) as shard:
                total = shard["action"].shape[0]
                for start in range(0, total, batch_size):
                    stop = min(start + batch_size, total)
                    chunk = {key: shard[key][start:stop] for key in SAMPLE_KEYS}
                    writer.write_table(to_parquet_table(chunk))
    finally:
        writer.close()
    return target


def build_sample(
    observation: dict[str, np.ndarray],
    *,
    action: int,
    candidates: list[Any],
    context: list[int] | tuple[int, ...],
    topk: int = 5,
) -> dict[str, np.ndarray]:
    """把一次決策打包成訓練樣本（State → Best Move）。

    ``candidates`` 需為已依分數排序的候選清單（具備 ``action`` 與 ``score`` 屬性）。
    """

    from envs.gym.obs_encoder import vector_from_observation

    top_actions = np.full(topk, -1, dtype=np.int16)
    top_scores = np.full(topk, -np.inf, dtype=np.float32)
    for index, candidate in enumerate(candidates[:topk]):
        top_actions[index] = int(candidate.action)
        top_scores[index] = float(candidate.score)
    board = (np.asarray(observation["board"]) > 0.5).astype(np.uint8)
    return {
        "board": board,
        "vector": vector_from_observation(observation).astype(np.float16),
        "mask": np.asarray(observation["action_mask"], dtype=np.uint8).reshape(-1),
        "action": np.asarray(int(action), dtype=np.int16),
        "topk_actions": top_actions,
        "topk_scores": top_scores,
        "context": np.asarray(context, dtype=np.int32),
    }
