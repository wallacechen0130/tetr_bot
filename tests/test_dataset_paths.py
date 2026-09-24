"""資料集 manifest 的跨平台路徑測試。

Windows 產生的 manifest 會存反斜線（``shards\\worker00_00000.npz``），
在 Colab / Linux 上必須仍能正確解析，否則會出現
``FileNotFoundError: .../shards\\worker00_00000.npz``。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from datasets.manifest import DatasetManifest, write_manifest
from datasets.reader import DatasetReader, resolve_relative
from datasets.writer import SAMPLE_KEYS


def test_resolve_relative_handles_backslashes(tmp_path):
    resolved = resolve_relative(tmp_path, r"shards\worker00_00000.npz")
    assert resolved == tmp_path / "shards" / "worker00_00000.npz"
    # 關鍵：反斜線必須被當成「分隔符」拆開，而不是留在檔名裡
    tail = resolved.parts[len(tmp_path.parts) :]
    assert tail == ("shards", "worker00_00000.npz")
    assert all("\\" not in part for part in tail)
    assert resolve_relative(tmp_path, "shards/worker00_00000.npz") == tmp_path / "shards" / "worker00_00000.npz"
    assert resolve_relative(tmp_path, r"C:\data\shards\a.npz") == tmp_path / "data" / "shards" / "a.npz"
    assert resolve_relative(tmp_path, r"..\..\etc\passwd") == tmp_path / "etc" / "passwd"


def _write_dataset(root, shard_entries) -> None:
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "board": np.zeros((2, 20, 10), dtype=np.uint8),
        "vector": np.zeros((160,), dtype=np.float16),
        "mask": np.ones((80,), dtype=np.uint8),
        "action": np.int16(3),
        "topk_actions": np.array([3, 4, 5, -1, -1], dtype=np.int16),
        "topk_scores": np.array([1.0, 0.5, 0.1, -np.inf, -np.inf], dtype=np.float32),
        "context": np.zeros((4,), dtype=np.int32),
    }
    for index, _entry in enumerate(shard_entries):
        np.savez_compressed(shard_dir / f"worker{index:02d}_00000.npz", **{k: np.array([sample[k]]) for k in SAMPLE_KEYS})
    write_manifest(root / "manifest.json", DatasetManifest(name="t", num_samples=len(shard_entries), shards=list(shard_entries)))


def test_reader_handles_windows_style_manifest(tmp_path):
    """模擬 Windows 產生的 manifest（反斜線）在 Linux/Colab 上讀取。"""

    entries = [r"shards\worker00_00000.npz", r"shards\worker01_00000.npz"]
    _write_dataset(tmp_path, entries)

    reader = DatasetReader(tmp_path)
    paths = reader.shard_paths()
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    arrays = reader.arrays()
    assert arrays["action"].shape[0] == 2


def test_reader_falls_back_when_manifest_is_stale(tmp_path, capsys):
    """manifest 指向不存在的檔案時，改用實際掃到的 shard 並印出警告。"""

    _write_dataset(tmp_path, [r"shards\worker00_00000.npz"])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["shards"] = ["shards/not-there.npz"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    reader = DatasetReader(tmp_path)
    paths = reader.shard_paths()
    assert [path.name for path in paths] == ["worker00_00000.npz"]
    assert "改用" in capsys.readouterr().out


def test_reader_raises_clear_error_when_nothing_found(tmp_path):
    _write_dataset(tmp_path, [r"shards\worker00_00000.npz"])
    (tmp_path / "shards" / "worker00_00000.npz").unlink()

    reader = DatasetReader(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        reader.shard_paths()
    assert "Drive" in str(excinfo.value) or "shards" in str(excinfo.value)


def test_writer_stores_posix_paths(tmp_path):
    from datasets.writer import DatasetWriter

    sample = {
        "board": np.zeros((2, 20, 10), dtype=np.uint8),
        "vector": np.zeros((160,), dtype=np.float16),
        "mask": np.ones((80,), dtype=np.uint8),
        "action": np.int16(1),
        "topk_actions": np.array([1, 2, -1, -1, -1], dtype=np.int16),
        "topk_scores": np.array([1.0, 0.5, -np.inf, -np.inf, -np.inf], dtype=np.float32),
        "context": np.zeros((4,), dtype=np.int32),
    }
    writer = DatasetWriter(tmp_path, formats=("npz",), shard_size=1)
    writer.add(sample)
    manifest = writer.close()
    assert manifest.shards == ["shards/shard_00000.npz"]
    assert "\\" not in manifest.shards[0]


def test_limit_stops_reading_early(tmp_path):
    """--limit 應該只讀到足夠的筆數，而不是先載入全部再切片。"""

    _write_dataset(tmp_path, [r"shards\worker00_00000.npz", r"shards\worker01_00000.npz"])
    # 每個 shard 只有 1 筆，總共 2 筆
    reader = DatasetReader(tmp_path, limit=1)
    arrays = reader.arrays()
    assert arrays["action"].shape[0] == 1

    full = DatasetReader(tmp_path).arrays()
    assert full["action"].shape[0] == 2
