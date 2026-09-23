"""資料集 manifest：版本、來源設定與 shard 清單。"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def config_hash(config: dict[str, Any]) -> str:
    """設定內容的穩定雜湊（用來判斷資料集是否可比較）。"""

    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class DatasetManifest:
    """資料集中繼資料。"""

    name: str
    version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    num_samples: int = 0
    num_shards: int = 0
    shards: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    format: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    generator: str = "heuristic"
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config_hash"] = self.config_hash
        return data


def write_manifest(path: str | Path, manifest: DatasetManifest) -> Path:
    """寫出 manifest.json。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> DatasetManifest:
    """讀取 manifest.json（可傳目錄或檔案路徑）。"""

    target = Path(path)
    if target.is_dir():
        target = target / "manifest.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data.pop("config_hash", None)
    return DatasetManifest(**data)
