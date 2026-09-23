"""設定檔載入工具。

所有模組都透過本檔讀取 YAML，確保「設定檔是唯一真實來源」，
程式碼中不出現散落的魔術數字。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """把相對路徑解析成專案根目錄下的絕對路徑。"""

    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """讀取 YAML 設定檔，回傳 dict（空檔案回傳空 dict）。"""

    target = resolve_path(path)
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"設定檔必須是 mapping：{target}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """遞迴合併兩個 dict，override 覆蓋 base，回傳新 dict。"""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """載入主設定檔，並把 rules / reward / difficulty / heuristic 內嵌。"""

    config = load_yaml(path)
    paths = config.get("paths", {})
    for key, sub_path in paths.items():
        config[key] = load_yaml(sub_path)
    return config
