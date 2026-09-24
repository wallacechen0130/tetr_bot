"""設定檔載入工具。

所有模組都透過本檔讀取 YAML，確保「設定檔是唯一真實來源」，
程式碼中不出現散落的魔術數字。
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Google Drive 桌面版的「我的雲端硬碟」資料夾名稱會跟著系統語系改變
MY_DRIVE_FOLDER_NAMES: tuple[str, ...] = (
    "My Drive",
    "MyDrive",
    "我的雲端硬碟",
    "我的云端硬盘",
    "マイドライブ",
    "내 드라이브",
    "Мой диск",
    "Mon Drive",
    "Meine Ablage",
    "Mijn Drive",
)


def normalize_user_path(value: str | Path) -> Path:
    """把使用者手打或複製貼上的路徑正規化。

    處理常見的地雷：

    * 前後空白與多餘的引號（複製貼上常帶 ``"`` 或 ``'``）
    * 重複的分隔符（``G:\\\\MyDrive`` → ``G:\\MyDrive``），但保留 UNC 開頭的 ``\\\\``
    * 正斜線與反斜線混用（``G:/我的雲端硬碟/tetrio-ai``）
    * ``~`` 與環境變數（``%USERPROFILE%``）
    * 結尾多餘的分隔符（``G:\\MyDrive\\`` → ``G:\\MyDrive``，但保留磁碟根 ``C:\\``）
    """

    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = os.path.expandvars(os.path.expanduser(text))
    if not text:
        return Path("")

    if os.name == "nt":
        text = text.replace("/", "\\")
        prefix = ""
        match = re.match(r"^(\\\\+)(?=\S)", text)
        if match:  # UNC 路徑（\\server\share）要保留開頭的雙反斜線
            prefix = "\\\\"
            text = text[len(match.group(1)) :]
        text = re.sub(r"\\{2,}", "\\\\", text)
        text = prefix + text
        stripped = text.rstrip("\\")
        if not stripped:
            return Path(text)
        text = stripped + "\\" if re.fullmatch(r"[A-Za-z]:", stripped) else stripped
    else:
        text = re.sub(r"/{2,}", "/", text)
    return Path(text)


def find_google_drive_roots() -> list[Path]:
    """掃描所有磁碟代號，找出 Google Drive 的「我的雲端硬碟」資料夾。

    找不到時回傳空清單（非 Windows 環境一律回傳空清單）。
    """

    roots: list[Path] = []
    if os.name != "nt":
        return roots
    for letter in string.ascii_uppercase:
        base = Path(f"{letter}:\\")
        try:
            if not base.exists():
                continue
        except OSError:  # pragma: no cover - 沒有權限的磁碟
            continue
        for name in MY_DRIVE_FOLDER_NAMES:
            candidate = base / name
            try:
                if candidate.is_dir():
                    roots.append(candidate)
                    break
            except OSError:  # pragma: no cover
                continue
    return roots


def resolve_path(path: str | Path) -> Path:
    """把相對路徑解析成專案根目錄下的絕對路徑。"""

    p = normalize_user_path(path)
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
