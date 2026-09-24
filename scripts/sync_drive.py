"""Google Drive 同步：把資料集 / checkpoint / log 複製到 Drive（或反向）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from envs.config import find_google_drive_roots, normalize_user_path
from trainers.common import write_json

DEFAULT_SUBDIRS = ("datasets", "checkpoints", "logs")

# 不需要同步的本機產物（把 .pyc 搬到 Google Drive 沒有任何意義）
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", ".venv"}
)


def is_excluded(path: Path) -> bool:
    """判斷是否為不需同步的雜項檔案。"""

    if path.suffix == ".pyc":
        return True
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """計算檔案雜湊（同步驗證用）。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SyncReport:
    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    direction: str = "to_drive"

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "copied": len(self.copied),
            "skipped": len(self.skipped),
            "copied_files": self.copied,
        }


def sync(src_root: Path, dst_root: Path, *, subdirs: tuple[str, ...] = DEFAULT_SUBDIRS, overwrite: bool = False) -> SyncReport:
    """複製指定子目錄；已存在且大小相同者略過。"""

    report = SyncReport()
    for name in subdirs:
        src = src_root / name
        if not src.exists():
            continue
        for path in src.rglob("*"):
            if path.is_dir():
                continue
            if is_excluded(path.relative_to(src_root)):
                report.skipped.append(str(path.relative_to(src_root)))
                continue
            relative = path.relative_to(src_root)
            target = dst_root / relative
            if target.exists() and not overwrite and target.stat().st_size == path.stat().st_size:
                report.skipped.append(str(relative))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            report.copied.append(str(relative))
    return report


def resolve_drive_root(raw: str | None) -> Path:
    """解析並驗證 Drive 根目錄，失敗時給出可執行的錯誤訊息。

    Google Drive 桌面版的「我的雲端硬碟」資料夾名稱會跟著系統語系
    （中文版是 ``我的雲端硬碟``，英文版是 ``My Drive``），
    而且它**不是** ``MyDrive``；直接 mkdir 不存在的路徑會得到
    ``FileNotFoundError: [WinError 2]`` 這種看不懂的錯誤。
    """

    if not raw:
        raise SystemExit(
            "請設定 --drive 或環境變數 TETRIO_AI_DRIVE\n"
            '  本機範例：--drive "G:\\我的雲端硬碟\\tetrio-ai"\n'
            "  英文版 Drive：--drive \"G:\\My Drive\\tetrio-ai\"\n"
            "  Colab：--drive \"/content/drive/MyDrive/tetrio-ai\"\n"
            "  不確定路徑時可先執行：python -m scripts.sync_drive --list-drives"
        )

    root = normalize_user_path(raw)
    if root.exists():
        return root

    candidates = find_google_drive_roots()
    lines = [f"找不到 Drive 資料夾：{root}", ""]
    if candidates:
        lines.append("這台電腦偵測到的 Google Drive 資料夾：")
        lines.extend(f"  {candidate}" for candidate in candidates)
        lines.append("")
        lines.append(f'建議：--drive "{candidates[0] / "tetrio-ai"}"')
    else:
        lines.append("沒有偵測到任何 Google Drive 資料夾，請確認已安裝並登入 Google Drive 桌面版。")
    lines += [
        "",
        "提醒：Google Drive 桌面版的資料夾叫「我的雲端硬碟」或「My Drive」，不是 MyDrive。",
    ]
    raise SystemExit("\n".join(lines))


def list_drive_roots() -> None:
    """列出偵測到的 Google Drive 資料夾（--list-drives）。"""

    roots = find_google_drive_roots()
    if not roots:
        print("沒有偵測到 Google Drive 資料夾（請確認已安裝並登入 Google Drive 桌面版）。")
        return
    print("偵測到的 Google Drive 資料夾：")
    for root in roots:
        print(f"  {root}")
        print(f'    建議的專案路徑：--drive "{root / "tetrio-ai"}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="與 Google Drive 同步")
    parser.add_argument("--drive", default=os.environ.get("TETRIO_AI_DRIVE", ""), help="Drive 上的專案根目錄")
    parser.add_argument("--local", default=".")
    parser.add_argument("--direction", default="to_drive", choices=["to_drive", "from_drive"])
    parser.add_argument("--subdirs", default=",".join(DEFAULT_SUBDIRS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out", default="logs/sync_report.json")
    parser.add_argument("--list-drives", action="store_true", help="列出偵測到的 Google Drive 資料夾後結束")
    args = parser.parse_args()

    if args.list_drives:
        list_drive_roots()
        return

    local_root = normalize_user_path(args.local).resolve()
    drive_root = resolve_drive_root(args.drive)
    subdirs = tuple(part.strip() for part in args.subdirs.split(",") if part.strip())
    print(f"本機：{local_root}")
    print(f"Drive：{drive_root}")
    try:
        if args.direction == "to_drive":
            report = sync(local_root, drive_root, subdirs=subdirs, overwrite=args.overwrite)
        else:
            report = sync(drive_root, local_root, subdirs=subdirs, overwrite=args.overwrite)
    except OSError as exc:
        raise SystemExit(
            f"同步失敗：{exc}\n"
            f"請確認 Drive 路徑存在且可寫入：{drive_root}\n"
            "提示：Google Drive 的虛擬磁碟不允許建立最上層資料夾，"
            "路徑必須指向既有的「我的雲端硬碟」（或 My Drive）底下的資料夾。"
        ) from exc
    report.direction = args.direction
    write_json(local_root / args.out, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
