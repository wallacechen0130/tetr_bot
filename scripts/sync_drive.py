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
# Windows / Google Drive 產生的中繼檔，搬到哪都沒用
EXCLUDED_FILE_NAMES: frozenset[str] = frozenset({"desktop.ini", "Thumbs.db", ".DS_Store"})
# 這個指令是同步「資料與模型」，不是同步原始碼。
# 把 .py / .md / .yaml 也同步會讓 --direction from_drive 用 Drive 上的舊程式碼
# 覆蓋掉本機較新的原始碼（實際踩過這個坑）。
EXCLUDED_SUFFIXES: frozenset[str] = frozenset(
    {".py", ".pyc", ".pyi", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json5"}
)


def is_excluded(path: Path) -> bool:
    """判斷是否為不需同步的雜項檔案。"""

    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def should_copy(src: Path, dst: Path, *, overwrite: bool = False) -> tuple[bool, str]:
    """決定是否要複製這個檔案，回傳 (是否複製, 原因)。

    除了大小不同就複製之外，這裡也用**修改時間**判斷新舊：
    目標檔比來源新時不覆蓋，避免 ``--direction from_drive`` 用較舊的
    Drive 內容蓋掉本機剛更新過的檔案。
    """

    if not dst.exists():
        return True, "new"
    if overwrite:
        return True, "overwrite"
    src_stat = src.stat()
    dst_stat = dst.stat()
    if dst_stat.st_size == src_stat.st_size:
        return False, "same-size"
    if dst_stat.st_mtime > src_stat.st_mtime + 1.0:
        return False, "target-newer"
    return True, "source-newer"


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
    kept_newer: list[str] = field(default_factory=list)
    direction: str = "to_drive"

    def to_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "copied": len(self.copied),
            "skipped": len(self.skipped),
            "kept_newer": len(self.kept_newer),
            "copied_files": self.copied,
            "kept_newer_files": self.kept_newer,
        }


def sync(src_root: Path, dst_root: Path, *, subdirs: tuple[str, ...] = DEFAULT_SUBDIRS, overwrite: bool = False) -> SyncReport:
    """複製指定子目錄下的資料與模型。

    * 原始碼與文件（``.py`` / ``.md`` / ``.yaml`` …）一律不同步。
    * 大小相同者略過；大小不同時再比修改時間，**不會**用較舊的檔案蓋掉較新的檔案
      （除非加 ``--overwrite``）。
    """

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
            copy_it, reason = should_copy(path, target, overwrite=overwrite)
            if not copy_it:
                if reason == "target-newer":
                    report.kept_newer.append(str(relative))
                else:
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
    summary = report.to_dict()
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_files")}, indent=2, ensure_ascii=False))
    if report.kept_newer:
        print(f"\n有 {len(report.kept_newer)} 個檔案因為本機版本較新而保留（未覆蓋）：")
        for name in report.kept_newer[:10]:
            print("  ", name)
        if len(report.kept_newer) > 10:
            print(f"   ...（其餘 {len(report.kept_newer) - 10} 個見 {args.out}）")
        print("確定要用來源覆蓋時，加上 --overwrite。")
    if report.copied:
        print(f"\n已複製 {len(report.copied)} 個檔案，清單見 {args.out}")


if __name__ == "__main__":
    main()
