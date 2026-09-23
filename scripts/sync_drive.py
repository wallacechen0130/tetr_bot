"""Google Drive 同步：把資料集 / checkpoint / log 複製到 Drive（或反向）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from trainers.common import write_json

DEFAULT_SUBDIRS = ("datasets", "checkpoints", "logs")


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
            relative = path.relative_to(src_root)
            target = dst_root / relative
            if target.exists() and not overwrite and target.stat().st_size == path.stat().st_size:
                report.skipped.append(str(relative))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            report.copied.append(str(relative))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="與 Google Drive 同步")
    parser.add_argument("--drive", default=os.environ.get("TETRIO_AI_DRIVE", ""), help="Drive 上的專案根目錄")
    parser.add_argument("--local", default=".")
    parser.add_argument("--direction", default="to_drive", choices=["to_drive", "from_drive"])
    parser.add_argument("--subdirs", default=",".join(DEFAULT_SUBDIRS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out", default="logs/sync_report.json")
    args = parser.parse_args()

    if not args.drive:
        raise SystemExit(
            "請設定 --drive 或環境變數 TETRIO_AI_DRIVE（例如 /content/drive/MyDrive/tetrio-ai）"
        )
    local_root = Path(args.local).resolve()
    drive_root = Path(args.drive).resolve()
    subdirs = tuple(part.strip() for part in args.subdirs.split(",") if part.strip())
    if args.direction == "to_drive":
        report = sync(local_root, drive_root, subdirs=subdirs, overwrite=args.overwrite)
    else:
        report = sync(drive_root, local_root, subdirs=subdirs, overwrite=args.overwrite)
    report.direction = args.direction
    write_json(local_root / args.out, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
