"""統一的進度條工具（tqdm 包裝，附 ETA）。

設計要點：
* 只要給 ``total``，tqdm 就會自動顯示剩餘時間（ETA）與速度。
* 非終端機環境（輸出被重導向、CI、pytest）自動停用，避免日誌被控制字元汙染。
* 沒安裝 tqdm 時回傳一個介面相同的空物件，程式不會因此壞掉。
"""

from __future__ import annotations

import sys
from typing import Any

try:  # pragma: no cover - 取決於環境是否安裝 tqdm
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def progress_enabled(explicit: bool | None = None) -> bool:
    """是否要顯示進度條。

    ``explicit`` 為 None 時依「標準錯誤是否為終端機」自動判斷，
    因此 VSCode 終端機會顯示、被重導向到檔案或 pytest 捕捉時會自動關閉。
    """

    if explicit is not None:
        return bool(explicit)
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # pragma: no cover - 極端環境
        return False


class _NullProgressBar:
    """tqdm 缺席時的替代品（介面與 tqdm 相容）。"""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.n = 0
        self.total: Any = None

    def update(self, n: float = 1) -> None:
        self.n += int(n)

    def close(self) -> None:
        return None

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_description(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def __enter__(self) -> _NullProgressBar:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __iter__(self) -> Any:
        return iter(())

    def __getattr__(self, _name: str) -> Any:  # pragma: no cover - 保險用
        return lambda *_args, **_kwargs: None


def progress_bar(
    total: int | float | None = None,
    *,
    desc: str = "",
    unit: str = "it",
    enable: bool | None = None,
    **kwargs: Any,
) -> Any:
    """建立進度條（含 ETA）。

    範例::

        with progress_bar(total=100, desc="收集資料", unit="episode") as bar:
            for _ in range(100):
                ...
                bar.update(1)
    """

    if tqdm is None or not progress_enabled(enable):
        return _NullProgressBar()
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(total=total, desc=desc, unit=unit, **kwargs)


def format_seconds(seconds: float) -> str:
    """把秒數格式化成 ``1h23m`` / ``4m05s`` / ``12.3s``。"""

    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"
