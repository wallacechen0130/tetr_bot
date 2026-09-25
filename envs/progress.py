"""統一的進度條工具（tqdm 包裝，附 ETA）。

設計要點：

* 只要給 ``total``，tqdm 就會自動顯示剩餘時間（ETA）與速度。
* **終端機**：正常顯示。
* **Colab / Jupyter**：使用 ``tqdm.auto``，在 notebook kernel 內會渲染成原生進度條 widget；
  以子行程執行時（``!python -m scripts...``）則用 ``\\r`` 更新，Colab 也會即時重畫。
* **其他非互動環境**（輸出被重導向、CI、pytest）：自動停用，避免日誌被控制字元汙染。
* 沒安裝 tqdm 時回傳一個介面相同的空物件，程式不會因此壞掉。
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:  # pragma: no cover - 取決於環境是否安裝 tqdm
    from tqdm.auto import tqdm as _tqdm_auto
except Exception:  # pragma: no cover - 沒有 tqdm 或 ipywidgets 異常時
    _tqdm_auto = None

try:  # pragma: no cover
    from tqdm import tqdm as _tqdm_std
except ImportError:  # pragma: no cover
    _tqdm_std = None


def in_notebook() -> bool:
    """是否在 Jupyter / Colab 這類 notebook 環境（含 Colab 的子行程）。"""

    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    # Colab 的子行程不會載入 ipykernel，但會繼承這些環境變數
    if any(key.startswith("COLAB_") for key in os.environ):
        return True
    try:  # pragma: no cover - 只有在 notebook 內才會有 IPython shell
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def progress_enabled(explicit: bool | None = None) -> bool:
    """是否要顯示進度條。

    ``explicit`` 為 None 時依「標準錯誤是否為終端機」自動判斷，
    因此 VSCode 終端機會顯示、被重導向到檔案或 pytest 捕捉時會自動關閉。
    """

    if explicit is not None:
        return bool(explicit)
    if in_notebook():
        return True
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

    implementation = _tqdm_auto or _tqdm_std
    if implementation is None or not progress_enabled(enable):
        return _NullProgressBar()
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    try:
        return implementation(total=total, desc=desc, unit=unit, **kwargs)
    except Exception:  # pragma: no cover - 例如 notebook widget 建不起來
        if _tqdm_std is None:
            return _NullProgressBar()
        return _tqdm_std(total=total, desc=desc, unit=unit, **kwargs)


def format_seconds(seconds: float) -> str:
    """把秒數格式化成 ``1h23m`` / ``4m05s`` / ``12.3s``。"""

    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"
