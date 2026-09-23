"""可執行腳本集合（以 ``python -m scripts.<name>`` 執行）。"""

from __future__ import annotations

import sys

# Windows 主控台預設不是 UTF-8，這裡統一改成 UTF-8 以免中文輸出變成亂碼
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - 非文字串流時忽略
        pass
