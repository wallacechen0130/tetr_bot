"""整合介面（Protocol）：設計給 Phase 6 的 TETR.IO 接入使用。

MVP 不安裝 OpenCV / mss / pynput，也不執行任何瀏覽器自動化；
這些 Protocol 讓未來的實作可以插進同一條推論管線。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ScreenCapture(Protocol):
    """螢幕擷取來源。"""

    def start(self) -> None:  # pragma: no cover - protocol
        ...

    def grab(self) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def stop(self) -> None:  # pragma: no cover - protocol
        ...


@runtime_checkable
class BoardRecognizer(Protocol):
    """把畫面轉成 20x10 的棋盤狀態。"""

    def recognize(self, frame: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


@runtime_checkable
class GameStateReader(Protocol):
    """讀取 hold / next queue / 垃圾行等周邊資訊。"""

    def read(self, frame: np.ndarray) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


@runtime_checkable
class InputController(Protocol):
    """鍵盤輸入控制器。"""

    def press(self, key: str, duration_s: float = 0.02) -> None:  # pragma: no cover - protocol
        ...

    def tap_sequence(self, keys: list[str]) -> None:  # pragma: no cover - protocol
        ...

    def release_all(self) -> None:  # pragma: no cover - protocol
        ...
