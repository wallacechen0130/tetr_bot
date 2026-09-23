"""TETR.IO 整合 stub。

本模組刻意不 import OpenCV / mss / pynput：MVP 階段不引入任何
螢幕擷取或鍵盤注入依賴（安裝方式：``pip install -e .[tetrio]``）。

重要：TETR.IO 服務條款禁止在線上多人對戰使用自動化程式。
本專案的接入目標僅限離線練習模式（40L / Blitz），多人對戰請勿使用。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class _NotImplementedIntegration:
    """共用基底：明確指出尚未實作的功能與實作指引。"""

    feature: str = "integration"
    milestone: str = "Phase 6"

    def _todo(self) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} 屬於 {self.milestone} 的 {self.feature}，MVP 尚未實作。"
            "請參考 docs/01_architecture.md 的 TETR.IO Integration Pipeline 與 docs/03_roadmap.md。"
        )


class TetrioScreenCapture(_NotImplementedIntegration):
    """TETR.IO 視窗擷取（建議：mss + 視窗標題定位，fps ≥ 60）。"""

    feature = "Screen Capture"

    def __init__(self, window_title: str = "TETR.IO", fps: int = 60) -> None:
        self.window_title = window_title
        self.fps = int(fps)

    def start(self) -> None:
        self._todo()

    def grab(self) -> np.ndarray:
        self._todo()
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def stop(self) -> None:
        self._todo()


class TetrioBoardRecognizer(_NotImplementedIntegration):
    """棋盤辨識（建議：OpenCV 色彩遮罩 + 模板比對 + 網格校正）。"""

    feature = "Board Recognition"

    def __init__(self, rows: int = 20, cols: int = 10) -> None:
        self.rows = int(rows)
        self.cols = int(cols)

    def recognize(self, frame: np.ndarray) -> np.ndarray:
        self._todo()
        return np.zeros((self.rows, self.cols), dtype=np.uint8)


class TetrioInputController(_NotImplementedIntegration):
    """鍵盤控制（建議：pynput/ctypes SendInput + 每動作延遲與人類抖動）。"""

    feature = "Input Controller"

    def __init__(self, keymap: dict[str, str] | None = None) -> None:
        self.keymap: dict[str, Any] = keymap or {
            "left": "left",
            "right": "right",
            "rotate_cw": "x",
            "rotate_ccw": "z",
            "rotate_180": "a",
            "soft_drop": "down",
            "hard_drop": "space",
            "hold": "c",
        }

    def press(self, key: str, duration_s: float = 0.02) -> None:
        self._todo()

    def tap_sequence(self, keys: list[str]) -> None:
        self._todo()

    def release_all(self) -> None:
        self._todo()
