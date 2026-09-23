"""離線測試用的 mock 實作（不需要螢幕或鍵盤）。"""

from __future__ import annotations

import numpy as np

from envs.engine.board import Board


class NullScreenCapture:
    """回傳空白畫面的假擷取器。"""

    def __init__(self, shape: tuple[int, int, int] = (480, 320, 3)) -> None:
        self.shape = shape
        self.started = False
        self.frames = 0

    def start(self) -> None:
        self.started = True

    def grab(self) -> np.ndarray:
        if not self.started:
            raise RuntimeError("請先呼叫 start()")
        self.frames += 1
        return np.zeros(self.shape, dtype=np.uint8)

    def stop(self) -> None:
        self.started = False


class SyntheticBoardRecognizer:
    """把 20x10 的整數矩陣當作「已辨識棋盤」，用來測試下游管線。"""

    def recognize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (20, 10):
            raise ValueError(f"預期 (20, 10) 的棋盤，收到 {frame.shape[:2]}")
        return (frame != 0).astype(np.uint8)

    @staticmethod
    def render(board: Board) -> np.ndarray:
        return board.visible.copy()


class RecordingInputController:
    """記錄所有按鍵事件，不做任何真實輸入。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.pressed: set[str] = set()

    def press(self, key: str, duration_s: float = 0.02) -> None:
        self.events.append((key, duration_s))
        self.pressed.add(key)

    def tap_sequence(self, keys: list[str]) -> None:
        for key in keys:
            self.press(key)

    def release_all(self) -> None:
        self.pressed.clear()

    def key_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key, _ in self.events:
            counts[key] = counts.get(key, 0) + 1
        return counts
