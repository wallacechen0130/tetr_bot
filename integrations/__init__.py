"""外部整合層：螢幕擷取 / 棋盤辨識 / 輸入控制介面（MVP 僅提供介面與 mock）。"""

from integrations.base import BoardRecognizer, GameStateReader, InputController, ScreenCapture
from integrations.mock import NullScreenCapture, RecordingInputController

__all__ = [
    "BoardRecognizer",
    "GameStateReader",
    "InputController",
    "ScreenCapture",
    "NullScreenCapture",
    "RecordingInputController",
]
