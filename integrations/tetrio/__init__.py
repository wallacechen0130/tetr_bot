"""TETR.IO 接入（Phase 6，MVP 僅提供 stub）。"""

from integrations.tetrio.stub import (
    TetrioBoardRecognizer,
    TetrioInputController,
    TetrioScreenCapture,
)

__all__ = ["TetrioScreenCapture", "TetrioBoardRecognizer", "TetrioInputController"]
