"""整合介面與 stub 測試。"""

from __future__ import annotations

import numpy as np
import pytest

from envs.engine.board import Board
from envs.engine.piece import PIECE_VALUE
from integrations.base import BoardRecognizer, InputController, ScreenCapture
from integrations.mock import NullScreenCapture, RecordingInputController, SyntheticBoardRecognizer
from integrations.tetrio.stub import TetrioBoardRecognizer, TetrioInputController, TetrioScreenCapture


def test_protocols_are_structural() -> None:
    assert isinstance(NullScreenCapture(), ScreenCapture)
    assert isinstance(RecordingInputController(), InputController)
    assert isinstance(SyntheticBoardRecognizer(), BoardRecognizer)


def test_null_screen_capture_lifecycle() -> None:
    capture = NullScreenCapture(shape=(10, 10, 3))
    with pytest.raises(RuntimeError):
        capture.grab()
    capture.start()
    frame = capture.grab()
    assert frame.shape == (10, 10, 3)
    capture.stop()
    assert capture.frames == 1


def test_synthetic_recognizer_reads_board() -> None:
    board = Board()
    board.cells[-1, 0] = PIECE_VALUE["T"]
    frame = SyntheticBoardRecognizer.render(board)
    recognized = SyntheticBoardRecognizer().recognize(frame)
    assert recognized.shape == (20, 10)
    assert recognized[-1, 0] == 1
    with pytest.raises(ValueError):
        SyntheticBoardRecognizer().recognize(np.zeros((3, 3), dtype=np.uint8))


def test_recording_input_controller_counts_keys() -> None:
    controller = RecordingInputController()
    controller.tap_sequence(["left", "left", "rotate_cw", "hard_drop"])
    counts = controller.key_counts()
    assert counts["left"] == 2
    assert counts["hard_drop"] == 1
    controller.release_all()
    assert controller.pressed == set()


@pytest.mark.parametrize(
    "factory",
    [TetrioScreenCapture, TetrioBoardRecognizer, TetrioInputController],
)
def test_tetrio_stubs_are_not_implemented(factory) -> None:
    instance = factory()
    with pytest.raises(NotImplementedError):
        if isinstance(instance, TetrioScreenCapture):
            instance.start()
        elif isinstance(instance, TetrioBoardRecognizer):
            instance.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
        else:
            instance.press("x")


def test_stub_documents_terms_of_service():
    import integrations.tetrio.stub as stub

    assert "服務條款" in stub.__doc__
