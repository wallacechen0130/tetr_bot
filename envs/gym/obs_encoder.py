"""Observation Space 編碼器（dense + board 影像 + action mask）。"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from envs.engine.board import (
    bumpiness_from_grid,
    holes_from_grid,
    max_height_from_grid,
)
from envs.engine.piece import GARBAGE_VALUE, PIECE_INDEX, PIECE_KINDS
from envs.engine.simulator import GameSnapshot
from envs.gym.placement import N_ACTIONS

BOARD_CHANNELS = 2
QUEUE_ROWS = 7
HOLD_DIM = len(PIECE_KINDS) + 1
COMBO_B2B_DIM = 4
GARBAGE_DIM = 3
STATS_DIM = 8

# flatten 的固定鍵順序（MLP 基線與 ONNX 匯出都依賴這個順序）
FLAT_KEYS: tuple[str, ...] = (
    "board",
    "piece_onehot",
    "hold_onehot",
    "hold_available",
    "next_queue",
    "combo_b2b",
    "garbage",
    "stats",
)

# 非棋盤欄位（網路與資料集使用的固定順序；action_mask 也一併編入）
VECTOR_KEYS: tuple[str, ...] = (
    "piece_onehot",
    "hold_onehot",
    "hold_available",
    "next_queue",
    "combo_b2b",
    "garbage",
    "stats",
    "action_mask",
)

VECTOR_DIM: int = (
    len(PIECE_KINDS) + HOLD_DIM + 1 + QUEUE_ROWS * len(PIECE_KINDS) + COMBO_B2B_DIM + GARBAGE_DIM + STATS_DIM + N_ACTIONS
)


def vector_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    """把非棋盤欄位攤平成固定長度向量（VECTOR_KEYS 順序）。"""

    parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in VECTOR_KEYS]
    return np.concatenate(parts).astype(np.float32)


class ObservationEncoder:
    """把 GameSnapshot 轉成固定形狀的 Dict observation。"""

    def __init__(
        self,
        *,
        visible_rows: int = 20,
        cols: int = 10,
        queue_rows: int = QUEUE_ROWS,
    ) -> None:
        self.visible_rows = int(visible_rows)
        self.cols = int(cols)
        self.queue_rows = int(queue_rows)

    # ------------------------------------------------------------------ 空間定義
    @property
    def board_shape(self) -> tuple[int, int, int]:
        return (BOARD_CHANNELS, self.visible_rows, self.cols)

    @property
    def feature_dim(self) -> int:
        return (
            int(np.prod(self.board_shape))
            + len(PIECE_KINDS)
            + HOLD_DIM
            + 1
            + self.queue_rows * len(PIECE_KINDS)
            + COMBO_B2B_DIM
            + GARBAGE_DIM
            + STATS_DIM
        )

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "board": spaces.Box(0.0, 1.0, shape=self.board_shape, dtype=np.float32),
                "piece_onehot": spaces.Box(0.0, 1.0, shape=(len(PIECE_KINDS),), dtype=np.float32),
                "hold_onehot": spaces.Box(0.0, 1.0, shape=(HOLD_DIM,), dtype=np.float32),
                "hold_available": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "next_queue": spaces.Box(0.0, 1.0, shape=(self.queue_rows, len(PIECE_KINDS)), dtype=np.float32),
                "combo_b2b": spaces.Box(0.0, 1.0, shape=(COMBO_B2B_DIM,), dtype=np.float32),
                "garbage": spaces.Box(0.0, 1.0, shape=(GARBAGE_DIM,), dtype=np.float32),
                "stats": spaces.Box(0.0, 1.0, shape=(STATS_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )

    # ------------------------------------------------------------------ 編碼
    def encode(self, snapshot: GameSnapshot, action_mask: np.ndarray) -> dict[str, np.ndarray]:
        grid = snapshot.board
        visible = grid[grid.shape[0] - self.visible_rows :]

        board = np.zeros(self.board_shape, dtype=np.float32)
        board[0] = (visible != 0).astype(np.float32)
        board[1] = (visible == GARBAGE_VALUE).astype(np.float32)

        piece_onehot = np.zeros(len(PIECE_KINDS), dtype=np.float32)
        piece_onehot[PIECE_INDEX[snapshot.current]] = 1.0

        hold_onehot = np.zeros(HOLD_DIM, dtype=np.float32)
        hold_onehot[PIECE_INDEX[snapshot.hold] if snapshot.hold else len(PIECE_KINDS)] = 1.0

        queue = np.zeros((self.queue_rows, len(PIECE_KINDS)), dtype=np.float32)
        for row, kind in enumerate(snapshot.next_queue[: self.queue_rows]):
            queue[row, PIECE_INDEX[kind]] = 1.0

        combo_b2b = np.array(
            [
                min(snapshot.combo, 20) / 20.0,
                1.0 if snapshot.b2b_chain >= 2 else 0.0,
                min(snapshot.b2b_chain, 20) / 20.0,
                min(snapshot.pieces_placed / 1000.0, 1.0),
            ],
            dtype=np.float32,
        )

        garbage = np.array(
            [
                min(snapshot.garbage_pending, 20) / 20.0,
                min(snapshot.attack_sent, 200) / 200.0,
                min(snapshot.garbage_received, 200) / 200.0,
            ],
            dtype=np.float32,
        )

        elapsed = max(snapshot.elapsed, 1e-6)
        pps = snapshot.pieces_placed / elapsed
        apm = (snapshot.attack_sent / elapsed) * 60.0
        remaining = snapshot.lines_remaining
        stats = np.array(
            [
                min(snapshot.lines_total / 40.0, 1.0),
                1.0 if remaining is None else min(remaining / 40.0, 1.0),
                min(snapshot.elapsed / 120.0, 1.0),
                min(pps / 5.0, 1.0),
                min(apm / 300.0, 1.0),
                min(max_height_from_grid(grid) / float(self.board_shape[1] * 2), 1.0),
                min(holes_from_grid(grid) / 100.0, 1.0),
                min(bumpiness_from_grid(grid) / 100.0, 1.0),
            ],
            dtype=np.float32,
        )

        return {
            "board": board,
            "piece_onehot": piece_onehot,
            "hold_onehot": hold_onehot,
            "hold_available": np.array([1.0 if snapshot.hold_available else 0.0], dtype=np.float32),
            "next_queue": queue,
            "combo_b2b": combo_b2b,
            "garbage": garbage,
            "stats": stats,
            "action_mask": np.asarray(action_mask, dtype=np.int8).reshape(N_ACTIONS).copy(),
        }

    def flatten(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """把 Dict observation 攤平成固定長度向量（含 action mask）。"""

        parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in FLAT_KEYS]
        parts.append(np.asarray(observation["action_mask"], dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    @property
    def flat_dim(self) -> int:
        return self.feature_dim + N_ACTIONS
