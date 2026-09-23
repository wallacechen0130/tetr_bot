"""純規則層（pure rules layer）。

本套件不含任何 Gymnasium 或 PyTorch 依賴，只負責 Tetris 的規則、幾何與計分，
可獨立測試與重用於資料產生、評估與未來的 TETR.IO 接入。
"""

from envs.engine.bag import SevenBag
from envs.engine.board import Board
from envs.engine.events import StepEvents
from envs.engine.garbage import GarbageConfig, GarbageManager
from envs.engine.piece import BOX_SIZE, PIECE_KINDS, Piece, piece_cells, spawn_origin
from envs.engine.placement import Placement, enumerate_placements
from envs.engine.rules import ClearType, Ruleset, load_ruleset
from envs.engine.simulator import GameConfig, GameSnapshot, TetrisSimulator

__all__ = [
    "Board",
    "SevenBag",
    "StepEvents",
    "GarbageConfig",
    "GarbageManager",
    "Placement",
    "enumerate_placements",
    "PIECE_KINDS",
    "BOX_SIZE",
    "Piece",
    "piece_cells",
    "spawn_origin",
    "ClearType",
    "Ruleset",
    "load_ruleset",
    "GameConfig",
    "GameSnapshot",
    "TetrisSimulator",
]
