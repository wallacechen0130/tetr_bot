"""渲染工具（ASCII 與 numpy RGB，不需 pygame）。"""

from envs.render.ascii import render_board_ascii
from envs.render.rgb_array import board_to_rgb

__all__ = ["render_board_ascii", "board_to_rgb"]
