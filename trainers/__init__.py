"""訓練層：模仿學習（IL）與 PPO 微調。

注意：PPO 需要 ``stable-baselines3``，但 **IL 只需要 torch**。
因此 ``PPOTrainer`` / ``TransferReport`` 採延遲載入（PEP 562），
否則只跑 IL 的使用者會被強迫安裝 SB3，在 Colab 上就會看到
``ModuleNotFoundError: No module named 'stable_baselines3'``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trainers.il_trainer import ILConfig, ILTrainer

if TYPE_CHECKING:  # pragma: no cover - 只給型別檢查器
    from trainers.ppo_trainer import PPOTrainer, TransferReport

__all__ = ["ILConfig", "ILTrainer", "PPOTrainer", "TransferReport"]

_LAZY_ATTRS: dict[str, str] = {
    "PPOTrainer": "trainers.ppo_trainer",
    "TransferReport": "trainers.ppo_trainer",
}


def __getattr__(name: str) -> Any:
    """延遲載入 PPO 相關類別（避免 IL 被迫依賴 SB3）。"""

    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
