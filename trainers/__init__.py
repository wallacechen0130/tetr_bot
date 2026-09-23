"""訓練層：模仿學習（IL）與 PPO 微調。"""

from trainers.il_trainer import ILConfig, ILTrainer
from trainers.ppo_trainer import PPOTrainer, TransferReport

__all__ = ["ILConfig", "ILTrainer", "PPOTrainer", "TransferReport"]
