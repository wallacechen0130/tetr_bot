"""訓練 callback：checkpoint、最佳模型與課程排程。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.progress import progress_bar
from trainers.common import ensure_dir


class BestModelCallback(BaseCallback):
    """依平均 episode reward 保存最佳模型。"""

    def __init__(
        self,
        save_path: str | Path = "checkpoints/ppo/best.zip",
        *,
        verbose: int = 0,
        min_episodes: int = 3,
    ) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.min_episodes = int(min_episodes)
        self.best_mean = -np.inf
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
        if len(self.episode_rewards) >= self.min_episodes:
            recent = float(np.mean(self.episode_rewards[-20:]))
            if recent > self.best_mean:
                self.best_mean = recent
                ensure_dir(self.save_path.parent)
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"[BestModelCallback] 新最佳平均 reward {recent:.2f} -> {self.save_path}")
        return True


class CurriculumCallback(BaseCallback):
    """依訓練進度更新環境的 reward 課程比重。"""

    def __init__(self, *, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        total = int(getattr(self.model, "_total_timesteps", 0) or 0)
        progress = float(self.num_timesteps) / max(1, total)
        try:
            for env in self.model.get_env().envs:  # type: ignore[union-attr]
                unwrapped = env.unwrapped
                if hasattr(unwrapped, "reward"):
                    unwrapped.reward.set_progress(progress)
        except Exception:  # pragma: no cover - 環境不支援時忽略
            pass
        return True


def make_wandb_callback(project: str = "tetrio-ai", mode: str = "offline") -> Any | None:
    """建立 W&B callback（未安裝 wandb 時回傳 None）。"""

    try:
        from wandb.integration.sb3 import WandbCallback
    except Exception:  # pragma: no cover - wandb 為可選依賴
        return None
    return WandbCallback(gradient_save_freq=1000, model_save_path=None, verbose=0)


class TqdmProgressCallback(BaseCallback):
    """顯示 PPO 訓練進度（timesteps）與 ETA，並把 logger 指標放到 postfix。

    之所以不用 SB3 內建的 ``progress_bar=True``：那個選項需要額外的 ``rich`` 依賴，
    而本專案只想用已經裝好的 tqdm。
    """

    def __init__(
        self,
        *,
        desc: str = "PPO 訓練",
        enable: bool | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.desc = desc
        self.enable = enable
        self.bar: Any = None
        self._last_timesteps = 0

    def _on_training_start(self) -> None:
        total = int(getattr(self.model, "_total_timesteps", 0) or 0)
        self.bar = progress_bar(
            total=total or None,
            desc=self.desc,
            unit="step",
            enable=self.enable,
            position=0,
        )
        self._last_timesteps = 0

    def _on_step(self) -> bool:
        if self.bar is None:
            return True
        delta = int(self.num_timesteps) - self._last_timesteps
        if delta > 0:
            self._last_timesteps += delta
            self.bar.update(delta)
        self.bar.set_postfix(self._postfix(), refresh=False)
        return True

    def _on_training_end(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def _postfix(self) -> dict[str, str]:
        values = getattr(self.model.logger, "name_to_value", None) or {}
        result: dict[str, str] = {}
        for key, label in (
            ("time/fps", "fps"),
            ("rollout/ep_rew_mean", "rew"),
            ("rollout/ep_len_mean", "len"),
            ("train/approx_kl", "kl"),
        ):
            if key in values:
                try:
                    result[label] = f"{float(values[key]):.4g}"
                except (TypeError, ValueError):  # pragma: no cover - logger 值異常時忽略
                    continue
        return result
