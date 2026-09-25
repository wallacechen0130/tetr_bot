"""PPO 微調：SB3 + sb3-contrib MaskablePPO。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from envs.config import resolve_path
from policies.sb3_extractor import TetrisFeaturesExtractor
from trainers.callbacks import (
    BestModelCallback,
    CurriculumCallback,
    TqdmProgressCallback,
    make_wandb_callback,
)
from trainers.common import ensure_dir, set_seed


@dataclass
class TransferReport:
    """IL → PPO 權重轉移結果。"""

    loaded: int = 0
    total: int = 0
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"loaded": self.loaded, "total": self.total, "source": self.source}


def make_env_worker(env_id: str, rank: int, seed: int, kwargs: dict[str, Any]):
    """建立單一環境（module-level 以便 SubprocVecEnv pickle）。"""

    def _init():
        import gymnasium as gym
        from sb3_contrib.common.wrappers import ActionMasker

        import envs  # noqa: F401 - 匯入即完成環境註冊

        env = gym.make(env_id, **kwargs)
        env = ActionMasker(env, lambda wrapped: wrapped.unwrapped.action_masks())
        env.reset(seed=seed + rank)
        return env

    return _init


class PPOTrainer:
    """包裝 MaskablePPO 的訓練器。"""

    def __init__(self, config: dict[str, Any], *, seed: int | None = None) -> None:
        self.config = config
        model_cfg = config.get("model", {})
        env_cfg = config.get("env", {})
        self.seed = int(seed if seed is not None else config.get("train", {}).get("seed", 12345))
        set_seed(self.seed)
        self.env_id = str(env_cfg.get("env_id", "TetrisSurvival-v0"))
        self.n_envs = int(env_cfg.get("n_envs", 4))
        self.vec_type = str(env_cfg.get("vec", "dummy"))
        self.env_kwargs: dict[str, Any] = {"opponent_apm": float(env_cfg.get("opponent_apm", 0.0))}
        if "mode" in env_cfg:
            self.env_kwargs["mode"] = str(env_cfg["mode"])
        self.network = str(model_cfg.get("network", "resnet"))
        self.policy_kwargs: dict[str, Any] = {
            "features_extractor_class": TetrisFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 256, "network": self.network},
            "net_arch": list(model_cfg.get("net_arch", [256, 256])),
        }
        self.hyperparams: dict[str, Any] = {
            "gamma": float(model_cfg.get("gamma", 0.997)),
            "learning_rate": float(model_cfg.get("learning_rate", 2.5e-4)),
            "n_steps": int(model_cfg.get("n_steps", 512)),
            "batch_size": int(model_cfg.get("batch_size", 1024)),
            "n_epochs": int(model_cfg.get("n_epochs", 4)),
            "clip_range": float(model_cfg.get("clip_range", 0.2)),
            "ent_coef": float(model_cfg.get("ent_coef", 0.01)),
            "vf_coef": float(model_cfg.get("vf_coef", 0.5)),
            "gae_lambda": float(model_cfg.get("gae_lambda", 0.95)),
            "max_grad_norm": float(model_cfg.get("max_grad_norm", 0.5)),
        }
        self.checkpoint_dir = Path(config.get("checkpoint", {}).get("dir", "checkpoints/ppo"))
        self.tensorboard_dir = str(config.get("logging", {}).get("tensorboard", "logs/tensorboard"))
        self.env: Any = None
        self.model: Any = None
        self.transfer = TransferReport()

    # ------------------------------------------------------------------ 環境
    def build_vec_env(self, *, n_envs: int | None = None, vec_type: str | None = None) -> Any:
        """建立向量化環境（SubprocVecEnv 失敗時自動退回 DummyVecEnv）。

        外層一定要包 ``VecMonitor``：SB3 的 episode 統計（``info["episode"]``、
        TensorBoard 的 ``rollout/ep_rew_mean``）是由 Monitor 產生的。
        少了它 ``BestModelCallback`` 永遠不會存 ``best.zip``，
        TensorBoard 也看不到 reward 曲線。
        """

        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

        n_envs = int(n_envs or self.n_envs)
        vec_type = (vec_type or self.vec_type).lower()
        factories = [make_env_worker(self.env_id, rank, self.seed, self.env_kwargs) for rank in range(n_envs)]
        if vec_type == "subproc" and n_envs > 1:
            try:
                base_env: Any = SubprocVecEnv(factories, start_method="spawn")
            except Exception as exc:  # pragma: no cover - 平台限制時退回
                print(f"[PPOTrainer] SubprocVecEnv 失敗（{exc}），改用 DummyVecEnv")
                base_env = DummyVecEnv(factories)
        else:
            base_env = DummyVecEnv(factories)

        ensure_dir(self.tensorboard_dir)
        self.env = VecMonitor(base_env, filename=str(Path(self.tensorboard_dir) / "monitor.csv"))
        return self.env

    # ------------------------------------------------------------------ 模型
    def _tensorboard_log(self, override: str | None = None) -> str | None:
        """若環境未安裝 tensorboard 則回傳 None（避免訓練直接失敗）。"""

        target = override or self.tensorboard_dir
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            print("[PPOTrainer] 未安裝 tensorboard，改用 stdout 記錄（pip install tensorboard 可開啟）")
            return None
        return target

    def build_model(self, *, tensorboard_log: str | None = None) -> Any:
        from sb3_contrib import MaskablePPO

        env = self.env if self.env is not None else self.build_vec_env()
        self.model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=self.policy_kwargs,
            seed=self.seed,
            tensorboard_log=self._tensorboard_log(tensorboard_log),
            verbose=1,
            **self.hyperparams,
        )
        return self.model

    def warm_start_from_il(self, path: str | Path | None) -> TransferReport:
        """把 IL 訓練好的 encoder 權重載入 PPO policy。"""

        if self.model is None:
            raise RuntimeError("請先呼叫 build_model()")
        if not path:
            return self.transfer
        import torch

        target = resolve_path(path)
        if not target.exists():
            print(f"[PPOTrainer] 找不到 IL 權重：{target}（略過 warm start）")
            return self.transfer
        payload = torch.load(target, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        extractor = self.model.policy.features_extractor
        report = extractor.load_pretrained(state)
        self.transfer = TransferReport(loaded=report["loaded"], total=report["total"], source=str(target))
        print(f"[PPOTrainer] IL warm start：{report['loaded']}/{report['total']} 個張量已載入")
        ratio = report["loaded"] / max(1, report["total"])
        if ratio < 0.8:
            print(
                f"[PPOTrainer] ⚠️ 只匹配到 {ratio:.0%} 的張量，等於幾乎沒有熱啟動。\n"
                "              請確認 IL 與 PPO 的網路名稱一致："
                "configs/ppo.yaml 的 model.network 必須等於 IL 訓練時的 --network。"
            )
        else:
            config = payload.get("extra", {}).get("config", {}) if isinstance(payload, dict) else {}
            if isinstance(config, dict) and config.get("limit") is not None:
                print(
                    f"[PPOTrainer] ⚠️ 來源 IL 模型是 limit={config['limit']} 的小樣本模型"
                    f"（top1={payload.get('metrics', {}).get('top1')}），建議先完成全量 IL 訓練。"
                )
        return self.transfer

    # ------------------------------------------------------------------ 訓練
    def run(
        self,
        *,
        total_timesteps: int | None = None,
        resume_from: str | Path | None = None,
        il_weights: str | Path | None = None,
        use_wandb: bool = False,
        show_progress: bool | None = None,
    ) -> dict[str, Any]:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CheckpointCallback

        ensure_dir(self.checkpoint_dir)
        if resume_from and resolve_path(resume_from).exists():
            self.env = self.build_vec_env()
            self.model = MaskablePPO.load(
                resolve_path(resume_from),
                env=self.env,
                tensorboard_log=self._tensorboard_log(),
            )
            print(f"[PPOTrainer] 由 checkpoint 續訓：{resume_from}")
        else:
            self.build_model()
            self.warm_start_from_il(
                il_weights if il_weights is not None else self.config.get("train", {}).get("il_warm_start")
            )

        steps = int(total_timesteps or self.config.get("train", {}).get("total_timesteps", 200_000))
        save_freq = max(1, int(self.config.get("checkpoint", {}).get("save_freq", 50_000)) // max(1, self.n_envs))
        callbacks: list[Any] = [
            CheckpointCallback(save_freq=save_freq, save_path=str(self.checkpoint_dir), name_prefix="ppo"),
            BestModelCallback(self.checkpoint_dir / "best.zip"),
            CurriculumCallback(),
            TqdmProgressCallback(enable=show_progress),
        ]
        if use_wandb:
            wandb_callback = make_wandb_callback(
                project=str(self.config.get("logging", {}).get("wandb_project", "tetrio-ai")),
                mode=str(self.config.get("logging", {}).get("wandb_mode", "offline")),
            )
            if wandb_callback is not None:
                callbacks.append(wandb_callback)

        self.model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
        final_path = self.checkpoint_dir / "final.zip"
        self.model.save(final_path)
        return {
            "model_path": str(final_path),
            "total_timesteps": steps,
            "n_envs": self.n_envs,
            "transfer": self.transfer.to_dict(),
        }

    # ------------------------------------------------------------------ 評估
    def evaluate(
        self,
        *,
        episodes: int = 5,
        deterministic: bool = True,
        show_progress: bool | None = None,
    ) -> dict[str, float]:
        """用單一環境跑幾局並回傳平均 reward / 行數。"""

        import gymnasium as gym

        import envs  # noqa: F401 - 匯入即完成環境註冊
        from envs.progress import progress_bar

        env = gym.make(self.env_id, **self.env_kwargs)
        rewards: list[float] = []
        lines: list[int] = []
        with progress_bar(
            total=episodes,
            desc="評估",
            unit="episode",
            enable=show_progress,
            position=0,
        ) as bar:
            for episode in range(episodes):
                observation, info = env.reset(seed=self.seed + episode)
                done = False
                total_reward = 0.0
                while not done:
                    mask = env.unwrapped.action_masks()
                    action, _ = self.model.predict(observation, deterministic=deterministic, action_masks=mask)
                    observation, reward, terminated, truncated, info = env.step(int(action))
                    total_reward += float(reward)
                    done = bool(terminated or truncated)
                rewards.append(total_reward)
                lines.append(int(info.get("lines_total", 0)))
                bar.update(1)
                bar.set_postfix(rew=f"{np.mean(rewards):.1f}", lines=f"{np.mean(lines):.1f}")
        env.close()
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_lines": float(np.mean(lines)),
        }
