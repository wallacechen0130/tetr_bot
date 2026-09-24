"""最小可訓練驗證：資料集 → IL → PPO（需要 torch / sb3）。"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import envs  # noqa: E402,F401
from agents.heuristic_agent import HeuristicAgent  # noqa: E402
from datasets.reader import DatasetReader, TensorDataset, split_indices  # noqa: E402
from datasets.writer import DatasetWriter, build_sample  # noqa: E402
from policies.factory import build_network, count_parameters  # noqa: E402
from policies.sb3_extractor import TetrisFeaturesExtractor  # noqa: E402
from trainers.il_trainer import ILConfig, ILTrainer  # noqa: E402
from trainers.ppo_trainer import PPOTrainer  # noqa: E402


def _make_dataset(root) -> None:
    env = gym.make("TetrisSurvival-v0")
    agent = HeuristicAgent(depth=1, seed=3)
    writer = DatasetWriter(root, formats=("npz",), shard_size=16, config={"depth": 1})
    for episode in range(2):
        observation, info = env.reset(seed=3 + episode)
        for step in range(16):
            candidates = agent.rank(info["snapshot"], info["placements"])
            if not candidates:
                break
            writer.add(
                build_sample(
                    observation,
                    action=candidates[0].action,
                    candidates=candidates,
                    context=[episode, step, 0, 0],
                )
            )
            observation, _reward, terminated, truncated, info = env.step(candidates[0].action)
            if terminated or truncated:
                break
    writer.close()
    env.close()


def test_dataset_to_il_to_ppo(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    reader = DatasetReader(data_root)
    arrays = reader.arrays()
    assert arrays["action"].shape[0] >= 16
    train_idx, val_idx = split_indices(arrays["action"].shape[0], val_fraction=0.2, seed=0)
    assert len(train_idx) > 0 and len(val_idx) > 0
    _ = TensorDataset(arrays, train_idx)

    config = ILConfig(
        data_root=str(data_root),
        epochs=1,
        batch_size=8,
        checkpoint_dir=str(tmp_path / "il"),
    )
    trainer = ILTrainer(config)
    result = trainer.fit(epochs=1)
    assert result["best_top1"] >= 0.0
    # 回歸測試：loss 必須在正常範圍（修正前會是 ~1e7）
    assert result["history"][0]["train_loss"] < 30.0, result["history"][0]
    assert result["history"][0]["val_loss"] < 30.0, result["history"][0]
    assert (tmp_path / "il" / "best.pt").exists()

    network = build_network("resnet")
    assert count_parameters(network) > 0
    from envs.gym.obs_encoder import VECTOR_DIM

    extractor = TetrisFeaturesExtractor(
        gym.make("TetrisSurvival-v0").observation_space,
        features_dim=64,
        network="small_cnn",
    )
    assert extractor._infer_vector_dim(gym.make("TetrisSurvival-v0").observation_space) == VECTOR_DIM

    ppo_config = {
        "env": {"env_id": "TetrisSurvival-v0", "n_envs": 2, "vec": "dummy", "opponent_apm": 0.0},
        "model": {
            "network": "small_cnn",
            "net_arch": [64, 64],
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
        },
        "train": {"seed": 0, "il_warm_start": str(tmp_path / "il" / "best.pt")},
        "checkpoint": {"dir": str(tmp_path / "ppo"), "save_freq": 1000},
        "logging": {"tensorboard": str(tmp_path / "tb")},
    }
    ppo = PPOTrainer(ppo_config)
    ppo.policy_kwargs["features_extractor_kwargs"] = {"features_dim": 64, "network": "small_cnn"}
    out = ppo.run(total_timesteps=256)
    assert (tmp_path / "ppo" / "final.zip").exists()
    assert out["transfer"]["total"] > 0

    evaluation = ppo.evaluate(episodes=1)
    assert np.isfinite(evaluation["mean_reward"])
