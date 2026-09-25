"""PPO 評估路徑與 episode 統計的測試。

對應兩個實測 bug：

1. VecEnv 少了 ``VecMonitor`` → 不會產生 ``info["episode"]``
   → ``BestModelCallback`` 永遠存不出 ``best.zip``，TensorBoard 也沒有 reward 曲線。
2. PPO 的 ``.zip`` 不能餵給 ``PolicyAgent``（IL 的 ``.pt`` 載入器）。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from agents.policy_agent import PolicyAgent, PPOAgent  # noqa: E402
from envs.config import load_yaml  # noqa: E402
from trainers.ppo_trainer import PPOTrainer  # noqa: E402


def _tiny_config(tmp_path) -> dict:
    config = load_yaml("configs/ppo.yaml")
    config["env"].update({"n_envs": 2, "vec": "dummy"})
    config["model"].update(
        {
            "network": "small_cnn",
            "net_arch": [32, 32],
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
        }
    )
    config["checkpoint"].update({"dir": str(tmp_path / "ppo"), "save_freq": 1000})
    config["logging"]["tensorboard"] = str(tmp_path / "tb")
    config["train"]["il_warm_start"] = None
    return config


def test_vec_env_reports_episode_statistics(tmp_path):
    """有 VecMonitor 才會有 info["episode"]（best.zip 與 reward 曲線的前提）。"""

    trainer = PPOTrainer(_tiny_config(tmp_path), seed=0)
    env = trainer.build_vec_env()
    assert type(env).__name__ == "VecMonitor"

    env.reset()
    saw_episode = False
    for _ in range(400):
        actions = np.array([0 for _ in range(env.num_envs)])
        _obs, _rewards, dones, infos = env.step(actions)
        if any("episode" in info for info in infos):
            saw_episode = True
            break
        if dones.any():
            break
    env.close()
    assert saw_episode, "VecMonitor 應該要在 episode 結束時提供 info['episode']"


def test_ppo_agent_loads_and_acts(tmp_path):
    config = _tiny_config(tmp_path)
    trainer = PPOTrainer(config, seed=0)
    trainer.run(total_timesteps=64)
    model_path = tmp_path / "ppo" / "final.zip"
    assert model_path.exists()

    agent = PPOAgent(str(model_path), seed=0)
    observation, info = __import__("gymnasium").make("TetrisSurvival-v0").reset(seed=0)
    action = agent.act(observation, info)
    assert action in info["placements"]


def test_policy_agent_rejects_sb3_zip(tmp_path):
    config = _tiny_config(tmp_path)
    PPOTrainer(config, seed=0).run(total_timesteps=64)
    model_path = tmp_path / "ppo" / "final.zip"

    with pytest.raises(ValueError) as excinfo:
        PolicyAgent(str(model_path))
    assert "ppo" in str(excinfo.value).lower()


def test_build_agent_auto_detects_zip(tmp_path):
    import gymnasium as gym

    import envs  # noqa: F401
    from scripts.evaluate import build_agent

    config = _tiny_config(tmp_path)
    PPOTrainer(config, seed=0).run(total_timesteps=64)
    agent = build_agent("policy", depth=1, model_path=str(tmp_path / "ppo" / "final.zip"), seed=0)
    assert isinstance(agent, PPOAgent)

    obs, info = gym.make("Tetris40L-v0").reset(seed=1)
    assert agent.act(obs, info) in info["placements"]


def test_sync_drive_skips_windows_junk():
    from pathlib import Path

    from scripts.sync_drive import is_excluded

    assert is_excluded(Path("datasets/heuristic-v1/desktop.ini")) is True
    assert is_excluded(Path("checkpoints/Thumbs.db")) is True
    assert is_excluded(Path("datasets/__pycache__/reader.cpython-313.pyc")) is True
    assert is_excluded(Path("datasets/heuristic-v1/shards/worker00_00000.npz")) is False


def test_sync_drive_does_not_copy_source_code():
    """原始碼不該被同步，否則 from_drive 會用舊程式碼蓋掉本機新版本。"""

    from pathlib import Path

    from scripts.sync_drive import is_excluded

    assert is_excluded(Path("datasets/reader.py")) is True
    assert is_excluded(Path("datasets/README.md")) is True
    assert is_excluded(Path("checkpoints/il/best.pt")) is False
    assert is_excluded(Path("datasets/heuristic-v1/data.parquet")) is False


def test_sync_drive_keeps_newer_file(tmp_path):
    """大小不同時，較新的檔案不該被舊檔覆蓋。"""

    import os
    import time

    from scripts.sync_drive import should_copy

    src = tmp_path / "src.npz"
    dst = tmp_path / "dst.npz"
    src.write_bytes(b"aaa")
    dst.write_bytes(b"bbbbbb")
    now = time.time()
    os.utime(src, (now - 100, now - 100))
    os.utime(dst, (now, now))
    assert should_copy(src, dst) == (False, "target-newer")
    assert should_copy(src, dst, overwrite=True) == (True, "overwrite")

    os.utime(src, (now, now))
    assert should_copy(src, dst)[0] is True
    assert should_copy(src, dst.parent / "missing.npz") == (True, "new")
