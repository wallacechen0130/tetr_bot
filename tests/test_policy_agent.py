"""PolicyAgent（IL 模型）的實際執行測試。

對應實測 bug：PolicyAgent 以前把棋盤也塞進 vector（560 維），
但網路要的是 160 維 → 一執行 forward 就維度不符。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import envs  # noqa: E402,F401
from agents.policy_agent import PolicyAgent  # noqa: E402
from envs.gym.obs_encoder import VECTOR_DIM, vector_from_observation  # noqa: E402
from policies.factory import build_network  # noqa: E402
from trainers.common import save_checkpoint  # noqa: E402


def _save_il_checkpoint(path, network: str = "small_cnn") -> None:
    model = build_network(network)
    save_checkpoint(
        path,
        model=model,
        epoch=1,
        metrics={"top1": 0.0},
        extra={"network": network, "config": {"network": network, "limit": None}},
    )


def test_policy_agent_runs_an_episode(tmp_path):
    checkpoint = tmp_path / "il" / "best.pt"
    _save_il_checkpoint(checkpoint)

    env = gym.make("Tetris40L-v0")
    agent = PolicyAgent(str(checkpoint), seed=1)
    observation, info = env.reset(seed=1)

    for _ in range(20):
        action = agent.act(observation, info)
        assert action in info["placements"]
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


def test_policy_agent_infers_network_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "il" / "best.pt"
    _save_il_checkpoint(checkpoint, network="small_cnn")
    agent = PolicyAgent(str(checkpoint))
    assert type(agent.network.board_encoder).__name__ == "SmallCNN"


def test_policy_agent_vector_dimension_matches_network(tmp_path):
    checkpoint = tmp_path / "il" / "best.pt"
    _save_il_checkpoint(checkpoint)
    agent = PolicyAgent(str(checkpoint))
    env = gym.make("TetrisSurvival-v0")
    observation, _info = env.reset(seed=0)

    vector = vector_from_observation(observation)
    assert vector.shape == (VECTOR_DIM,)

    probs = agent.action_probs(observation)
    assert probs.shape == (80,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    # 不合法動作的機率必須是 0
    mask = np.asarray(observation["action_mask"]).reshape(-1)
    assert np.all(probs[mask == 0] < 1e-6)
    env.close()
