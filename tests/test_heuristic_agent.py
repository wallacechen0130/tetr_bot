"""啟發式教師：搜尋深度、評分排序與 40L 實戰。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from agents.heuristic_agent import FEATURE_ORDER, HeuristicAgent


def test_weights_loaded_from_config() -> None:
    agent = HeuristicAgent()
    assert agent.weights["landing_height"] < 0
    assert agent.weights["holes"] < 0
    assert agent.weights["eroded_piece_cells"] > 0
    assert set(agent.weights) == set(FEATURE_ORDER)


def test_rank_returns_sorted_candidates() -> None:
    env = gym.make("TetrisSurvival-v0")
    _observation, info = env.reset(seed=101)
    agent = HeuristicAgent(depth=1)
    candidates = agent.rank(info["snapshot"], info["placements"])
    assert candidates
    scores = [candidate.score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)
    assert candidates[0].action in info["placements"]
    env.close()


def test_act_returns_legal_action() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=103)
    agent = HeuristicAgent()
    for _ in range(10):
        action = agent.act(observation, info)
        assert action in info["placements"]
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


def test_depth_two_search_runs() -> None:
    env = gym.make("TetrisSurvival-v0")
    _observation, info = env.reset(seed=107)
    shallow = HeuristicAgent(depth=1)
    deep = HeuristicAgent(depth=2, beam_width=3)
    shallow_rank = shallow.rank(info["snapshot"], info["placements"])
    deep_rank = deep.rank(info["snapshot"], info["placements"])
    assert len(deep_rank) == len(shallow_rank)
    assert all(np.isfinite(candidate.score) for candidate in deep_rank[:5])
    env.close()


def test_heuristic_completes_40l() -> None:
    env = gym.make("Tetris40L-v0", seed=202)
    env.unwrapped.set_piece_time(0.5)
    agent = HeuristicAgent(depth=1, seed=202)
    observation, info = env.reset(seed=202)
    done = False
    steps = 0
    while not done and steps < 500:
        action = agent.act(observation, info)
        observation, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    assert info["lines_remaining"] == 0, "啟發式教師應該要能完成 40L"
    assert info["pieces"] <= 200
    assert 10.0 <= info["elapsed"] <= 120.0
    env.close()


def test_heuristic_survives_survival_mode() -> None:
    env = gym.make("TetrisSurvival-v0", seed=303)
    env.unwrapped.game_config.time_limit = 20.0
    env.unwrapped.set_piece_time(0.5)
    agent = HeuristicAgent(depth=1, seed=303)
    observation, info = env.reset(seed=303)
    done = False
    steps = 0
    while not done and steps < 200:
        action = agent.act(observation, info)
        observation, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    assert not info["top_out"], "啟發式教師在 20 秒內不應該 top out"
    env.close()
