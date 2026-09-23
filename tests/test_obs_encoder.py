"""Observation 編碼器測試。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from envs.gym.obs_encoder import VECTOR_DIM, VECTOR_KEYS, ObservationEncoder, vector_from_observation


def test_encoder_shapes_and_flatten_dim() -> None:
    encoder = ObservationEncoder()
    space = encoder.observation_space()
    assert space["board"].shape == (2, 20, 10)
    assert encoder.feature_dim == 480
    assert encoder.flat_dim == 480 + 80
    assert VECTOR_DIM == 160


def test_observations_stay_inside_space() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=29)
    space = env.observation_space
    assert space.contains(observation)
    for action in list(info["placements"])[:25]:
        observation, _reward, terminated, truncated, info = env.step(action)
        assert space.contains(observation)
        if terminated or truncated:
            break
    env.close()


def test_vector_from_observation_matches_encoder() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, _info = env.reset(seed=31)
    vector = vector_from_observation(observation)
    assert vector.shape == (VECTOR_DIM,)
    assert vector.dtype == np.float32
    parts = [np.asarray(observation[key]).reshape(-1) for key in VECTOR_KEYS]
    assert vector.shape[0] == sum(part.size for part in parts)
    env.close()


def test_encoding_is_deterministic() -> None:
    env = gym.make("TetrisSurvival-v0")
    observation, info = env.reset(seed=37)
    encoder = env.unwrapped.encoder
    snapshot = info["snapshot"]
    mask = observation["action_mask"]
    first = encoder.encode(snapshot, mask)
    second = encoder.encode(snapshot.copy(), mask)
    for key in first:
        assert np.array_equal(first[key], second[key])
    env.close()
