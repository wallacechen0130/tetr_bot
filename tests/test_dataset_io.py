"""資料集寫入 / 讀取 / manifest / parquet 測試。"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from agents.heuristic_agent import HeuristicAgent
from datasets.manifest import DatasetManifest, config_hash
from datasets.reader import DatasetReader, split_indices
from datasets.writer import DatasetWriter, build_sample


def _collect_samples(count: int = 12) -> list[dict[str, np.ndarray]]:
    env = gym.make("TetrisSurvival-v0")
    agent = HeuristicAgent(depth=1, seed=7)
    observation, info = env.reset(seed=7)
    samples = []
    for step in range(count):
        candidates = agent.rank(info["snapshot"], info["placements"])
        if not candidates:
            break
        samples.append(
            build_sample(
                observation,
                action=candidates[0].action,
                candidates=candidates,
                context=[0, step, int(info["lines_total"]), 0],
            )
        )
        observation, _reward, terminated, truncated, info = env.step(candidates[0].action)
        if terminated or truncated:
            break
    env.close()
    return samples


def test_write_and_read_dataset(tmp_path) -> None:
    samples = _collect_samples()
    assert len(samples) >= 5
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, name="unit-test", formats=("npz", "parquet"), shard_size=4, config={"depth": 1})
    for sample in samples:
        writer.add(sample)
    manifest = writer.close()

    assert manifest.num_samples == len(samples)
    assert manifest.num_shards == -(-len(samples) // 4)
    assert (root / "manifest.json").exists()
    assert (root / "data.parquet").exists()

    reader = DatasetReader(root)
    arrays = reader.arrays()
    assert arrays["board"].shape == (len(samples), 2, 20, 10)
    assert arrays["vector"].shape[1] == 160
    assert arrays["mask"].shape[1] == 80
    assert arrays["action"].shape == (len(samples),)
    assert arrays["topk_actions"].shape[1] == 5

    reloaded = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert reloaded["num_samples"] == len(samples)
    assert reloaded["config_hash"] == config_hash({"depth": 1})


def test_parquet_matches_npz(tmp_path) -> None:
    samples = _collect_samples(8)
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, formats=("npz", "parquet"), shard_size=3)
    writer.add_many(samples)
    writer.close()

    reader = DatasetReader(root)
    npz_arrays = reader.arrays()
    parquet_arrays = reader.parquet_arrays()
    assert np.array_equal(npz_arrays["action"], parquet_arrays["action"])
    assert np.array_equal(npz_arrays["mask"], parquet_arrays["mask"])


def test_tensor_dataset_and_split() -> None:
    torch = __import__("pytest").importorskip("torch")
    from datasets.reader import TensorDataset

    samples = _collect_samples(10)
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    dataset = TensorDataset(arrays)
    assert len(dataset) == len(samples)
    item = dataset[0]
    assert item["board"].shape == (2, 20, 10)
    assert item["vector"].shape == (160,)
    assert item["action"].dtype == torch.long

    train, val = split_indices(len(samples), val_fraction=0.2, seed=0)
    assert len(train) + len(val) == len(samples)
    assert len(val) == 2
    assert not set(train) & set(val)


def test_manifest_roundtrip(tmp_path) -> None:
    from datasets.manifest import load_manifest, write_manifest

    manifest = DatasetManifest(name="x", num_samples=3, shards=["shards/a.npz"], config={"seed": 1})
    write_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_manifest(tmp_path)
    assert loaded.name == "x"
    assert loaded.num_samples == 3
    assert loaded.config_hash == manifest.config_hash
