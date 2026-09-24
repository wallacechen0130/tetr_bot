"""Phase 3：用啟發式教師平行產生 State → Best Move 資料集。"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Any

from agents.heuristic_agent import HeuristicAgent
from datasets.manifest import DatasetManifest, write_manifest
from datasets.writer import DatasetWriter, build_sample, write_parquet_from_shards
from envs.config import resolve_path
from envs.progress import progress_bar


def _play_episode(env: Any, teacher: HeuristicAgent, writer: DatasetWriter, *, seed: int, episode: int, max_pieces: int) -> int:
    observation, info = env.reset(seed=seed)
    samples = 0
    for step in range(max_pieces):
        candidates = teacher.rank(info["snapshot"], info["placements"])
        if not candidates:
            break
        best = candidates[0]
        sample = build_sample(
            observation,
            action=best.action,
            candidates=candidates,
            context=[episode, step, int(info.get("lines_total", 0)), int(info["snapshot"].combo)],
        )
        writer.add(sample)
        samples += 1
        observation, _reward, terminated, truncated, info = env.step(best.action)
        if terminated or truncated:
            break
    return samples


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    """單一 worker：產生數個 episode 並寫成獨立 shard。"""

    import gymnasium as gym

    import envs  # noqa: F401 - 匯入即完成環境註冊

    worker_id = int(payload["worker_id"])
    out_dir = Path(payload["out"])
    teacher = HeuristicAgent(depth=int(payload["depth"]), seed=int(payload["seed"]) + worker_id)
    env = gym.make(str(payload["env_id"]), seed=int(payload["seed"]) + worker_id)
    env.unwrapped.set_piece_time(float(payload["piece_time"]))
    writer = DatasetWriter(
        out_dir,
        name=str(payload["name"]),
        formats=("npz",),
        shard_size=int(payload["shard_size"]),
        shard_prefix=f"worker{worker_id:02d}",
        write_manifest_file=False,
        config=dict(payload["config"]),
        generator="heuristic",
    )
    samples = 0
    progress_queue = payload.get("progress_queue")
    for episode in range(int(payload["episodes"])):
        episode_samples = _play_episode(
            env,
            teacher,
            writer,
            seed=int(payload["seed"]) + worker_id * 1000 + episode,
            episode=episode,
            max_pieces=int(payload["max_pieces"]),
        )
        samples += episode_samples
        if progress_queue is not None:
            # 每完成一局回報一次，讓主行程能顯示進度與 ETA
            progress_queue.put((1, episode_samples))
    manifest = writer.close()
    env.close()
    return {"worker": worker_id, "shards": manifest.shards, "samples": samples}


def generate(
    *,
    out: str = "datasets/heuristic-v1",
    episodes: int = 20,
    workers: int = 4,
    seed: int = 12345,
    depth: int = 1,
    env_id: str = "TetrisSurvival-v0",
    max_pieces: int = 200,
    shard_size: int = 5000,
    piece_time: float = 0.5,
    formats: tuple[str, ...] = ("npz", "parquet"),
    name: str = "heuristic-v1",
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """平行產生資料集並寫出 manifest（可選 parquet）。"""

    out_dir = resolve_path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_worker = max(1, episodes // max(1, workers))
    payloads = [
        {
            "worker_id": worker_id,
            "out": str(out_dir),
            "name": name,
            "episodes": per_worker,
            "seed": seed,
            "depth": depth,
            "env_id": env_id,
            "max_pieces": max_pieces,
            "shard_size": shard_size,
            "piece_time": piece_time,
            "config": {"env_id": env_id, "depth": depth, "max_pieces": max_pieces, "seed": seed},
        }
        for worker_id in range(max(1, workers))
    ]
    started = time.time()
    context = mp.get_context("spawn")
    total_episodes = per_worker * max(1, workers)
    results: list[dict[str, Any]] = []
    with context.Manager() as manager:
        progress_queue = manager.Queue()
        for payload in payloads:
            payload["progress_queue"] = progress_queue
        with context.Pool(processes=max(1, workers)) as pool:
            async_result = pool.map_async(_worker, payloads)
            episodes_done = 0
            samples_done = 0
            with progress_bar(
                total=total_episodes,
                desc="收集資料",
                unit="episode",
                enable=show_progress,
                postfix={"samples": 0},
            ) as bar:
                while not async_result.ready():
                    try:
                        episode_delta, sample_delta = progress_queue.get(timeout=0.2)
                    except QueueEmpty:
                        continue
                    episodes_done += int(episode_delta)
                    samples_done += int(sample_delta)
                    bar.update(int(episode_delta))
                    elapsed = max(time.time() - started, 1e-6)
                    bar.set_postfix(
                        samples=samples_done,
                        rate=f"{samples_done / elapsed:.0f} sample/s",
                    )
                while True:  # 清空最後幾筆回報
                    try:
                        episode_delta, sample_delta = progress_queue.get_nowait()
                    except QueueEmpty:
                        break
                    episodes_done += int(episode_delta)
                    samples_done += int(sample_delta)
                    bar.update(int(episode_delta))
            results = async_result.get()

    shards: list[str] = []
    total = 0
    for result in results:
        shards.extend(result["shards"])
        total += int(result["samples"])
    shards.sort()

    files: list[str] = []
    if "parquet" in formats and shards:
        write_parquet_from_shards(out_dir, [out_dir / shard for shard in shards])
        files.append("data.parquet")

    manifest = DatasetManifest(
        name=name,
        num_samples=total,
        num_shards=len(shards),
        shards=shards,
        files=files,
        format=[fmt for fmt in formats if fmt == "npz" or files],
        config={"env_id": env_id, "depth": depth, "max_pieces": max_pieces, "episodes": episodes, "seed": seed},
        generator="heuristic",
    )
    write_manifest(out_dir / "manifest.json", manifest)
    elapsed = time.time() - started
    return {
        "root": str(out_dir),
        "num_samples": total,
        "num_shards": len(shards),
        "workers": workers,
        "elapsed_s": elapsed,
        "samples_per_second": total / max(elapsed, 1e-6),
        "manifest": str(out_dir / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="產生啟發式教師資料集（State → Best Move）")
    parser.add_argument("--out", default="datasets/heuristic-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--depth", type=int, default=1, help="教師搜尋深度（1=單層）")
    parser.add_argument("--env-id", default="TetrisSurvival-v0")
    parser.add_argument("--max-pieces", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--formats", default="npz,parquet")
    parser.add_argument("--no-progress", action="store_true", help="關閉進度條")
    args = parser.parse_args()

    summary = generate(
        out=args.out,
        episodes=args.episodes,
        workers=args.workers,
        seed=args.seed,
        depth=args.depth,
        env_id=args.env_id,
        max_pieces=args.max_pieces,
        shard_size=args.shard_size,
        formats=tuple(part.strip() for part in args.formats.split(",") if part.strip()),
        show_progress=False if args.no_progress else None,
    )
    print("[generate_dataset] 完成：")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
