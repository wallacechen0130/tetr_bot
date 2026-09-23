"""對局基準測試：跑 N 局並收集 EpisodeMetrics。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evaluators.metrics import EpisodeMetrics


class BenchmarkRunner:
    """以固定 agent（可選搭 DifficultyController）跑基準測試。"""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent: Any,
        *,
        controller: Any = None,
        seed: int = 0,
        max_steps: int = 2000,
    ) -> None:
        self.env_factory = env_factory
        self.agent = agent
        self.controller = controller
        self.seed = int(seed)
        self.max_steps = int(max_steps)

    def run_episode(self, *, seed: int) -> EpisodeMetrics:
        env = self.env_factory()
        observation, info = env.reset(seed=seed)
        controller = self.controller
        clock = getattr(controller, "clock", None) if controller is not None else None
        if controller is not None:
            controller.reset()
        elif hasattr(self.agent, "reset"):
            self.agent.reset(seed=seed)

        total_reward = 0.0
        max_combo = 0
        b2b_clears = 0
        tspins = 0
        perfect_clears = 0
        holds = 0
        invalid = 0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated) and steps < self.max_steps:
            if controller is not None:
                decision = controller.decide_from_observation(observation, info)
                if clock is not None:
                    clock.advance(decision.delay_s)
                env.unwrapped.set_piece_time(decision.delay_s)
                action = decision.action
            else:
                action = self.agent.act(observation, info)

            observation, reward, terminated, truncated, info = env.step(action)
            events = info.get("events", {})
            total_reward += float(reward)
            max_combo = max(max_combo, int(events.get("combo", 0)))
            b2b_clears += 1 if events.get("b2b_active") else 0
            tspins += 1 if events.get("tspin") else 0
            perfect_clears += 1 if events.get("perfect_clear") else 0
            holds += 1 if events.get("used_hold") else 0
            invalid += 1 if info.get("invalid_action") else 0
            if controller is not None:
                controller.note_action(
                    attack=int(events.get("attack_sent", 0)),
                    lines=int(events.get("lines_cleared", 0)),
                )
            steps += 1

        snapshot = info.get("snapshot")
        holes = 0
        height = 0
        if snapshot is not None:
            from envs.engine.board import holes_from_grid, max_height_from_grid

            holes = int(holes_from_grid(snapshot.board))
            height = int(max_height_from_grid(snapshot.board))

        metrics = EpisodeMetrics(
            tier=getattr(controller, "tier", "N/A") if controller is not None else "N/A",
            mode=str(info.get("mode", "survival")),
            agent=getattr(self.agent, "name", self.agent.__class__.__name__),
            reward=total_reward,
            pieces=int(info.get("pieces", 0)),
            lines=int(info.get("lines_total", 0)),
            elapsed=float(info.get("elapsed", 0.0)),
            pps=float(info.get("pps", 0.0)),
            apm=float(info.get("apm", 0.0)),
            attack_sent=int(info.get("attack_sent", 0)),
            garbage_received=int(info.get("garbage_received", 0)),
            max_combo=max_combo,
            b2b_clears=b2b_clears,
            tspins=tspins,
            perfect_clears=perfect_clears,
            holds=holds,
            invalid_actions=invalid,
            holes_final=holes,
            max_height_final=height,
            top_out=bool(info.get("top_out", False)),
            completed=bool(info.get("lines_remaining") == 0),
            lines_remaining=info.get("lines_remaining"),
        )
        env.close()
        return metrics

    def run(self, episodes: int = 10, *, seeds: list[int] | None = None) -> list[EpisodeMetrics]:
        seeds = seeds or [self.seed + index for index in range(episodes)]
        return [self.run_episode(seed=seed) for seed in seeds[:episodes]]
