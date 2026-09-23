"""在終端機用 ASCII 觀看 AI 遊玩（除錯與教學用）。"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym

import envs  # noqa: F401 - 匯入即完成環境註冊
from agents.heuristic_agent import HeuristicAgent
from agents.policy_agent import PolicyAgent
from agents.random_agent import RandomAgent
from controllers.difficulty import DifficultyController
from controllers.timing import FakeClock
from envs.render.ascii import render_board_ascii


def main() -> None:
    parser = argparse.ArgumentParser(description="ASCII 觀戰")
    parser.add_argument("--agent", default="heuristic", choices=["heuristic", "random", "policy"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-id", default="TetrisSurvival-v0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--delay", type=float, default=0.0, help="每顆方塊之間的延遲（秒）")
    parser.add_argument("--every", type=int, default=1, help="每 N 顆方塊才重畫一次")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    env = gym.make(args.env_id)
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    elif args.agent == "policy":
        agent = PolicyAgent(args.model, seed=args.seed)
    else:
        agent = HeuristicAgent(seed=args.seed)
    controller = DifficultyController(args.tier, clock=FakeClock(), seed=args.seed) if args.tier else None

    for episode in range(args.episodes):
        observation, info = env.reset(seed=args.seed + episode)
        done = False
        step = 0
        while not done and step < args.max_steps:
            if controller is not None:
                decision = controller.decide_from_observation(observation, info)
                controller.clock.advance(decision.delay_s)
                env.unwrapped.set_piece_time(decision.delay_s)
                action = decision.action
            else:
                action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            if step % max(1, args.every) == 0:
                print(f"\n=== 第 {episode + 1} 局 / 第 {step} 顆 | 行數 {info['lines_total']} | reward {reward:+.2f} ===")
                print(render_board_ascii(env.unwrapped.sim.board))
            if args.delay > 0:
                time.sleep(args.delay)
            done = bool(terminated or truncated)
        print(
            f"[第 {episode + 1} 局] 行數={info['lines_total']} 方塊={info['pieces']} "
            f"耗時={info['elapsed']:.1f}s PPS={info['pps']:.2f} APM={info['apm']:.1f} top_out={info['top_out']}"
        )
    env.close()


if __name__ == "__main__":
    main()
