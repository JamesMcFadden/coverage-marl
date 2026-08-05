"""Milestone 1 sanity check: run CoverageWorld with a random policy, render
frames to a GIF, and print per-episode reward/outcome stats so the reward
signal can be eyeballed for sanity before any learning is introduced.
"""
import argparse
import os
import sys

import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.coverage_world import CoverageWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="replays/milestone1_random_rollout.gif")
    args = parser.parse_args()

    env = CoverageWorld(max_steps=args.max_steps, render_mode="rgb_array")

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        frames = [env.render()]
        episode_reward = {a: 0.0 for a in env.agents}
        step = 0

        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            for a, r in rewards.items():
                episode_reward[a] += r
            frames.append(env.render())
            step += 1

        serviced = sum(1 for s in env.sites if s.serviced)
        expired = sum(1 for s in env.sites if s.expired)
        rounded_rewards = {a: round(r, 2) for a, r in episode_reward.items()}
        print(
            f"episode {ep}: steps={step} serviced={serviced}/{len(env.sites)} "
            f"expired={expired} reward_by_agent={rounded_rewards}"
        )

        out_path = args.out if args.episodes == 1 else args.out.replace(".gif", f"_ep{ep}.gif")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imageio.mimsave(out_path, frames, fps=10)
        print(f"saved rollout gif to {out_path}")


if __name__ == "__main__":
    main()
