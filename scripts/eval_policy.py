"""Roll out a trained policy greedily (argmax actions) and render it to a
GIF, for visually sanity-checking learned behavior against a config/checkpoint.
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.coverage_world import CoverageWorld
from models.policy import MultiRolePolicy
from ppo.rollout import build_edge_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg.get("run_name", os.path.splitext(os.path.basename(args.config))[0])
    checkpoint = args.checkpoint or os.path.join("runs", run_name, "model.pt")
    out = args.out or os.path.join("replays", f"{run_name}_trained_rollout.gif")

    env = CoverageWorld(**cfg.get("env", {}), render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    agent_order = env.possible_agents
    agent_to_idx = {a: i for i, a in enumerate(agent_order)}

    roles = sorted(set(env.roles.values()))
    obs_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).n
    model_cfg = cfg.get("model", {})

    policy = MultiRolePolicy(
        roles,
        obs_dim,
        action_dim,
        env.state_size,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        comm_aggregation=model_cfg.get("comm_aggregation", "none"),
        gat_heads=model_cfg.get("gat_heads", 4),
    )
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()

    frames = [env.render()]
    episode_reward = {a: 0.0 for a in env.agents}

    with torch.no_grad():
        while env.agents:
            edge_index = torch.as_tensor(build_edge_index(env, agent_to_idx))
            actions = policy.act_greedy(obs, env.roles, agent_order, edge_index, "cpu")
            obs, rewards, _, _, _ = env.step(actions)
            for a, r in rewards.items():
                episode_reward[a] += r
            frames.append(env.render())

    serviced = sum(1 for s in env.sites if s.serviced)
    expired = sum(1 for s in env.sites if s.expired)
    rounded_rewards = {a: round(r, 2) for a, r in episode_reward.items()}
    print(
        f"trained rollout: serviced={serviced}/{len(env.sites)} expired={expired} "
        f"reward_by_agent={rounded_rewards}"
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=10)
    print(f"saved rollout gif to {out}")


if __name__ == "__main__":
    main()
