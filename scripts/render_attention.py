"""Render a trained GAT-communication policy's rollout with attention
weights overlaid on the comms graph (milestone 5 deliverable): an arrow
from agent A to agent B, with width proportional to how much B's policy
attends to A, shows who's actually being listened to at each step.
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
from ppo.rollout import build_edge_index, build_full_edge_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    if model_cfg.get("comm_aggregation") != "gat":
        raise ValueError("render_attention.py requires a config with model.comm_aggregation: gat")

    run_name = cfg.get("run_name", os.path.splitext(os.path.basename(args.config))[0])
    checkpoint = args.checkpoint or os.path.join("runs", run_name, "model.pt")
    out = args.out or os.path.join("replays", f"{run_name}_attention.gif")

    env = CoverageWorld(**cfg.get("env", {}), render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    agent_order = env.possible_agents
    agent_to_idx = {a: i for i, a in enumerate(agent_order)}

    roles = sorted(set(env.roles.values()))
    policy = MultiRolePolicy(
        roles,
        env.observation_space(agent_order[0]).shape[0],
        env.action_space(agent_order[0]).n,
        env.state_size,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        comm_aggregation="gat",
        gat_heads=model_cfg.get("gat_heads", 4),
    )
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()

    graph_connectivity = model_cfg.get("graph_connectivity", "range")
    full_edge_index = build_full_edge_index(len(agent_order)) if graph_connectivity == "full" else None

    frames = [env.render()]

    with torch.no_grad():
        while env.agents:
            edge_index_np = full_edge_index if graph_connectivity == "full" else build_edge_index(env, agent_to_idx)
            edge_index = torch.as_tensor(edge_index_np)
            actions, attn = policy.act_greedy(obs, env.roles, agent_order, edge_index, "cpu", return_attention=True)

            attention_edges = []
            if attn is not None:
                edge_index_out, alpha = attn
                weights = alpha.mean(dim=-1)  # average over attention heads -> (E,)
                for e in range(edge_index_out.shape[1]):
                    src_i, dst_i = int(edge_index_out[0, e]), int(edge_index_out[1, e])
                    attention_edges.append((agent_order[src_i], agent_order[dst_i], float(weights[e])))

            obs, _, _, _, _ = env.step(actions)
            frames.append(env.render(attention_edges=attention_edges))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=10)
    print(f"saved attention rollout gif to {out}")


if __name__ == "__main__":
    main()
