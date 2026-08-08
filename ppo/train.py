import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from env.coverage_world import CoverageWorld
from models.policy import MultiRolePolicy
from ppo.ppo import compute_gae, ppo_update
from ppo.rollout import collect_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    run_name = cfg.get("run_name", Path(args.config).stem)
    seed = train_cfg.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CoverageWorld(**cfg.get("env", {}))
    obs, _ = env.reset(seed=seed)

    roles = sorted(set(env.roles.values()))
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).n

    model_cfg = cfg.get("model", {})
    policy = MultiRolePolicy(
        roles=roles,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=env.state_size,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        comm_aggregation=model_cfg.get("comm_aggregation", "none"),
        gat_heads=model_cfg.get("gat_heads", 4),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.get("lr", 3e-4))
    graph_connectivity = model_cfg.get("graph_connectivity", "range")

    rollout_length = train_cfg["rollout_length"]
    total_timesteps = train_cfg["total_timesteps"]
    num_updates = max(1, total_timesteps // rollout_length)

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        buffer, episode_summaries, obs = collect_rollout(
            env, policy, rollout_length, device, obs=obs, graph_connectivity=graph_connectivity
        )
        global_step += rollout_length

        advantages, returns = {}, {}
        for agent in buffer.agents:
            advantages[agent], returns[agent] = compute_gae(
                buffer.rewards[agent],
                buffer.values,
                buffer.dones,
                buffer.bootstrap_values,
                gamma=train_cfg.get("gamma", 0.99),
                lam=train_cfg.get("gae_lambda", 0.95),
            )

        stats = ppo_update(
            policy,
            optimizer,
            buffer,
            advantages,
            returns,
            env.roles,
            clip_coef=train_cfg.get("clip_coef", 0.2),
            value_coef=train_cfg.get("value_coef", 0.5),
            entropy_coef=train_cfg.get("entropy_coef", 0.01),
            epochs=train_cfg.get("epochs", 4),
            minibatch_size=train_cfg.get("minibatch_size", 256),
            device=device,
        )

        writer.add_scalar("loss/policy", stats["policy_loss"], global_step)
        writer.add_scalar("loss/value", stats["value_loss"], global_step)
        writer.add_scalar("loss/entropy", stats["entropy"], global_step)

        if episode_summaries:
            mean_return = float(np.mean([e["return"] for e in episode_summaries]))
            mean_coverage = float(np.mean([e["serviced"] / e["num_sites"] for e in episode_summaries]))
            mean_len = float(np.mean([e["length"] for e in episode_summaries]))
            writer.add_scalar("episode/return", mean_return, global_step)
            writer.add_scalar("episode/coverage", mean_coverage, global_step)
            writer.add_scalar("episode/length", mean_len, global_step)

            print(
                f"update {update}/{num_updates} step {global_step} "
                f"episodes={len(episode_summaries)} return={mean_return:.2f} "
                f"coverage={mean_coverage:.2f} len={mean_len:.0f} "
                f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f}"
            )
        else:
            print(
                f"update {update}/{num_updates} step {global_step} "
                f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f}"
            )

    torch.save(policy.state_dict(), run_dir / "model.pt")
    writer.close()
    print(f"done in {time.time() - start_time:.1f}s, saved model to {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
