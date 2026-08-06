import numpy as np
import torch

from ppo.rollout import batch_edge_indices


def compute_gae(rewards, values, dones, bootstrap_values, gamma, lam):
    """Per-agent GAE. `dones` marks true termination (bootstrap value is
    zeroed); a truncation-only boundary still bootstraps from the real
    next-state value but must not let the advantage carry across the
    boundary, since the next buffer index belongs to a different episode.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if dones[t] else bootstrap_values[t]
        carry_mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * carry_mask * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy,
    optimizer,
    buffer,
    advantages,
    returns,
    roles_by_agent,
    clip_coef=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
    minibatch_size=256,
    device="cpu",
):
    """Minibatches are over *timesteps*, not flattened (agent, t) samples,
    because evaluating an agent's log-prob under the current policy needs
    that timestep's full communication graph (all agents' embeddings and
    edges), not just that one agent's observation. Each minibatch's set of
    per-timestep graphs is combined into one batched graph (node indices
    offset per-timestep) so `evaluate_batch` runs once per minibatch rather
    than once per timestep. `minibatch_size` means timesteps per minibatch;
    each contributes len(agent_order) samples.
    """
    agent_order = buffer.agents
    T = buffer.rollout_length
    n_agents = len(agent_order)

    role_groups = {}
    for agent in agent_order:
        role_groups.setdefault(roles_by_agent[agent], []).append(agent)

    normalized_advantages = {}
    for role, agents_in_role in role_groups.items():
        pooled = np.concatenate([advantages[a] for a in agents_in_role])
        mean, std = pooled.mean(), pooled.std()
        for a in agents_in_role:
            normalized_advantages[a] = (advantages[a] - mean) / (std + 1e-8)

    mean_returns = np.mean([returns[a] for a in agent_order], axis=0)

    actor_loss_sum, entropy_sum, actor_updates = 0.0, 0.0, 0
    value_loss_sum, critic_updates = 0.0, 0

    for _ in range(epochs):
        idx = np.random.permutation(T)
        for start in range(0, T, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]

            obs_b = {a: buffer.obs[a][mb_idx] for a in agent_order}
            actions_b = {a: buffer.actions[a][mb_idx] for a in agent_order}
            edge_index_b = torch.as_tensor(
                batch_edge_indices(buffer.edge_indices, mb_idx, n_agents), device=device
            )

            logprobs_new, entropies = policy.evaluate_batch(
                obs_b, roles_by_agent, agent_order, edge_index_b, actions_b, device
            )

            policy_loss_terms, entropy_terms = [], []
            for a in agent_order:
                old_logprob = torch.as_tensor(buffer.logprobs[a][mb_idx], device=device)
                adv = torch.as_tensor(normalized_advantages[a][mb_idx], device=device)
                ratio = torch.exp(logprobs_new[a] - old_logprob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
                policy_loss_terms.append(-torch.min(surr1, surr2))
                entropy_terms.append(entropies[a])

            policy_loss = torch.cat(policy_loss_terms).mean()
            entropy_loss = -torch.cat(entropy_terms).mean()

            optimizer.zero_grad()
            (policy_loss + entropy_coef * entropy_loss).backward()
            optimizer.step()

            actor_loss_sum += policy_loss.item()
            entropy_sum += -entropy_loss.item()
            actor_updates += 1

        idx_c = np.random.permutation(T)
        for start in range(0, T, minibatch_size):
            mb_idx = idx_c[start : start + minibatch_size]
            states_b = torch.as_tensor(buffer.states[mb_idx], device=device)
            returns_b = torch.as_tensor(mean_returns[mb_idx], device=device)

            values_pred = policy.value(states_b)
            value_loss = 0.5 * (returns_b - values_pred).pow(2).mean()

            optimizer.zero_grad()
            (value_coef * value_loss).backward()
            optimizer.step()

            value_loss_sum += value_loss.item()
            critic_updates += 1

    return {
        "policy_loss": actor_loss_sum / max(1, actor_updates),
        "entropy": entropy_sum / max(1, actor_updates),
        "value_loss": value_loss_sum / max(1, critic_updates),
    }
