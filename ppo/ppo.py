import numpy as np
import torch


def compute_gae(rewards, values, dones, bootstrap_values, gamma, lam):
    """Per-agent GAE. `dones` marks true termination (bootstrap value is
    zeroed); `truncs`-driven episode boundaries still bootstrap from the
    real next-state value but must not let the advantage carry across the
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
    role_data = {}
    for agent in buffer.agents:
        role = roles_by_agent[agent]
        bucket = role_data.setdefault(
            role, {"obs": [], "actions": [], "logprobs": [], "advantages": []}
        )
        bucket["obs"].append(buffer.obs[agent])
        bucket["actions"].append(buffer.actions[agent])
        bucket["logprobs"].append(buffer.logprobs[agent])
        bucket["advantages"].append(advantages[agent])

    for role, bucket in role_data.items():
        for key in ("obs", "actions", "logprobs"):
            bucket[key] = np.concatenate(bucket[key], axis=0)
        adv = np.concatenate(bucket["advantages"], axis=0)
        bucket["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    critic_states = np.concatenate([buffer.states for _ in buffer.agents], axis=0)
    critic_returns = np.concatenate([returns[agent] for agent in buffer.agents], axis=0)

    actor_loss_sum, entropy_sum, actor_steps = 0.0, 0.0, 0
    value_loss_sum, critic_steps = 0.0, 0

    for _ in range(epochs):
        for role, data in role_data.items():
            n = data["obs"].shape[0]
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                mb_idx = idx[start : start + minibatch_size]
                obs_b = torch.as_tensor(data["obs"][mb_idx], device=device)
                actions_b = torch.as_tensor(data["actions"][mb_idx], device=device)
                old_logprobs_b = torch.as_tensor(data["logprobs"][mb_idx], device=device)
                adv_b = torch.as_tensor(data["advantages"][mb_idx], device=device)

                new_logprobs, entropy = policy.evaluate_actions(role, obs_b, actions_b)
                ratio = torch.exp(new_logprobs - old_logprobs_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                optimizer.zero_grad()
                (policy_loss + entropy_coef * entropy_loss).backward()
                optimizer.step()

                actor_loss_sum += policy_loss.item()
                entropy_sum += entropy.mean().item()
                actor_steps += 1

        n = critic_states.shape[0]
        idx = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]
            states_b = torch.as_tensor(critic_states[mb_idx], device=device)
            returns_b = torch.as_tensor(critic_returns[mb_idx], device=device)

            values_pred = policy.value(states_b)
            value_loss = 0.5 * (returns_b - values_pred).pow(2).mean()

            optimizer.zero_grad()
            (value_coef * value_loss).backward()
            optimizer.step()

            value_loss_sum += value_loss.item()
            critic_steps += 1

    return {
        "policy_loss": actor_loss_sum / max(1, actor_steps),
        "entropy": entropy_sum / max(1, actor_steps),
        "value_loss": value_loss_sum / max(1, critic_steps),
    }
