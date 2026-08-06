import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, agents, obs_dim, state_dim, rollout_length):
        self.agents = list(agents)
        self.rollout_length = rollout_length
        self.obs = {a: np.zeros((rollout_length, obs_dim), dtype=np.float32) for a in self.agents}
        self.actions = {a: np.zeros(rollout_length, dtype=np.int64) for a in self.agents}
        self.logprobs = {a: np.zeros(rollout_length, dtype=np.float32) for a in self.agents}
        self.rewards = {a: np.zeros(rollout_length, dtype=np.float32) for a in self.agents}
        self.states = np.zeros((rollout_length, state_dim), dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=bool)
        self.truncs = np.zeros(rollout_length, dtype=bool)
        self.bootstrap_values = np.zeros(rollout_length, dtype=np.float32)


@torch.no_grad()
def collect_rollout(env, policy, rollout_length, device, obs=None):
    """Collect `rollout_length` env steps, resetting internally on episode
    end. `obs` carries the current observations across successive calls so
    training doesn't waste a reset between rollouts. Returns the filled
    buffer, a list of finished-episode summaries, and the observations to
    pass into the next call.
    """
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    buffer = RolloutBuffer(env.possible_agents, obs_dim, env.state_size, rollout_length)
    episode_summaries = []

    if obs is None:
        obs, _ = env.reset()

    ep_return = {a: 0.0 for a in env.possible_agents}
    ep_len = 0

    for t in range(rollout_length):
        state = env.state()
        value = policy.value(torch.as_tensor(state, device=device).unsqueeze(0)).item()

        actions = {}
        for agent in env.agents:
            role = env.roles[agent]
            obs_t = torch.as_tensor(obs[agent], device=device).unsqueeze(0)
            action, logprob = policy.act(role, obs_t)
            actions[agent] = int(action.item())
            buffer.obs[agent][t] = obs[agent]
            buffer.actions[agent][t] = int(action.item())
            buffer.logprobs[agent][t] = logprob.item()

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        next_state = env.state()
        next_value = policy.value(torch.as_tensor(next_state, device=device).unsqueeze(0)).item()

        done = any(terminations.values())
        trunc = any(truncations.values())

        for agent in buffer.agents:
            r = rewards[agent]
            buffer.rewards[agent][t] = r
            ep_return[agent] += r

        buffer.states[t] = state
        buffer.values[t] = value
        buffer.dones[t] = done
        buffer.truncs[t] = trunc
        buffer.bootstrap_values[t] = next_value
        ep_len += 1

        if done or trunc:
            episode_summaries.append(
                {
                    "length": ep_len,
                    "return": sum(ep_return.values()),
                    "serviced": sum(1 for s in env.sites if s.serviced),
                    "expired": sum(1 for s in env.sites if s.expired),
                    "num_sites": len(env.sites),
                }
            )
            obs, _ = env.reset()
            ep_return = {a: 0.0 for a in env.possible_agents}
            ep_len = 0
        else:
            obs = next_obs

    return buffer, episode_summaries, obs

