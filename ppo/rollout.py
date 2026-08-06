import numpy as np
import torch


def build_edge_index(env, agent_to_idx):
    """Bidirectional edge_index (2, E) over agents currently in comm range,
    as node indices per `agent_to_idx`. Empty (2, 0) if nobody's in range.
    """
    edges = env.comm_edges()
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    src, dst = [], []
    for a, b in edges:
        ia, ib = agent_to_idx[a], agent_to_idx[b]
        src.extend([ia, ib])
        dst.extend([ib, ia])
    return np.array([src, dst], dtype=np.int64)


def batch_edge_indices(edge_index_list, timestep_indices, n_nodes):
    """Combine several timesteps' (2, E_t) edge_index arrays into one graph
    of B*n_nodes nodes, offsetting each timestep's node indices by its
    position in the batch so timesteps never share edges. Lets the GAT/mean
    communication step run once per minibatch instead of once per timestep.
    """
    parts = []
    for b, t in enumerate(timestep_indices):
        edge_index = edge_index_list[t]
        if edge_index.shape[1] == 0:
            continue
        parts.append(edge_index + b * n_nodes)
    if not parts:
        return np.zeros((2, 0), dtype=np.int64)
    return np.concatenate(parts, axis=1)


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
        # Per-timestep comms graph (edge set differs step to step), needed
        # to replay the same graph context when recomputing log-probs in
        # the PPO update.
        self.edge_indices = [None] * rollout_length


@torch.no_grad()
def collect_rollout(env, policy, rollout_length, device, obs=None):
    """Collect `rollout_length` env steps, resetting internally on episode
    end. `obs` carries the current observations across successive calls so
    training doesn't waste a reset between rollouts. Returns the filled
    buffer, a list of finished-episode summaries, and the observations to
    pass into the next call.
    """
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    agent_order = env.possible_agents
    agent_to_idx = {a: i for i, a in enumerate(agent_order)}
    buffer = RolloutBuffer(agent_order, obs_dim, env.state_size, rollout_length)
    episode_summaries = []

    if obs is None:
        obs, _ = env.reset()

    ep_return = {a: 0.0 for a in agent_order}
    ep_len = 0

    for t in range(rollout_length):
        state = env.state()
        value = policy.value(torch.as_tensor(state, device=device).unsqueeze(0)).item()

        edge_index_np = build_edge_index(env, agent_to_idx)
        edge_index = torch.as_tensor(edge_index_np, device=device)
        actions, logprobs = policy.step(obs, env.roles, agent_order, edge_index, device)

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        next_state = env.state()
        next_value = policy.value(torch.as_tensor(next_state, device=device).unsqueeze(0)).item()

        done = any(terminations.values())
        trunc = any(truncations.values())

        for agent in agent_order:
            buffer.obs[agent][t] = obs[agent]
            buffer.actions[agent][t] = actions[agent]
            buffer.logprobs[agent][t] = logprobs[agent]
            r = rewards[agent]
            buffer.rewards[agent][t] = r
            ep_return[agent] += r

        buffer.states[t] = state
        buffer.values[t] = value
        buffer.dones[t] = done
        buffer.truncs[t] = trunc
        buffer.bootstrap_values[t] = next_value
        buffer.edge_indices[t] = edge_index_np
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
            ep_return = {a: 0.0 for a in agent_order}
            ep_len = 0
        else:
            obs = next_obs

    return buffer, episode_summaries, obs
