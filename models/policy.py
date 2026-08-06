import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.gat import GATComm, mean_neighbor_aggregate


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActorHead(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        return self.net(x)


class MultiRolePolicy(nn.Module):
    """Per-role encoder -> optional communication step -> per-role actor
    head, plus a single centralized critic over the global state.

    comm_aggregation selects how each agent's own embedding is combined
    with in-range neighbors' embeddings before the actor head:
      - "none": own embedding only (milestones 2-3)
      - "mean": mean of in-range neighbors' embeddings (ablation baseline)
      - "gat":  learned attention over in-range neighbors (milestone 4)

    All three share this same class/forward path (unified milestone code,
    per the project's config-driven-milestones convention) rather than
    branching into separate model classes.
    """

    def __init__(
        self, roles, obs_dim, action_dim, state_dim, hidden_dim=64, comm_aggregation="none", gat_heads=4
    ):
        super().__init__()
        self.roles = list(roles)
        self.comm_aggregation = comm_aggregation
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict({role: MLP(obs_dim, hidden_dim, hidden_dim) for role in self.roles})

        head_in_dim = hidden_dim * 2 if comm_aggregation != "none" else hidden_dim
        self.actor_heads = nn.ModuleDict(
            {role: ActorHead(head_in_dim, action_dim, hidden_dim) for role in self.roles}
        )
        self.gat = GATComm(hidden_dim, hidden_dim, heads=gat_heads) if comm_aggregation == "gat" else None
        self.critic = MLP(state_dim, 1, hidden_dim)

    def _encode(self, obs_by_agent, roles_by_agent, agent_order, device):
        embeds = [
            self.encoders[roles_by_agent[agent]](
                torch.as_tensor(obs_by_agent[agent], device=device, dtype=torch.float32).unsqueeze(0)
            )
            for agent in agent_order
        ]
        return torch.cat(embeds, dim=0)

    def _communicate(self, node_embeds, edge_index, return_attention=False):
        if self.comm_aggregation == "gat":
            return self.gat(node_embeds, edge_index, return_attention=return_attention)
        if self.comm_aggregation == "mean":
            msg = mean_neighbor_aggregate(node_embeds, edge_index)
            return (msg, None) if return_attention else msg
        return (None, None) if return_attention else None

    def _logits(self, node_embeds, messages, roles_by_agent, agent_order):
        logits = {}
        for i, agent in enumerate(agent_order):
            role = roles_by_agent[agent]
            own = node_embeds[i : i + 1]
            combined = own if messages is None else torch.cat([own, messages[i : i + 1]], dim=-1)
            logits[agent] = self.actor_heads[role](combined)
        return logits

    def step(self, obs_by_agent, roles_by_agent, agent_order, edge_index, device):
        """Sample actions for one timestep (used during rollout collection)."""
        node_embeds = self._encode(obs_by_agent, roles_by_agent, agent_order, device)
        messages = self._communicate(node_embeds, edge_index)
        logits = self._logits(node_embeds, messages, roles_by_agent, agent_order)

        actions, logprobs = {}, {}
        for agent, agent_logits in logits.items():
            dist = Categorical(logits=agent_logits)
            action = dist.sample()
            actions[agent] = int(action.item())
            logprobs[agent] = dist.log_prob(action).item()
        return actions, logprobs

    def act_greedy(self, obs_by_agent, roles_by_agent, agent_order, edge_index, device, return_attention=False):
        """Deterministic (argmax) actions, for evaluation rollouts. With
        return_attention=True (only meaningful when comm_aggregation="gat"),
        also returns this timestep's attention weights for milestone 5.
        """
        node_embeds = self._encode(obs_by_agent, roles_by_agent, agent_order, device)
        if return_attention:
            messages, attn = self._communicate(node_embeds, edge_index, return_attention=True)
        else:
            messages, attn = self._communicate(node_embeds, edge_index), None
        logits = self._logits(node_embeds, messages, roles_by_agent, agent_order)
        actions = {agent: int(torch.argmax(agent_logits, dim=-1).item()) for agent, agent_logits in logits.items()}
        return (actions, attn) if return_attention else actions

    def _encode_batch(self, obs_batch, roles_by_agent, agent_order, device):
        """obs_batch: dict agent -> (B, obs_dim) array. Returns (B, N, hidden_dim),
        vectorized across the batch dimension B (one matmul per role, not a
        per-sample Python loop).
        """
        B = next(iter(obs_batch.values())).shape[0]
        role_to_positions = {}
        for i, agent in enumerate(agent_order):
            role_to_positions.setdefault(roles_by_agent[agent], []).append(i)

        embeds = torch.zeros(B, len(agent_order), self.hidden_dim, device=device)
        for role, positions in role_to_positions.items():
            agents_in_role = [agent_order[i] for i in positions]
            obs_stack = torch.stack(
                [torch.as_tensor(obs_batch[a], device=device, dtype=torch.float32) for a in agents_in_role],
                dim=1,
            )  # (B, n_role, obs_dim)
            n_role = len(agents_in_role)
            out = self.encoders[role](obs_stack.reshape(B * n_role, -1)).reshape(B, n_role, -1)
            embeds[:, positions, :] = out
        return embeds

    def _communicate_batch(self, node_embeds, edge_index_batched):
        """node_embeds: (B, N, H); edge_index_batched: (2, E) with node indices
        already offset per-timestep into [0, B*N). Reuses `_communicate`
        unchanged, since GATComm/mean_neighbor_aggregate operate on an
        arbitrary node/edge set regardless of how it's grouped into graphs.
        """
        B, N, H = node_embeds.shape
        messages_flat = self._communicate(node_embeds.reshape(B * N, H), edge_index_batched)
        return None if messages_flat is None else messages_flat.reshape(B, N, H)

    def evaluate_batch(self, obs_batch, roles_by_agent, agent_order, edge_index_batched, actions_batch, device):
        """Evaluate given actions across a batch of B timesteps at once (used
        during the PPO update): returns per-agent log-prob and entropy, each
        shape (B,), with gradients enabled. `edge_index_batched` must have
        node indices pre-offset per-timestep (see ppo.rollout.batch_edge_indices).
        """
        node_embeds = self._encode_batch(obs_batch, roles_by_agent, agent_order, device)
        messages = self._communicate_batch(node_embeds, edge_index_batched)

        logprobs, entropies = {}, {}
        for i, agent in enumerate(agent_order):
            role = roles_by_agent[agent]
            own = node_embeds[:, i, :]
            combined = own if messages is None else torch.cat([own, messages[:, i, :]], dim=-1)
            logits = self.actor_heads[role](combined)
            dist = Categorical(logits=logits)
            actions_t = torch.as_tensor(actions_batch[agent], device=device)
            logprobs[agent] = dist.log_prob(actions_t)
            entropies[agent] = dist.entropy()
        return logprobs, entropies

    def value(self, state):
        return self.critic(state).squeeze(-1)
