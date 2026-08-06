import torch
import torch.nn as nn
from torch.distributions import Categorical


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


class MultiRolePolicy(nn.Module):
    """One actor network per role (parameter-shared within a role) plus a
    single centralized critic over the global state. With a single role
    (milestone 2's homogeneous baseline) this reduces to one shared actor;
    milestone 3 adds a second role's actor without changing this class.
    """

    def __init__(self, roles, obs_dim, action_dim, state_dim, hidden_dim=64):
        super().__init__()
        self.roles = list(roles)
        self.actors = nn.ModuleDict({role: MLP(obs_dim, action_dim, hidden_dim) for role in self.roles})
        self.critic = MLP(state_dim, 1, hidden_dim)

    def act(self, role, obs):
        logits = self.actors[role](obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, role, obs, actions):
        logits = self.actors[role](obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def value(self, state):
        return self.critic(state).squeeze(-1)
