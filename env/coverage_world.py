import functools
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .entities import ROLE_INDEX, ROLE_PARAMS, Site

# Discrete movement actions: stay, up, down, left, right.
_MOVES = {
    0: np.array([0.0, 0.0]),
    1: np.array([0.0, 1.0]),
    2: np.array([0.0, -1.0]),
    3: np.array([-1.0, 0.0]),
    4: np.array([1.0, 0.0]),
}

# Number of nearest sensed sites included in each agent's observation.
K_SITES_OBS = 3
NUM_ROLE_TYPES = len(ROLE_INDEX)
OBS_DIM = 2 + NUM_ROLE_TYPES + K_SITES_OBS * 3
# Per-site global-state features: normalized position (2) + status one-hot (4).
SITE_STATE_DIM = 6


class CoverageWorld(ParallelEnv):
    """Cooperative search-and-service task with two heterogeneous roles.

    Scouts detect sites but can't service them; actors service detected
    sites but have a much narrower sensor. A site that isn't serviced
    within `expire_after_detect` steps of being detected is lost. Without
    communication, actors only learn about a site by wandering into their
    own (narrow) sensor range of it.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "coverage_world_v0"}

    def __init__(
        self,
        num_roles: int = 2,
        num_agents: int = 4,
        num_scouts: int = 2,
        num_actors: int = 2,
        num_sites: int = 6,
        world_size: float = 100.0,
        max_steps: int = 200,
        service_radius: float = 6.0,
        comm_range: float = 20.0,
        expire_after_detect: int = 50,
        render_mode: Optional[str] = None,
    ):
        """num_roles=1 spawns `num_agents` identical generalist agents (the
        homogeneous baseline); num_roles=2 spawns the scout/actor split
        (num_scouts + num_actors), which is the default and matches
        milestone 1's behavior.
        """
        self.num_roles = num_roles
        self.num_scouts = num_scouts
        self.num_actors = num_actors
        self.num_sites = num_sites
        self.world_size = world_size
        self.max_steps = max_steps
        self.service_radius = service_radius
        self.comm_range = comm_range
        self.expire_after_detect = expire_after_detect
        self.render_mode = render_mode

        if num_roles == 1:
            self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
            self.roles = {a: "generalist" for a in self.possible_agents}
        else:
            self.possible_agents = [f"scout_{i}" for i in range(num_scouts)] + [
                f"actor_{i}" for i in range(num_actors)
            ]
            self.roles = {a: ("scout" if a.startswith("scout") else "actor") for a in self.possible_agents}
        self.agents = list(self.possible_agents)

        self._rng = np.random.default_rng()
        self.sites: list[Site] = []
        self.positions: dict[str, np.ndarray] = {}
        self.step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.step_count = 0

        self.positions = {
            a: self._rng.uniform(0.0, self.world_size, size=2).astype(np.float32) for a in self.agents
        }
        self.sites = [
            Site(site_id=i, pos=self._rng.uniform(0.0, self.world_size, size=2).astype(np.float32))
            for i in range(self.num_sites)
        ]

        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.step_count += 1

        for agent, action in actions.items():
            role = self.roles[agent]
            speed = ROLE_PARAMS[role]["speed"]
            delta = _MOVES[int(action)] * speed
            self.positions[agent] = np.clip(self.positions[agent] + delta, 0.0, self.world_size)

        shared_reward = -0.01
        detection_bonus = {a: 0.0 for a in self.agents}

        for site in self.sites:
            if site.serviced or site.expired or site.detected:
                continue
            for agent in self.agents:
                sensor_radius = ROLE_PARAMS[self.roles[agent]]["sensor_radius"]
                if np.linalg.norm(self.positions[agent] - site.pos) <= sensor_radius:
                    site.detected = True
                    site.expire_timer = self.expire_after_detect
                    detection_bonus[agent] += 0.1
                    break

        for site in self.sites:
            if site.serviced or site.expired or not site.detected:
                continue
            for agent in self.agents:
                if not ROLE_PARAMS[self.roles[agent]]["can_service"]:
                    continue
                if np.linalg.norm(self.positions[agent] - site.pos) <= self.service_radius:
                    site.serviced = True
                    shared_reward += 1.0
                    break

        for site in self.sites:
            if site.detected and not site.serviced and not site.expired:
                site.expire_timer -= 1
                if site.expire_timer <= 0:
                    site.expired = True
                    shared_reward -= 0.5

        terminated = all(site.serviced or site.expired for site in self.sites)
        truncated = self.step_count >= self.max_steps

        rewards = {a: shared_reward + detection_bonus[a] for a in self.agents}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {} for a in self.agents}
        observations = {a: self._observe(a) for a in self.agents}

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def comm_edges(self):
        """Pairs of agents within communication range, for the comms graph."""
        edges = []
        agents = self.agents
        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                if np.linalg.norm(self.positions[a] - self.positions[b]) <= self.comm_range:
                    edges.append((a, b))
        return edges

    @property
    def state_size(self):
        return len(self.possible_agents) * 2 + self.num_sites * SITE_STATE_DIM

    def state(self):
        """Global state for the centralized critic: all agents' normalized
        positions plus all sites' normalized positions and status one-hot.
        Valid to call right after step(), before any reset.
        """
        agent_part = np.concatenate(
            [(self.positions[a] / self.world_size) * 2 - 1 for a in self.possible_agents]
        )
        site_parts = []
        for site in self.sites:
            pos_norm = (site.pos / self.world_size) * 2 - 1
            if site.expired:
                status = [0.0, 0.0, 0.0, 1.0]
            elif site.serviced:
                status = [0.0, 0.0, 1.0, 0.0]
            elif site.detected:
                status = [0.0, 1.0, 0.0, 0.0]
            else:
                status = [1.0, 0.0, 0.0, 0.0]
            site_parts.append(np.concatenate([pos_norm, np.array(status, dtype=np.float32)]))
        return np.concatenate([agent_part] + site_parts).astype(np.float32)

    def _observe(self, agent):
        role = self.roles[agent]
        pos = self.positions[agent]
        sensor_radius = ROLE_PARAMS[role]["sensor_radius"]

        own_pos_norm = (pos / self.world_size) * 2 - 1
        role_onehot = np.zeros(NUM_ROLE_TYPES, dtype=np.float32)
        role_onehot[ROLE_INDEX[role]] = 1.0

        visible = []
        for site in self.sites:
            if site.serviced or site.expired:
                continue
            dist = np.linalg.norm(pos - site.pos)
            if dist <= sensor_radius:
                visible.append((dist, site))
        visible.sort(key=lambda t: t[0])

        site_feats = []
        for i in range(K_SITES_OBS):
            if i < len(visible):
                _, site = visible[i]
                rel = (site.pos - pos) / sensor_radius
                site_feats.extend([float(rel[0]), float(rel[1]), 1.0])
            else:
                site_feats.extend([0.0, 0.0, 0.0])

        return np.concatenate(
            [own_pos_norm, role_onehot, np.array(site_feats, dtype=np.float32)]
        ).astype(np.float32)

    def render(self, attention_edges=None):
        """attention_edges: optional list of (src_agent, dst_agent, weight)
        for milestone 5's attention visualization - draws an arrow from src
        to dst (message direction: how much dst attends to src) with width
        scaled by weight, on top of the plain comm-range edges.
        """
        import matplotlib

        if self.render_mode == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.set_title(f"step {self.step_count}")

        site_colors = {
            "undetected": "#999999",
            "detected": "#e8a33d",
            "serviced": "#3daf5f",
            "expired": "#d1453d",
        }
        for site in self.sites:
            if site.serviced:
                color = site_colors["serviced"]
            elif site.expired:
                color = site_colors["expired"]
            elif site.detected:
                color = site_colors["detected"]
            else:
                color = site_colors["undetected"]
            ax.add_patch(patches.Circle(site.pos, 1.5, color=color, zorder=3))

        for a, b in self.comm_edges():
            pa, pb = self.positions[a], self.positions[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#555555", linewidth=0.8, linestyle="--", zorder=2)

        if attention_edges:
            max_w = max((w for _, _, w in attention_edges), default=0.0) or 1.0
            for src, dst, w in attention_edges:
                if src not in self.positions or dst not in self.positions:
                    continue
                p_src, p_dst = self.positions[src], self.positions[dst]
                norm_w = w / max_w
                ax.annotate(
                    "",
                    xy=p_dst,
                    xytext=p_src,
                    arrowprops=dict(arrowstyle="->", color="#d1453d", lw=0.5 + 4 * norm_w, alpha=0.7),
                    zorder=2.5,
                )

        role_colors = {"scout": "#3d7ae8", "actor": "#8a3de8", "generalist": "#c93fd1"}
        for agent in self.possible_agents:
            pos = self.positions.get(agent)
            if pos is None:
                continue
            role = self.roles[agent]
            sensor_radius = ROLE_PARAMS[role]["sensor_radius"]
            ax.add_patch(patches.Circle(pos, sensor_radius, color=role_colors[role], alpha=0.08, zorder=1))
            ax.scatter(*pos, color=role_colors[role], s=60, zorder=4, edgecolors="black", linewidths=0.5)

        fig.tight_layout()

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return frame

        plt.show()
        plt.close(fig)
        return None
