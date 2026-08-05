# coverage-marl

Multi-agent reinforcement learning experiment: heterogeneous team coordination via a
graph attention network (GAT) communication layer, trained with PPO (MAPPO-style
centralized-critic / decentralized-actor setup).

## Task

`CoverageWorld` is a cooperative PettingZoo `ParallelEnv`: a team of two roles,
**scouts** (fast, wide sensor, cannot service sites) and **actors** (slow, narrow
sensor, can service a detected site), searches a bounded 2D area for randomly placed
sites of interest and services them before they expire. Detection and servicing are
split across roles by design, so the team's performance is capped unless scouts
communicate detected site locations to actors — this is the mechanism the GAT
communication layer is meant to improve.

## Architecture

- Per-role observation encoders (heterogeneous obs spaces -> shared latent dim)
- Dynamic communication graph (edge iff two agents are within comm range)
- GAT (`GATv2Conv`) message passing over that graph, exposing attention weights
- Role-specific decentralized actor heads; centralized critic (global state) used
  only during training
- PPO with GAE and clipped surrogate objective

## Project structure

```
env/      CoverageWorld PettingZoo environment, render()
models/   per-role encoders, GAT comms module, actor/critic heads
ppo/      rollout collection, PPO update, training entrypoint
eval/     metrics, replay logging/rendering, attention visualization
configs/  one YAML per experiment/milestone (see below)
runs/     TensorBoard logs (gitignored contents)
replays/  saved episode replay files for offline rendering (gitignored contents)
```

## Milestones

Each milestone is a config, not a separate code path — the same training entrypoint
supports all of them via flags (`num_roles`, `use_comms`, `comm_aggregation`,
`graph_connectivity`), so earlier milestones stay runnable as later ones are added.

1. Environment sanity check (random policy, render)
2. Homogeneous single-role PPO baseline (no comms)
3. Heterogeneous two-role PPO, no communication (establishes the comms-gap baseline)
4. Heterogeneous two-role PPO + GAT communication
5. Ablations (GAT vs. mean-pooling vs. no-comms vs. fully-connected attention) +
   attention-weight visualization

## Visualization

- Scalar training metrics: TensorBoard (`runs/`)
- Episode/attention visualization: rollouts are logged to `replays/`, then rendered
  offline to video via `eval/render_replay.py` (comm-graph edges and GAT attention
  weights overlaid on agent positions)
