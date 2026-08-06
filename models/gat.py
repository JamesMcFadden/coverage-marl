import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATComm(nn.Module):
    """Learned attention over each agent's in-range neighbors. Falls back to
    a zero message for every node when nobody is in range this timestep,
    since GATv2Conv doesn't accept an empty edge_index.
    """

    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.out_dim = out_dim
        self.conv = GATv2Conv(in_dim, out_dim, heads=heads, concat=False, add_self_loops=False)

    def forward(self, x, edge_index, return_attention=False):
        if edge_index.shape[1] == 0:
            out = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
            return (out, None) if return_attention else out
        if return_attention:
            out, (edge_index_out, alpha) = self.conv(x, edge_index, return_attention_weights=True)
            return out, (edge_index_out, alpha)
        return self.conv(x, edge_index)


def mean_neighbor_aggregate(node_embeds, edge_index):
    """Unweighted mean of in-range neighbors' embeddings per node — the
    comm_aggregation="mean" ablation baseline for milestone 5. Nodes with no
    in-range neighbor get a zero message, matching GATComm's convention.
    """
    n = node_embeds.shape[0]
    out = torch.zeros_like(node_embeds)
    if edge_index.shape[1] == 0:
        return out
    src, dst = edge_index[0], edge_index[1]
    counts = torch.zeros(n, device=node_embeds.device, dtype=node_embeds.dtype)
    out.index_add_(0, dst, node_embeds[src])
    counts.index_add_(0, dst, torch.ones_like(dst, dtype=node_embeds.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(-1)
    return out / counts
