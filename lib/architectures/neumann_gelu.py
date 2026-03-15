"""Neumann architecture with GELU activation instead of ReLU."""
import torch
from torch import nn
import torch.nn.functional as F

from lib.architectures.neumann import (
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    PolynomialHead, PolynomialPreconditioner,
    poly_frobenius_loss, save_checkpoint as _save_checkpoint,
)


class MPNNConvGELU(nn.Module):
    def __init__(self, node_dim, out_dim, edge_feat_dim):
        super().__init__()
        self.out_dim = out_dim
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h, edge_index, edge_features, n):
        rows, cols = edge_index
        msg_input = torch.cat([h[rows], h[cols], edge_features], dim=-1)
        messages = self.message_fn(msg_input)
        out = torch.zeros(n, self.out_dim, dtype=h.dtype, device=h.device)
        out.scatter_add_(0, rows.unsqueeze(-1).expand_as(messages), messages)
        return out


class PolyMPNN_GELU(nn.Module):
    def __init__(self, num_layers, embed, hidden, edge_feat_dim, poly_degree):
        super().__init__()
        self.num_layers = num_layers
        self.embed = embed
        self.hidden = hidden
        self.edge_feat_dim = edge_feat_dim
        self.poly_degree = poly_degree

        self.node_encoder = nn.Sequential(
            nn.Linear(NUM_NODE_FEATURES, hidden), nn.GELU(), nn.Linear(hidden, embed),
        )
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNConvGELU(embed, embed, edge_feat_dim))
            self.skips.append(nn.Linear(embed, embed))
            self.norms.append(nn.LayerNorm(embed))
        self.poly_head = PolynomialHead(embed, poly_degree)

        self.edge_index = self.edge_features = self.node_features = None
        self.D_inv = self.D_inv_A = self.n = None

    def set_matrix(self, A):
        from lib.architectures.neumann import PolyMPNN
        # Reuse the parent's set_matrix logic
        tmp = PolyMPNN.__new__(PolyMPNN)
        tmp.__dict__ = self.__dict__
        PolyMPNN.set_matrix(tmp, A)
        self.__dict__.update(tmp.__dict__)

    def forward(self):
        h = self.node_encoder(self.node_features)
        for i in range(self.num_layers):
            h_new = self.convs[i](h, self.edge_index, self.edge_features, self.n)
            h_new = h_new + self.skips[i](h)
            h_new = self.norms[i](h_new)
            h_new = F.gelu(h_new)
            h = h_new
        return self.poly_head(h)


def save_checkpoint(model, path):
    torch.save({
        "model_type": "PolyMPNN_GELU",
        "config": {
            "num_layers": model.num_layers, "embed": model.embed,
            "hidden": model.hidden, "edge_feat_dim": model.edge_feat_dim,
            "poly_degree": model.poly_degree,
        },
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    c = ckpt["config"]
    m = PolyMPNN_GELU(c["num_layers"], c["embed"], c["hidden"], c["edge_feat_dim"], c["poly_degree"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    return m
