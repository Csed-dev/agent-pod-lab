import torch
from torch import nn
import torch.nn.functional as F


NUM_NODE_FEATURES = 3
NUM_EDGE_FEATURES = 2


class MPNNConv(nn.Module):

    def __init__(self, node_dim: int, out_dim: int, edge_feat_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_feat_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor, n: int) -> torch.Tensor:
        rows, cols = edge_index
        msg_input = torch.cat([h[rows], h[cols], edge_features], dim=-1)
        messages = self.message_fn(msg_input)
        out = torch.zeros(n, self.out_dim, dtype=h.dtype, device=h.device)
        out.scatter_add_(0, rows.unsqueeze(-1).expand_as(messages), messages)
        return out


class PolynomialHead(nn.Module):

    def __init__(self, node_dim: int, poly_degree: int):
        super().__init__()
        self.poly_degree = poly_degree
        self.coeff_net = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, poly_degree),
        )
        nn.init.zeros_(self.coeff_net[-1].weight)
        nn.init.zeros_(self.coeff_net[-1].bias)
        with torch.no_grad():
            self.coeff_net[-1].bias.fill_(1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.coeff_net(h)


class PolyMPNN(nn.Module):

    def __init__(self, num_layers: int, embed: int, hidden: int,
                 edge_feat_dim: int, poly_degree: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed = embed
        self.hidden = hidden
        self.edge_feat_dim = edge_feat_dim
        self.poly_degree = poly_degree

        self.node_encoder = nn.Sequential(
            nn.Linear(NUM_NODE_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed),
        )

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNConv(embed, embed, edge_feat_dim))
            self.skips.append(nn.Linear(embed, embed))
            self.norms.append(nn.LayerNorm(embed))

        self.poly_head = PolynomialHead(embed, poly_degree)

        self.edge_index = None
        self.edge_features = None
        self.node_features = None
        self.D_inv = None
        self.D_inv_A = None
        self.n = None

    def set_matrix(self, A: torch.Tensor):
        if A.layout == torch.sparse_csc:
            A_coo = A.to_sparse_coo().coalesce()
        else:
            A_coo = A.coalesce()

        indices = A_coo.indices()
        values = A_coo.values()
        n = A.shape[0]
        rows, cols = indices

        diag = torch.zeros(n, dtype=values.dtype, device=values.device)
        diag_mask = rows == cols
        diag[rows[diag_mask]] = values[diag_mask]

        if (diag.abs() < 1e-15).any():
            raise ValueError(f"Matrix has {(diag.abs() < 1e-15).sum()} near-zero diagonal entries")

        row_norms = torch.zeros(n, dtype=values.dtype, device=values.device)
        row_norms.scatter_add_(0, rows, values.abs())
        row_norms = row_norms.clamp(min=1e-12)

        gamma = row_norms.max().item()

        self.node_features = torch.stack([
            diag / gamma,
            diag.abs() / row_norms,
            row_norms / gamma,
        ], dim=-1).float()

        diag_at_row = diag[rows].abs()
        self.edge_features = torch.stack([
            values / gamma,
            values.abs() / diag_at_row,
        ], dim=-1).float()

        self.edge_index = indices
        self.n = n
        self.D_inv = 1.0 / diag

        d_inv_values = self.D_inv[rows] * values
        self.D_inv_A = torch.sparse_coo_tensor(
            indices, d_inv_values, (n, n)
        ).coalesce().to_sparse_csc()

    def forward(self) -> torch.Tensor:
        h = self.node_encoder(self.node_features)

        for i in range(self.num_layers):
            h_new = self.convs[i](h, self.edge_index, self.edge_features, self.n)
            h_new = h_new + self.skips[i](h)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h = h_new

        return self.poly_head(h)


class PolynomialPreconditioner:

    def __init__(self, coeffs: torch.Tensor, D_inv_A: torch.Tensor,
                 D_inv: torch.Tensor, omega: float = 0.9):
        self.coeffs = coeffs.double()
        self.D_inv_A = D_inv_A
        self.D_inv = D_inv
        self.omega = omega

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        K = self.coeffs.shape[1]
        omega = self.omega

        d_inv_r = omega * self.D_inv * r
        power = d_inv_r
        result = self.coeffs[:, 0] * power

        for k in range(1, K):
            power = power - omega * (self.D_inv_A @ power)
            result = result + self.coeffs[:, k] * power

        return result


def poly_frobenius_loss(A: torch.Tensor, coeffs: torch.Tensor,
                        D_inv_A: torch.Tensor, D_inv: torch.Tensor,
                        num_probes: int, omega: float = 0.9) -> torch.Tensor:
    n = A.shape[0]
    device = A.device
    K = coeffs.shape[1]

    v = torch.randn(n, num_probes, dtype=torch.float64, device=device)
    Av = A @ v

    D_inv_unsq = D_inv.unsqueeze(-1)
    d_inv_Av = omega * D_inv_unsq * Av

    power = d_inv_Av.float()
    coeffs_0 = coeffs[:, 0:1]
    MAv = coeffs_0 * power

    D_inv_A_f32 = D_inv_A.float()
    for k in range(1, K):
        power = power - omega * (D_inv_A_f32 @ power)
        coeffs_k = coeffs[:, k:k+1]
        MAv = MAv + coeffs_k * power

    v_f32 = v.float()
    residual = MAv - v_f32
    per_probe = (residual ** 2).sum(dim=0) / (v_f32 ** 2).sum(dim=0).clamp(min=1e-12)
    return per_probe.mean()


def save_checkpoint(model: PolyMPNN, path: str) -> None:
    torch.save({
        "model_type": "PolyMPNN",
        "config": {
            "num_layers": model.num_layers,
            "embed": model.embed,
            "hidden": model.hidden,
            "edge_feat_dim": model.edge_feat_dim,
            "poly_degree": model.poly_degree,
        },
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: str, device: torch.device) -> PolyMPNN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = PolyMPNN(
        num_layers=config["num_layers"],
        embed=config["embed"],
        hidden=config["hidden"],
        edge_feat_dim=config["edge_feat_dim"],
        poly_degree=config["poly_degree"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model
