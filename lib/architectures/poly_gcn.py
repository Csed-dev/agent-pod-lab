"""PolyGCN: GCN-Backbone-Variante des Neumann-Polynom-Preconditioners.

Spiegel zu lib.architectures.neumann.PolyMPNN, aber mit Kipf-Welling-GCN
statt MPNN. Kein Edge-Features, da GCN nur ueber die normalisierte
Adjazenz-Multiplikation aggregiert.

Das ist die Architektur aus ADR-13 (v3 minimal reference architecture)
des MatrixPFN-Thesis-Repos, hier in agent-pod-lab portiert fuer
GPU-Experimente.
"""
import torch
from torch import nn
import torch.nn.functional as F


NUM_NODE_FEATURES = 3


class GCNConv(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.AA = None

    def set_adjacency(self, AA: torch.Tensor) -> None:
        self.AA = AA

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.sparse.mm(self.AA, h))


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


class PolyGCN(nn.Module):

    def __init__(self, num_layers: int, embed: int, hidden: int,
                 poly_degree: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed = embed
        self.hidden = hidden
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
            self.convs.append(GCNConv(embed, embed))
            self.skips.append(nn.Linear(embed, embed))
            self.norms.append(nn.LayerNorm(embed))

        self.poly_head = PolynomialHead(embed, poly_degree)

        self.node_features = None
        self.D_inv = None
        self.D_inv_A = None
        self.n = None

    def set_matrix(self, A: torch.Tensor) -> None:
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

        scaled_values = (values / gamma).float()
        AA = torch.sparse_coo_tensor(indices, scaled_values, (n, n)).coalesce()
        for layer in self.convs:
            layer.set_adjacency(AA)

        self.n = n
        self.D_inv = 1.0 / diag
        d_inv_values = self.D_inv[rows] * values
        self.D_inv_A = torch.sparse_coo_tensor(
            indices, d_inv_values, (n, n)
        ).coalesce().to_sparse_csc()

    def forward(self) -> torch.Tensor:
        h = self.node_encoder(self.node_features)
        for i in range(self.num_layers):
            h_new = self.convs[i](h) + self.skips[i](h)
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
        coeffs_k = coeffs[:, k:k + 1]
        MAv = MAv + coeffs_k * power

    v_f32 = v.float()
    residual = MAv - v_f32
    per_probe = (residual ** 2).sum(dim=0) / (v_f32 ** 2).sum(dim=0).clamp(min=1e-12)
    return per_probe.mean()


def save_checkpoint(model: PolyGCN, path: str) -> None:
    torch.save({
        "model_type": "PolyGCN",
        "config": {
            "num_layers": model.num_layers,
            "embed": model.embed,
            "hidden": model.hidden,
            "poly_degree": model.poly_degree,
        },
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: str, device: torch.device) -> PolyGCN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = PolyGCN(
        num_layers=config["num_layers"],
        embed=config["embed"],
        hidden=config["hidden"],
        poly_degree=config["poly_degree"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model
