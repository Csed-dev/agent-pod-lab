"""Sign correction: use |D| for negative-diagonal matrices."""
import time
import random

import numpy as np
import torch

from lib.architectures.neumann import (
    PolyMPNN, poly_frobenius_loss,
    save_checkpoint, load_checkpoint,
)
from lib.training import TrainConfig, train_loop
from lib.evaluation import run_evaluation, print_results
from lib.data import build_dataset

SEED = 42
NUM_LAYERS = 2
EMBED_DIM = 64
HIDDEN_DIM = 128
POLY_DEGREE = 512
LR = 3e-4
JACOBI_OMEGA = 0.9
DOMAINS = "DIFFUSION,ELASTICITY,STOKES,DIFFUSION_ADVECTION,VARIABLE_DIFFUSION,SPECTRAL_STRESS,GRAPH_LAPLACIAN,ENHANCED_ADVECTION"
GRID_SIZES = (16, 24, 32, 48)

t_start = time.time()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset, num_domains = build_dataset(DOMAINS.split(","), GRID_SIZES, device)

model = PolyMPNN(
    num_layers=NUM_LAYERS,
    embed=EMBED_DIM,
    hidden=HIDDEN_DIM,
    edge_feat_dim=2,
    poly_degree=POLY_DEGREE,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: PolyMPNN ({num_params:,} params) + sign correction")


def loss_fn(mdl, A, num_probes):
    mdl.set_matrix(A)
    coeffs = mdl()
    return poly_frobenius_loss(A, coeffs, mdl.D_inv_A, mdl.D_inv, num_probes, omega=JACOBI_OMEGA)


train_result = train_loop(model, dataset, loss_fn, save_checkpoint, TrainConfig(lr=LR))

print(f"\nTraining done: {train_result.num_epochs} epochs in {train_result.training_seconds:.1f}s")
print(f"Best loss: {train_result.best_loss:.4e}")

print("\nEvaluating with sign correction...")
t_eval_start = time.time()
eval_model = load_checkpoint(train_result.checkpoint_path, device)


class SignCorrectedPreconditioner:
    """Neumann preconditioner with sign correction for negative diagonals."""

    def __init__(self, coeffs, D_inv_A, D_inv, omega, sign):
        self.coeffs = coeffs.double()
        self.D_inv_A = D_inv_A
        self.D_inv = D_inv
        self.omega = omega
        self.sign = sign

    def apply(self, r):
        K = self.coeffs.shape[1]
        omega = self.omega
        d_inv_r = omega * self.D_inv * r
        power = d_inv_r
        result = self.coeffs[:, 0] * power
        for k in range(1, K):
            power = power - omega * (self.D_inv_A @ power)
            result = result + self.coeffs[:, k] * power
        return self.sign * result


def build_preconditioner(mdl, A):
    mdl.set_matrix(A)
    # Detect negative diagonal and apply sign correction
    sign = 1.0
    if (mdl.D_inv < 0).sum() > mdl.n // 2:
        sign = -1.0
        # Rebuild D_inv_A with |D|
        D_inv_abs = 1.0 / mdl.D_inv.abs().clamp(min=1e-15)
        # Recompute D_inv_A with |D|^{-1}
        A_coo = A.to_sparse_coo().coalesce() if A.layout == torch.sparse_csc else A.coalesce()
        indices = A_coo.indices()
        values = A_coo.values()
        rows = indices[0]
        d_inv_values = D_inv_abs[rows] * values
        D_inv_A_corrected = torch.sparse_coo_tensor(
            indices, d_inv_values, (mdl.n, mdl.n)
        ).coalesce().to_sparse_csc()
        coeffs = mdl()
        return SignCorrectedPreconditioner(coeffs, D_inv_A_corrected, D_inv_abs, JACOBI_OMEGA, sign)
    else:
        coeffs = mdl()
        from lib.architectures.neumann import PolynomialPreconditioner
        return PolynomialPreconditioner(coeffs, mdl.D_inv_A, mdl.D_inv, omega=JACOBI_OMEGA)


results = run_evaluation(eval_model, build_preconditioner, device)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print_results(
    results, num_params, train_result.num_epochs,
    train_result.training_seconds, t_end - t_start,
    peak_vram_mb, num_domains, train_result.best_loss,
)
