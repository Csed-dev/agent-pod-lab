"""GNN Ablation: evaluate pure Neumann series (c_k=1) WITHOUT any GNN training."""
import time
import random

import numpy as np
import torch

from lib.architectures.neumann import (
    PolyMPNN, PolynomialPreconditioner, save_checkpoint, load_checkpoint,
)
from lib.evaluation import run_evaluation, print_results
from lib.data import build_dataset

SEED = 42
NUM_LAYERS = 2
EMBED_DIM = 64
HIDDEN_DIM = 128
POLY_DEGREE = 1024
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

# Build model but DON'T train — just use the Neumann-initialized weights (c_k=1)
model = PolyMPNN(
    num_layers=NUM_LAYERS,
    embed=EMBED_DIM,
    hidden=HIDDEN_DIM,
    edge_feat_dim=2,
    poly_degree=POLY_DEGREE,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: PolyMPNN ({num_params:,} params) — NO TRAINING, c_k=1 fixed")

# Save untrained model as checkpoint
save_checkpoint(model, "best_model.pt")


def build_preconditioner(mdl, A):
    mdl.set_matrix(A)
    coeffs = mdl()
    # Override with fixed c_k = 1 (pure Neumann series)
    coeffs = torch.ones_like(coeffs)
    return PolynomialPreconditioner(coeffs, mdl.D_inv_A, mdl.D_inv, omega=JACOBI_OMEGA)


print("\nEvaluating pure Neumann series (c_k=1, no GNN)...")
t_eval_start = time.time()
eval_model = load_checkpoint("best_model.pt", device)
results = run_evaluation(eval_model, build_preconditioner, device)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print_results(
    results, num_params, 0,  # 0 epochs (no training)
    0.0, t_end - t_start,
    peak_vram_mb, 0, 0.0,
)
