"""GNN-Ablation fuer PolyGCN: feste c_k = 1, kein Lernen.

Identisches Setup wie v3-gcn-capacity, aber die GCN wird nicht trainiert.
Stattdessen werden die Koeffizienten zur Eval-Zeit auf 1 gesetzt
(reine Neumann-Reihe ohne ML-Komponente).

Beantwortet: Liefert v3-gcn-capacity einen Mehrwert ueber feste c_k = 1?
Wenn dieses Experiment dieselben Scores liefert wie v3-gcn-capacity,
ist das gelernte GCN-Backbone nutzlos und mein "Durchbruch" ist nur die
Reproduktion des bekannten Run-67-Resultats (PolyMPNN K=1024) bzw. der
no-gnn-Ablation aus FINDINGS.md.
"""
import time
import random

import numpy as np
import torch

from lib.architectures.poly_gcn import (
    PolyGCN, PolynomialPreconditioner, save_checkpoint, load_checkpoint,
)
from lib.evaluation import run_evaluation, print_results

SEED = 42
NUM_LAYERS = 4
EMBED_DIM = 64
HIDDEN_DIM = 128
POLY_DEGREE = 1024
JACOBI_OMEGA = 0.9

t_start = time.time()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

model = PolyGCN(
    num_layers=NUM_LAYERS,
    embed=EMBED_DIM,
    hidden=HIDDEN_DIM,
    poly_degree=POLY_DEGREE,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: PolyGCN ({num_params:,} params) -- NO TRAINING, c_k = 1 fixed")

save_checkpoint(model, "best_model.pt")


def build_preconditioner(mdl, A):
    mdl.set_matrix(A)
    coeffs = mdl()
    coeffs = torch.ones_like(coeffs)
    return PolynomialPreconditioner(coeffs, mdl.D_inv_A, mdl.D_inv, omega=JACOBI_OMEGA)


print("\nEvaluating pure Neumann series (c_k = 1, no GCN)...")
eval_model = load_checkpoint("best_model.pt", device)
results = run_evaluation(eval_model, build_preconditioner, device)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print_results(
    results, num_params, 0,
    0.0, t_end - t_start,
    peak_vram_mb, 0, 0.0,
)
