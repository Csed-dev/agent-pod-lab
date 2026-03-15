from dataclasses import dataclass, field

import numpy as np
import torch

from matrixpfn.precond.jacobi import Jacobi
from matrixpfn.solver.fgmres import FGMRES
from matrixpfn.generator.domains.diffusion import DiffusionGenerator

from prepare import (
    EVAL_MATRICES, SYNTHETIC_EVAL_GRIDS, SYNTHETIC_TRAINING_GRIDS,
    NUM_RHS, NUM_SYNTHETIC_MATRICES, FGMRES_RESTART, FGMRES_MAX_ITERS,
    FGMRES_RTOL, FGMRES_TIMEOUT, ILU_REFERENCE, AMG_REFERENCE,
    load_suitesparse_matrix,
)


@dataclass
class SolveAccumulator:
    normalized_iters: list[float] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)

    def record_solve(self, result) -> None:
        self.normalized_iters.append(result.iterations / FGMRES_MAX_ITERS)
        self.converged.append(result.converged)

    def record_failure(self) -> None:
        self.normalized_iters.append(1.0)
        self.converged.append(False)

    def mean_normalized_iter(self) -> float:
        if not self.normalized_iters:
            return 1.0
        return sum(self.normalized_iters) / len(self.normalized_iters)

    def convergence_rate(self) -> float:
        if not self.converged:
            return 0.0
        return sum(self.converged) / len(self.converged)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    return sum(values) / len(values)


@torch.no_grad()
def run_evaluation(model, build_preconditioner_fn, device: torch.device) -> dict:
    model.eval()
    solver = FGMRES(
        restart=FGMRES_RESTART,
        max_iters=FGMRES_MAX_ITERS,
        rtol=FGMRES_RTOL,
        timeout=FGMRES_TIMEOUT,
    )

    torch.manual_seed(99999)
    np.random.seed(99999)

    synthetic_results = _evaluate_synthetic(model, build_preconditioner_fn, solver, device)
    suitesparse_results = _evaluate_suitesparse(model, build_preconditioner_fn, solver, device)

    synth_score = _safe_mean(synthetic_results["all_pfn_iters"])
    ss_score = _safe_mean(suitesparse_results["all_pfn_iters"])
    synth_jacobi = _safe_mean(synthetic_results["all_jacobi_iters"])
    ss_jacobi = _safe_mean(suitesparse_results["all_jacobi_iters"])

    synth_pfn_conv = synthetic_results["all_pfn_conv"]
    ss_pfn_conv = suitesparse_results["all_pfn_conv"]

    return {
        "score": 0.3 * synth_score + 0.7 * ss_score,
        "synthetic_score": synth_score,
        "suitesparse_score": ss_score,
        "synthetic_jacobi": synth_jacobi,
        "suitesparse_jacobi": ss_jacobi,
        "synthetic_conv_pct": sum(synth_pfn_conv) / len(synth_pfn_conv) * 100 if synth_pfn_conv else 0.0,
        "suitesparse_conv_pct": sum(ss_pfn_conv) / len(ss_pfn_conv) * 100 if ss_pfn_conv else 0.0,
        "synthetic_details": synthetic_results["details"],
        "suitesparse_details": suitesparse_results["details"],
        "ilu_reference": ILU_REFERENCE,
        "amg_reference": AMG_REFERENCE,
    }


def _solve_with_preconditioner(solver, A, b, preconditioner, accumulator: SolveAccumulator) -> None:
    try:
        result = solver.solve(A, b, M=preconditioner, progress_bar=False)
        accumulator.record_solve(result)
    except Exception:
        accumulator.record_failure()


def _evaluate_synthetic(model, build_preconditioner_fn, solver, device) -> dict:
    all_pfn_iters = []
    all_jacobi_iters = []
    all_pfn_conv = []
    details = {}

    for gs in SYNTHETIC_EVAL_GRIDS:
        ood_tag = " (OOD)" if gs not in SYNTHETIC_TRAINING_GRIDS else ""
        gen = DiffusionGenerator((gs,), device)
        pfn_acc = SolveAccumulator()
        jacobi_acc = SolveAccumulator()

        for _ in range(NUM_SYNTHETIC_MATRICES):
            batch = gen.generate_batch(1, 5)
            A = torch.sparse_coo_tensor(
                batch.indices, batch.values[0], (batch.n, batch.n)
            ).coalesce().to_sparse_csc()
            b = torch.randn(batch.n, dtype=torch.float64, device=device)

            try:
                precond = build_preconditioner_fn(model, A)
            except Exception:
                precond = None

            if precond is not None:
                _solve_with_preconditioner(solver, A, b, precond, pfn_acc)
            else:
                pfn_acc.record_failure()

            jacobi = Jacobi(A)
            _solve_with_preconditioner(solver, A, b, jacobi, jacobi_acc)

        all_pfn_iters.extend(pfn_acc.normalized_iters)
        all_jacobi_iters.extend(jacobi_acc.normalized_iters)
        all_pfn_conv.extend(pfn_acc.converged)

        details[f"{gs}x{gs}{ood_tag}"] = {
            "pfn_mean_norm_iter": pfn_acc.mean_normalized_iter(),
            "jacobi_mean_norm_iter": jacobi_acc.mean_normalized_iter(),
            "pfn_conv_rate": pfn_acc.convergence_rate(),
        }

    return {
        "all_pfn_iters": all_pfn_iters,
        "all_jacobi_iters": all_jacobi_iters,
        "all_pfn_conv": all_pfn_conv,
        "details": details,
    }


def _evaluate_suitesparse(model, build_preconditioner_fn, solver, device) -> dict:
    all_pfn_iters = []
    all_jacobi_iters = []
    all_pfn_conv = []
    details = {}

    for group, name in EVAL_MATRICES:
        try:
            A = load_suitesparse_matrix(name, device)
        except FileNotFoundError:
            print(f"  SKIP {name}: not downloaded")
            continue

        n = A.shape[0]
        pfn_acc = SolveAccumulator()
        jacobi_acc = SolveAccumulator()

        try:
            precond = build_preconditioner_fn(model, A)
        except Exception:
            precond = None

        for _ in range(NUM_RHS):
            b = torch.randn(n, dtype=torch.float64, device=device)

            if precond is not None:
                _solve_with_preconditioner(solver, A, b, precond, pfn_acc)
            else:
                pfn_acc.record_failure()

            jacobi = Jacobi(A)
            _solve_with_preconditioner(solver, A, b, jacobi, jacobi_acc)

        all_pfn_iters.extend(pfn_acc.normalized_iters)
        all_jacobi_iters.extend(jacobi_acc.normalized_iters)
        all_pfn_conv.extend(pfn_acc.converged)

        details[name] = {
            "pfn_mean_norm_iter": pfn_acc.mean_normalized_iter(),
            "jacobi_mean_norm_iter": jacobi_acc.mean_normalized_iter(),
            "pfn_conv_rate": pfn_acc.convergence_rate(),
            "n": n,
        }

    return {
        "all_pfn_iters": all_pfn_iters,
        "all_jacobi_iters": all_jacobi_iters,
        "all_pfn_conv": all_pfn_conv,
        "details": details,
    }


def print_results(results: dict, num_params: int, num_epochs: int,
                  training_seconds: float, total_seconds: float,
                  peak_vram_mb: float, num_domains: int,
                  best_loss: float) -> None:
    print()
    print("---")
    print(f"score:             {results['score']:.6f}")
    print(f"synthetic_score:   {results['synthetic_score']:.6f}")
    print(f"suitesparse_score: {results['suitesparse_score']:.6f}")
    print(f"synthetic_conv:    {results['synthetic_conv_pct']:.1f}%")
    print(f"suitesparse_conv:  {results['suitesparse_conv_pct']:.1f}%")
    print(f"synth_vs_jacobi:   {results['synthetic_score']:.4f} vs {results['synthetic_jacobi']:.4f}")
    print(f"ss_vs_jacobi:      {results['suitesparse_score']:.4f} vs {results['suitesparse_jacobi']:.4f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_params:        {num_params}")
    print(f"num_epochs:        {num_epochs}")
    print(f"best_loss:         {best_loss:.6e}")
    print(f"domains:           {num_domains}")

    ilu_ref = results.get("ilu_reference", {})
    amg_ref = results.get("amg_reference", {})

    print("\nSuiteSparse details:")
    for name, detail in results["suitesparse_details"].items():
        conv_pct = detail["pfn_conv_rate"] * 100
        pfn_iter = detail["pfn_mean_norm_iter"]
        jac_iter = detail["jacobi_mean_norm_iter"]
        ilu_iter = ilu_ref.get(name, {}).get("norm_iter", -1)
        amg_iter = amg_ref.get(name, {}).get("norm_iter", -1)
        n = detail["n"]
        status = "OK" if conv_pct > 50 else "FAIL"
        print(f"  {name:<12s} (n={n:>5d}): {status:<4s} pfn={pfn_iter:.3f} jac={jac_iter:.3f} ilu={ilu_iter:.3f} amg={amg_iter:.3f} conv={conv_pct:.0f}%")

    print("\nSynthetic details:")
    for grid, detail in results["synthetic_details"].items():
        conv_pct = detail["pfn_conv_rate"] * 100
        pfn_iter = detail["pfn_mean_norm_iter"]
        jac_iter = detail["jacobi_mean_norm_iter"]
        print(f"  {grid:<16s}: pfn={pfn_iter:.3f} jac={jac_iter:.3f} conv={conv_pct:.0f}%")
