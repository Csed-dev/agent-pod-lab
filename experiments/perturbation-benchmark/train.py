import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres, spilu, LinearOperator

warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROFILE = os.environ.get("PROFILE", "full")
DOMAINS_FILTER = os.environ.get("DOMAINS", "all")

if PROFILE == "smoke":
    SEED_BASE = 42
    N_SEEDS = 3
    EPSILON_VALUES = [0.0, 0.5, 2.0]
    GRIDS = [(3, 3)]
elif PROFILE == "small":
    SEED_BASE = 42
    N_SEEDS = 10
    EPSILON_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0]
    GRIDS = [(5, 5), (10, 10)]
else:
    SEED_BASE = 42
    N_SEEDS = 30
    EPSILON_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    GRIDS = [(5, 5), (10, 10), (50, 50), (100, 100)]

OMEGA = 0.9
K_MAX = 256
NEUMANN_TOL = 1e-10
GMRES_TOL = 1e-8
GMRES_MAXITER = 500


def build_diffusion(rows, cols):
    n = rows * cols
    diags = [4.0 * np.ones(n)]
    offsets = [0]
    off1 = -np.ones(n - 1)
    for i in range(1, rows):
        off1[i * cols - 1] = 0.0
    diags.extend([off1.copy(), off1.copy()])
    offsets.extend([-1, 1])
    off_col = -np.ones(n - cols)
    diags.extend([off_col.copy(), off_col.copy()])
    offsets.extend([-cols, cols])
    return sparse.diags(diags, offsets, shape=(n, n), format="csc")


def build_convection_diffusion(rows, cols, wind=1.0):
    A = build_diffusion(rows, cols)
    n = rows * cols
    data_lr = np.ones(n - 1) * (-wind / 2.0)
    for i in range(1, rows):
        data_lr[i * cols - 1] = 0.0
    data_ud = np.ones(n - cols) * (-wind / 2.0)
    C = sparse.diags(
        [data_lr, -data_lr, data_ud, -data_ud],
        [-1, 1, -cols, cols],
        shape=(n, n), format="csc",
    )
    return (A + C).tocsc()


def build_elasticity(rows, cols, nu=0.3):
    n_nodes = rows * cols
    n = 2 * n_nodes
    E = 1.0
    c1 = E / (1.0 - nu * nu)
    c2 = E / (2.0 * (1.0 + nu))
    c_cross = (c1 * nu + c2) / 4.0
    row_idx, col_idx, values = [], [], []

    def add(r, c, v):
        row_idx.append(r)
        col_idx.append(c)
        values.append(v)

    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            ux, uy = 2 * node, 2 * node + 1

            n_x_neighbors = (1 if j > 0 else 0) + (1 if j < cols - 1 else 0)
            n_y_neighbors = (1 if i > 0 else 0) + (1 if i < rows - 1 else 0)
            add(ux, ux, n_x_neighbors * c1 + n_y_neighbors * c2)
            add(uy, uy, n_y_neighbors * c1 + n_x_neighbors * c2)

            if j < cols - 1:
                nx_ux = 2 * (node + 1)
                add(ux, nx_ux, -c1)
                add(nx_ux, ux, -c1)
                nx_uy = nx_ux + 1
                add(uy, nx_uy, -c2)
                add(nx_uy, uy, -c2)

            if i < rows - 1:
                ny_uy = 2 * (node + cols) + 1
                add(uy, ny_uy, -c1)
                add(ny_uy, uy, -c1)
                ny_ux = ny_uy - 1
                add(ux, ny_ux, -c2)
                add(ny_ux, ux, -c2)

            if i < rows - 1 and j < cols - 1:
                diag_node = node + cols + 1
                diag_ux, diag_uy = 2 * diag_node, 2 * diag_node + 1
                add(ux, diag_uy, c_cross)
                add(diag_uy, ux, c_cross)
                add(uy, diag_ux, c_cross)
                add(diag_ux, uy, c_cross)

            if i < rows - 1 and j > 0:
                diag_node = node + cols - 1
                diag_ux, diag_uy = 2 * diag_node, 2 * diag_node + 1
                add(ux, diag_uy, -c_cross)
                add(diag_uy, ux, -c_cross)
                add(uy, diag_ux, -c_cross)
                add(diag_ux, uy, -c_cross)

    return sparse.csc_matrix((values, (row_idx, col_idx)), shape=(n, n))


DOMAIN_BUILDERS = {
    "diffusion": lambda r, c: build_diffusion(r, c),
    "convection_diffusion": lambda r, c: build_convection_diffusion(r, c, wind=1.0),
    "elasticity": lambda r, c: build_elasticity(r, c, nu=0.3),
}


def is_symmetric(A):
    diff = A - A.T
    return sparse.linalg.norm(diff, ord="fro") < 1e-12 * sparse.linalg.norm(A, ord="fro")


def add_noise(A, epsilon, rng, symmetric=None):
    if epsilon == 0.0:
        return A.copy()
    if symmetric is None:
        symmetric = is_symmetric(A)
    A_noisy = A.copy().astype(np.float64)
    rows_idx, cols_idx = A_noisy.nonzero()
    noise = rng.randn(len(rows_idx)) * epsilon
    for k in range(len(rows_idx)):
        i, j = rows_idx[k], cols_idx[k]
        A_noisy[i, j] += noise[k]
    if symmetric:
        A_noisy = (A_noisy + A_noisy.T) / 2.0
    return A_noisy.tocsc()


def weyl_bound(epsilon, A):
    rng_tmp = np.random.RandomState(9999)
    norms = []
    for _ in range(20):
        N = add_noise(A, 1.0, rng_tmp) - A
        norms.append(sparse.linalg.norm(N, ord="fro"))
    return epsilon * np.mean(norms)


def compute_eigenmetrics(A, omega):
    A_dense = A.toarray()
    sym = is_symmetric(sparse.csc_matrix(A_dense))

    if sym:
        eigenvalues = np.linalg.eigvalsh(A_dense)
    else:
        eigenvalues = np.linalg.eigvals(A_dense)

    real_parts = np.real(eigenvalues)
    abs_vals = np.abs(eigenvalues)
    lmin = float(np.min(real_parts))
    lmax = float(np.max(real_parts))
    lmin_abs = float(np.min(abs_vals))
    cond = float(np.max(abs_vals)) / lmin_abs if lmin_abs > 1e-15 else float("inf")

    D_inv = sparse.diags(1.0 / A.diagonal())
    J = sparse.eye(A.shape[0]) - omega * (D_inv @ A)
    J_dense = J.toarray()

    if sym:
        j_eigs = np.linalg.eigvalsh(J_dense)
    else:
        j_eigs = np.linalg.eigvals(J_dense)
    rho = float(np.max(np.abs(j_eigs)))

    pos_def = bool(np.all(real_parts > 0)) if sym else False
    diag_dom = True
    for i in range(A.shape[0]):
        row = np.abs(A_dense[i, :])
        if row[i] < np.sum(row) - row[i]:
            diag_dom = False
            break

    return {
        "lambda_min_re": lmin,
        "lambda_max_re": lmax,
        "cond_number": cond,
        "spectral_radius": rho,
        "neumann_converges": 1.0 if rho < 1.0 else 0.0,
        "positive_definite": 1.0 if pos_def else 0.0,
        "diag_dominant": 1.0 if diag_dom else 0.0,
        "symmetric": 1.0 if sym else 0.0,
    }


def solve_no_precond(A, b):
    iters = [0]
    def cb(_): iters[0] += 1
    t0 = time.monotonic()
    _, info = gmres(A, b, atol=GMRES_TOL, rtol=GMRES_TOL, maxiter=GMRES_MAXITER, callback=cb)
    elapsed = time.monotonic() - t0
    return {"iters": iters[0] if info == 0 else GMRES_MAXITER, "time": elapsed, "failed": 0}


def solve_jacobi(A, b, omega):
    D_inv = 1.0 / A.diagonal()
    M = LinearOperator(A.shape, matvec=lambda x: omega * D_inv * x)
    iters = [0]
    def cb(_): iters[0] += 1
    t0 = time.monotonic()
    _, info = gmres(A, b, M=M, atol=GMRES_TOL, rtol=GMRES_TOL, maxiter=GMRES_MAXITER, callback=cb)
    elapsed = time.monotonic() - t0
    return {"iters": iters[0] if info == 0 else GMRES_MAXITER, "time": elapsed, "failed": 0}


def solve_neumann(A, b, omega, k_max, tol):
    n = A.shape[0]
    D_inv = 1.0 / A.diagonal()
    D_inv_A = sparse.diags(D_inv) @ A

    def neumann_apply(r):
        power = omega * D_inv * r
        result = power.copy()
        for _ in range(1, k_max):
            power = power - omega * D_inv_A @ power
            result += power
            if np.linalg.norm(power) < tol * np.linalg.norm(result):
                break
        return result

    M = LinearOperator((n, n), matvec=neumann_apply)
    iters = [0]
    def cb(_): iters[0] += 1
    t0 = time.monotonic()
    _, info = gmres(A, b, M=M, atol=GMRES_TOL, rtol=GMRES_TOL, maxiter=GMRES_MAXITER, callback=cb)
    elapsed = time.monotonic() - t0
    return {"iters": iters[0] if info == 0 else GMRES_MAXITER, "time": elapsed, "failed": 0}


def solve_ilu(A, b):
    t0 = time.monotonic()
    try:
        ilu = spilu(A.tocsc())
    except Exception:
        return {"iters": GMRES_MAXITER, "time": time.monotonic() - t0, "failed": 1}
    M = LinearOperator(A.shape, matvec=ilu.solve)
    iters = [0]
    def cb(_): iters[0] += 1
    _, info = gmres(A, b, M=M, atol=GMRES_TOL, rtol=GMRES_TOL, maxiter=GMRES_MAXITER, callback=cb)
    elapsed = time.monotonic() - t0
    return {"iters": iters[0] if info == 0 else GMRES_MAXITER, "time": elapsed, "failed": 0}


def solve_amg(A, b):
    t0 = time.monotonic()
    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A.tocsr())
        M = ml.aspreconditioner()
    except Exception:
        return {"iters": GMRES_MAXITER, "time": time.monotonic() - t0, "failed": 1}
    iters = [0]
    def cb(_): iters[0] += 1
    _, info = gmres(A, b, M=M, atol=GMRES_TOL, rtol=GMRES_TOL, maxiter=GMRES_MAXITER, callback=cb)
    elapsed = time.monotonic() - t0
    return {"iters": iters[0] if info == 0 else GMRES_MAXITER, "time": elapsed, "failed": 0}


SOLVERS = {
    "none": lambda A, b: solve_no_precond(A, b),
    "jacobi": lambda A, b: solve_jacobi(A, b, OMEGA),
    "neumann": lambda A, b: solve_neumann(A, b, OMEGA, K_MAX, NEUMANN_TOL),
    "ilu": lambda A, b: solve_ilu(A, b),
    "amg": lambda A, b: solve_amg(A, b),
}


def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def save_intermediate(all_results, results_dir):
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    if DOMAINS_FILTER == "all":
        domains = list(DOMAIN_BUILDERS.keys())
    else:
        domains = [d.strip() for d in DOMAINS_FILTER.split(",")]

    total_configs = len(domains) * len(GRIDS) * len(EPSILON_VALUES) * N_SEEDS * len(SOLVERS)
    flush(f"PROFILE={PROFILE} DOMAINS={domains} GRIDS={GRIDS} EPS={EPSILON_VALUES} SEEDS={N_SEEDS} TOTAL={total_configs}")

    t_start = time.monotonic()
    all_results = []
    results_dir = Path("results_perturbation")
    completed = 0

    for domain_name in domains:
        builder = DOMAIN_BUILDERS[domain_name]

        for grid_rows, grid_cols in GRIDS:
            A_base = builder(grid_rows, grid_cols)
            n = A_base.shape[0]
            grid_label = f"{grid_rows}x{grid_cols}"
            sym = is_symmetric(A_base)

            if sym:
                base_eigenvalues = np.linalg.eigvalsh(A_base.toarray())
            else:
                base_eigenvalues = np.sort(np.real(np.linalg.eigvals(A_base.toarray())))
            wb = {eps: weyl_bound(eps, A_base) for eps in EPSILON_VALUES}

            b = np.zeros(n)
            b_len = min(n, grid_cols if domain_name != "elasticity" else 2 * grid_cols)
            b[:b_len] = 100.0

            flush(f"\n=== {domain_name} | Grid {grid_label} (n={n}, sym={sym}) ===")

            for eps in EPSILON_VALUES:
                for seed_idx in range(N_SEEDS):
                    seed = SEED_BASE + seed_idx
                    rng = np.random.RandomState(seed)
                    A_noisy = add_noise(A_base, eps, rng, symmetric=sym)

                    has_zero_diag = np.any(np.abs(A_noisy.diagonal()) < 1e-15)
                    if has_zero_diag:
                        for solver_name in SOLVERS:
                            all_results.append({
                                "domain": domain_name, "grid": grid_label, "n": n,
                                "epsilon": eps, "seed": seed, "solver": solver_name,
                                "iters": GMRES_MAXITER, "time": 0.0, "failed": 1,
                                "lambda_min_re": float("nan"), "lambda_max_re": float("nan"),
                                "cond_number": float("inf"), "spectral_radius": float("nan"),
                                "neumann_converges": 0.0, "positive_definite": 0.0,
                                "diag_dominant": 0.0, "symmetric": float(sym),
                                "weyl_bound": wb[eps], "empirical_shift": float("nan"),
                            })
                        completed += len(SOLVERS)
                        continue

                    metrics = compute_eigenmetrics(A_noisy, OMEGA)
                    empirical_shift = max(
                        abs(metrics["lambda_min_re"] - float(base_eigenvalues[0])),
                        abs(metrics["lambda_max_re"] - float(base_eigenvalues[-1])),
                    )

                    for solver_name, solver_fn in SOLVERS.items():
                        result = solver_fn(A_noisy, b)
                        all_results.append({
                            "domain": domain_name, "grid": grid_label, "n": n,
                            "epsilon": eps, "seed": seed, "solver": solver_name,
                            "iters": result["iters"], "time": result["time"],
                            "failed": result["failed"],
                            **metrics,
                            "weyl_bound": wb[eps],
                            "empirical_shift": empirical_shift,
                        })
                        completed += 1

                done_pct = completed / total_configs * 100
                flush(f"  eps={eps:.2f}: {completed}/{total_configs} ({done_pct:.0f}%)")
                save_intermediate(all_results, results_dir)

    t_total = time.monotonic() - t_start
    save_intermediate(all_results, results_dir)

    flush(f"\n{'='*70}")
    flush(f"SUMMARY")
    flush(f"{'='*70}")

    for domain_name in domains:
        for grid_rows, grid_cols in GRIDS:
            A_base = DOMAIN_BUILDERS[domain_name](grid_rows, grid_cols)
            n = A_base.shape[0]
            grid_label = f"{grid_rows}x{grid_cols}"
            flush(f"\n--- {domain_name} | {grid_label} (n={n}) ---")
            flush(f"{'eps':>6} | {'solver':>8} | {'iters':>6} | {'time_ms':>8} | {'fail%':>5} | {'rho':>6} | {'kappa':>8} | {'weyl':>6} | {'emp_shift':>9}")
            for eps in EPSILON_VALUES:
                for solver_name in SOLVERS:
                    rows = [r for r in all_results
                            if r["domain"] == domain_name and r["grid"] == grid_label
                            and r["epsilon"] == eps and r["solver"] == solver_name]
                    if not rows:
                        continue
                    avg_iters = np.mean([r["iters"] for r in rows])
                    avg_time = np.mean([r["time"] for r in rows]) * 1000
                    fail_rate = np.mean([r["failed"] for r in rows]) * 100
                    avg_rho = np.nanmean([r["spectral_radius"] for r in rows])
                    avg_kappa = np.nanmean([r["cond_number"] for r in rows])
                    avg_weyl = rows[0]["weyl_bound"]
                    avg_shift = np.nanmean([r["empirical_shift"] for r in rows])
                    flush(f"{eps:6.2f} | {solver_name:>8} | {avg_iters:6.1f} | {avg_time:8.1f} | {fail_rate:5.1f} | {avg_rho:6.3f} | {avg_kappa:8.1f} | {avg_weyl:6.3f} | {avg_shift:9.3f}")

    flush(f"\ntotal_time: {t_total:.1f}")
    flush(f"num_completed_configs: {len(all_results)}")
    flush(f"num_domains: {len(domains)}")
    flush(f"num_grids: {len(GRIDS)}")
    flush(f"num_epsilon_values: {len(EPSILON_VALUES)}")
    flush(f"num_seeds: {N_SEEDS}")
    flush(f"num_solvers: {len(SOLVERS)}")


if __name__ == "__main__":
    main()
