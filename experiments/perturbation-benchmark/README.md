<experiment>
<description>
Monte-Carlo perturbation analysis of sparse matrices from 5-point stencil discretization.
Compares 5 solvers (None, Jacobi, Neumann K=256, ILU, AMG) across 8 noise levels
on 4 grid sizes (5x5, 10x10, 50x50, 100x100) with 30 seeds each.
Measures: eigenvalues, condition number, spectral radius, GMRES iterations,
construction failures, wall-clock time. Includes analytical Weyl bounds.
</description>

<hypothesis>
Neumann preconditioner is robust until epsilon ~0.25 (25% relative perturbation).
ILU should be more effective per iteration but prone to construction failures at high noise.
AMG expected to degrade gracefully. No-preconditioner baseline establishes that
preconditioning helps. Analytical Weyl bounds should match empirical eigenvalue shifts.
</hypothesis>

<acceptance>
- total_time <= 14400
- num_completed_configs >= 1200
</acceptance>

<context>
Perturbation analysis for bachelor thesis on learned preconditioners (MatrixPFN).
Results show robustness of Neumann series preconditioner vs classical alternatives.
Statistical analysis via ml-experiment-stats (Holm-Bonferroni, Cliff's delta, Friedman).
</context>
</experiment>
