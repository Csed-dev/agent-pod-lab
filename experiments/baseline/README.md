<experiment>
<description>
Baseline Neumann polynomial preconditioner.
Architecture: PolyMPNN with K=1024, 2 layers (64/128), LR=3e-4, omega=0.9.
8 training domains, equal weights.
</description>

<hypothesis>
Expect ~0.048 composite score with 11/11 SuiteSparse convergence.
Sets reference point for all subsequent experiments.
Based on previous run achieving score=0.048 with 63K params, 54 epochs in 300s.
</hypothesis>

<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>

<context>
This is the proven best configuration from manual experimentation.
All subsequent experiments should compare against this baseline.
</context>
</experiment>
