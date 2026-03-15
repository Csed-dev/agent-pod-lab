<experiment>
<description>
GNN Ablation: Fixed Neumann coefficients c_k=1 for all nodes and all k.
No GNN training, no learned parameters. Pure classical Neumann series.
The model is built but coefficients are overridden to 1.0 at eval time.
</description>

<hypothesis>
Previous ablation (K=256) showed fixed c_k=1 matches or beats GNN (SS mean 0.1273 vs 0.1283).
At K=1024, expect similar or better score than baseline (~0.048) because the GNN adds noise.
This proves the ML component is unnecessary.
</hypothesis>

<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>

<context>
This is the critical thesis experiment. If no-gnn matches baseline, the conclusion is:
"A classical weighted-Jacobi Neumann series preconditioner achieves state-of-the-art results.
The GNN provides zero measurable benefit."
Builds on: Finding 26 from FINDINGS.md (GNN is unnecessary at K=256).
</context>
</experiment>
