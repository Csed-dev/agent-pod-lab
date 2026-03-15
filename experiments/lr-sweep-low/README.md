<experiment>
<description>
Lower learning rate test. LR=5e-4 vs baseline LR=3e-4.
Same architecture and domains as baseline.
</description>

<hypothesis>
Slower convergence during training but potentially better final score.
Lower LR may reduce overshooting and improve SuiteSparse performance.
</hypothesis>

<acceptance>
- score <= 0.10
</acceptance>

<context>
Part of LR sweep (phase-1). Compare directly against baseline.
If LR=5e-4 improves, try LR=2e-4 next.
</context>
</experiment>
