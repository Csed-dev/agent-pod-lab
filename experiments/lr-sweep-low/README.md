<experiment>
<description>LR=5e-4 instead of baseline 3e-4. Higher learning rate for faster convergence.</description>
<hypothesis>With only ~54 epochs at K=1024, higher LR might find better coefficients faster. Previous Run 27 (K=6, LR=5e-4) got 0.498 vs 0.482 — worse. But K=1024 Neumann init is much stronger.</hypothesis>
<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>
<context>Simple hyperparameter sweep. Tests if the 3e-4 default is optimal for K=1024.</context>
</experiment>
