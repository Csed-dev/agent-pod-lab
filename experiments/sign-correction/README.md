<experiment>
<description>
Sign correction for negative-diagonal matrices.
If majority of diagonal entries are negative (like saylr4), use |D| instead of D
and flip preconditioner output sign. This makes the Neumann series converge
for matrices where rho(J) = 1 due to negative diagonal.
</description>

<hypothesis>
saylr4 has all-negative diagonal, causing rho(J)=1.000 for any omega.
At K=1024 without sign correction, saylr4 converges at pfn=0.447 (through K brute force).
With sign correction at K=1024, expect saylr4 pfn to improve significantly (<0.1)
because the Neumann series properly converges instead of oscillating.
Other matrices should be unaffected (positive diagonal).
</hypothesis>

<acceptance>
- score <= 0.05
- suitesparse_conv >= 95.0
</acceptance>

<context>
saylr4 eigenvalue analysis showed: ALL 3564 diagonal entries are negative.
rho(J_omega) = 1.000000 for ALL omega. Sign correction transforms the problem
so the Neumann series converges. This could also help saylr3, sherman2,
and other negative-diagonal matrices in the broader benchmark.
</context>
</experiment>
