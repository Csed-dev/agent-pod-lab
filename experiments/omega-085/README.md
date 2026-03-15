<experiment>
<description>Omega=0.85 sweep point. Lower omega for potentially better thermal convergence.</description>
<hypothesis>omega=0.85 gave 0.0955 in previous experiments (Run 61). Expect similar with K=1024.</hypothesis>
<acceptance>
- score <= 0.10
- suitesparse_conv >= 90.0
</acceptance>
<context>Omega sweep: 0.9 is optimal at K=256. At K=1024 the optimum might shift.</context>
</experiment>
