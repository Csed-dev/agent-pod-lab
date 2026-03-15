<experiment>
<description>K=512 with sign correction. Tests if sign-fix makes K=512 sufficient for saylr4.</description>
<hypothesis>Without sign fix: K=512 gives saylr4 20% conv. With sign fix: the Neumann series properly converges, so K=512 should give saylr4 ~100% conv. Score should be between K=256 (0.095) and K=1024 (0.048).</hypothesis>
<acceptance>
- score <= 0.08
- suitesparse_conv >= 95.0
</acceptance>
<context>If sign-fix + K=512 achieves 11/11, it's 2x cheaper than K=1024 for deployment.</context>
</experiment>
