<experiment>
<description>K=256 with omega=0.9. The efficient config (10/11 SS, score ~0.095).</description>
<hypothesis>Previous Run 62: score 0.0948, 10/11 conv, ~150 epochs. Establishes K=256 reference in agent-pod-lab framework.</hypothesis>
<acceptance>
- score <= 0.12
- suitesparse_conv >= 85.0
</acceptance>
<context>K=256 is 4x cheaper per FGMRES step than K=1024. Important for deployment.</context>
</experiment>
