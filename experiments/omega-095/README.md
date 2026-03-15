<experiment>
<description>Omega=0.95. Previously broke thermal at K=256. Does K=1024 rescue it?</description>
<hypothesis>At K=256, thermal FAILS at omega=0.95. K=1024 might provide enough terms for thermal to converge even at higher omega, improving other matrices.</hypothesis>
<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>
<context>Critical test: is the omega=0.9 limit a K=256 artifact or fundamental?</context>
</experiment>
