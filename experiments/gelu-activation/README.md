<experiment>
<description>GELU activation instead of ReLU in GNN (encoder + MPNN + head).</description>
<hypothesis>GELU has smooth gradients avoiding dead neurons. With 64-dim embed, ReLU can kill entire channels. Expect marginal improvement (0-2%) since GNN contributes little.</hypothesis>
<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>
<context>Modern standard (BERT, GPT). Trivial change. Tests if activation choice affects the GNN's (small) contribution.</context>
</experiment>
