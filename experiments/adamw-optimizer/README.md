<experiment>
<description>AdamW optimizer instead of Adam. Correct decoupled weight decay.</description>
<hypothesis>Adam with weight_decay is L2 regularization, not true weight decay. AdamW implements it correctly. Expect marginal improvement since training is short (~54 epochs).</hypothesis>
<acceptance>
- score <= 0.10
- suitesparse_conv >= 80.0
</acceptance>
<context>One-line fix. Standard best practice since 2018 (Loshchilov & Hutter).</context>
</experiment>
