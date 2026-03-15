---
name: add-experiment
description: Create a new experiment directory with train.py and README.md
disable-model-invocation: true
allowed-tools: Bash, Read, Write
argument-hint: "<name>"
---

Create experiment "$ARGUMENTS":

1. Create `experiments/$ARGUMENTS/train.py` — copy from existing experiment, modify hyperparameters. Import shared code from `lib/`. All values explicit, no hidden defaults.

2. Create `experiments/$ARGUMENTS/README.md` with:
```xml
<experiment>
<description>What this experiment does.</description>
<hypothesis>Expected outcome and reasoning.</hypothesis>
<acceptance>
- score <= X.XX
- suitesparse_conv >= XX.X
</acceptance>
<context>Why this matters. What it builds on.</context>
</experiment>
```

3. Register:
```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
mgr.add_experiment("$ARGUMENTS", script="experiments/$ARGUMENTS/train.py")
```
