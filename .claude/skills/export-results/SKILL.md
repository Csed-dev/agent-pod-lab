---
name: export-results
description: Export all experiment results as CSV for analysis
disable-model-invocation: true
allowed-tools: Bash, Read
argument-hint: "[output-file.csv]"
---

Export all experiment results to CSV.

```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
output = "$ARGUMENTS" if "$ARGUMENTS" else "results.csv"
print(mgr.export_csv(output))
```
