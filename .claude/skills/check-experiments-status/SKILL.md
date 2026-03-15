---
name: check-experiments-status
description: Check experiment status, cleanup orphaned pods, show ready experiments
disable-model-invocation: true
allowed-tools: Bash, Read
argument-hint: "[experiment-name]"
---

Check experiment status. If $ARGUMENTS provided, show details for that experiment.

Run this Python script:
```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
print(mgr.cleanup_orphaned_pods())
print(mgr.experiments_status())
ready = mgr.get_ready()
print(f"\nReady: {ready}")
```

If $ARGUMENTS is not empty, also run:
```python
print(mgr.result("$ARGUMENTS"))
```
