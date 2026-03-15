---
name: list-available-gpus
description: Show available RunPod GPU and CPU pod types with current prices
disable-model-invocation: true
allowed-tools: Bash
---

Show available GPUs sorted by price.

```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
print(mgr.available_gpus())
```
