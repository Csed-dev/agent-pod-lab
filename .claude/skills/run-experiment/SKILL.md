---
name: run-experiment
description: Run a single experiment on RunPod GPU pod
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Agent
argument-hint: "<experiment-name>"
---

Run experiment "$ARGUMENTS" through full lifecycle.

1. Read `experiments/$ARGUMENTS/README.md` for context
2. Start: `mgr.start("$ARGUMENTS")` — creates pod, SCPs train.py, returns SSH command
3. If GPU unavailable: `mgr.available_gpus()` → try cheapest alternative, up to 3 attempts
4. Launch experiment-runner sub-agent with the SSH command
5. When sub-agent finishes: `mgr.finish("$ARGUMENTS", exit_code, output)`
6. Evaluate: `mgr.result("$ARGUMENTS")` and `mgr.meets_acceptance("$ARGUMENTS")`

Setup:
```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
```
