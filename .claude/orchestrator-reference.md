# Orchestrator Reference

## Setup

```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
```

## Experiment Flow

```python
ready = mgr.get_ready()
result = mgr.start("baseline")          # returns ExperimentStartResult
print(result.sub_agent_prompt())         # complete prompt for sub-agent
mgr.finish("baseline", exit_code=0, output=stdout)
print(mgr.result("baseline"))
```

## All Methods

| Method | Returns | Purpose |
|---|---|---|
| `mgr.experiments_status()` | str | Overview of all experiments |
| `mgr.get_ready()` | list[str] | Experiments with all deps completed |
| `mgr.start(name, gpu_type=)` | ExperimentStartResult | Create pod, setup, SCP script. Has .ssh_command, .run_id, .sub_agent_prompt() |
| `mgr.finish(name, exit_code, output)` | None | Save results, terminate pod |
| `mgr.cancel(name)` | str | Cancel running experiment |
| `mgr.result(name)` | str | Metrics + interpretation + log tail |
| `mgr.meets_acceptance(name)` | bool/None | Check acceptance from README.md |
| `mgr.reset(name)` | str | Reset to PENDING |
| `mgr.add_experiment(name, script, deps=, tags=)` | str | Add + persist to YAML |
| `mgr.all_results(tags=, status=)` | list[dict] | Batch query |
| `mgr.export_csv(path)` | str | Export as CSV |
| `mgr.available_gpus(min_memory_gb=)` | str | GPUs with prices |
| `mgr.available_experiments_detail()` | str | Full experiment info |
| `mgr.cleanup_orphaned_pods()` | str | Terminate orphaned pods |
| `mgr.write_interpretation(name, text)` | None | Save interpretation.md |

## Results per Run

`results/{name}-{NNNN}/`: config.yaml, log.txt, metrics.json, status.json, interpretation.md, train.py

## Error Messages

All errors include actionable info:
- `ExperimentNotFoundError` — lists available, suggests "did you mean?"
- `ExperimentNotReadyError` — shows pending deps
- `GpuUnavailableError` — lists cheapest available GPUs
- `PodPoolFullError` — shows running experiments
- `CyclicDependencyError` — shows the cycle
