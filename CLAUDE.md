# Agent Pod Lab

Autonomous ML experiment system. Manages experiment queues with DAG dependencies on RunPod GPU pods.

## On Every Session Start (including after context clear)

Your context may be compacted at any time. All state is on disk. Always start by recovering:

```python
import sys; sys.path.insert(0, ".")
from orchestrator import ExperimentManager
mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")
print(mgr.cleanup_orphaned_pods())    # terminate orphaned pods first
print(mgr.experiments_status())     # see where things stand
```

Then check `results/` for any completed experiments that haven't been reviewed yet. Read their `interpretation.md` and `metrics.json`. Continue from where you left off — the DAG state in `scheduler_state.json` is the source of truth.

## Structure

```
experiments/{name}/train.py   ← Code + hyperparameters (one per experiment)
experiments/{name}/README.md  ← Description, hypothesis, acceptance (XML tags)
experiments.yaml              ← Orchestration only (name, script, deps, tags)
lib/                          ← Shared modules (architectures, training, eval)
orchestrator/                 ← Pod management, state, cloud storage
results/{run_id}/             ← Outputs per run (auto-suffixed: baseline-0001)
scheduler_state.json          ← DAG state (survives restarts and context clears)
```

## Experiment Lifecycle

```python
result = mgr.start("name")                 # returns ExperimentStartResult
prompt = result.sub_agent_prompt()          # complete prompt for sub-agent
# launch experiment-runner sub-agent with prompt
mgr.finish("name", exit_code, output)      # saves results, terminates pod
```

Use `/check-experiments-status`, `/run-experiment`, `/list-available-gpus`, `/export-results`, `/add-experiment`.

## Rules

- Only Father calls `mgr.start()` and `mgr.finish()` — sub-agents just run SSH commands
- ALWAYS `mgr.cleanup_orphaned_pods()` at session start — orphaned pods burn money
- Cheapest GPU that fits. `mgr.available_gpus()` for live prices. Only upgrade if VRAM justifies it
- Each experiment = own `train.py` with explicit hyperparameters. No env vars
- README.md per experiment uses XML tags: `<description>`, `<hypothesis>`, `<acceptance>`, `<context>`
- Do not stop work because context is getting large — state is on disk, sessions can be cleared safely

## Father Agent Instructions

Read `prompts/father-prompt.md` for the full orchestration protocol. Key loop:
1. `mgr.cleanup_orphaned_pods()` → `mgr.experiments_status()` → review completed results
2. `mgr.get_ready()` → start experiments → launch sub-agents
3. When sub-agents finish → `mgr.finish()` → evaluate → design next batch

## Reference

- RunPod API patterns: `.claude/runpod-reference.md`
- Orchestrator API details: `.claude/orchestrator-reference.md`
- VPS connection: `.claude/vps-reference.md`
- Cloud storage env vars: `CF_ACCOUNT_ID`, `CF_API_TOKEN`, `CF_D1_DATABASE_ID`, `CF_R2_ACCESS_KEY`, `CF_R2_SECRET_KEY`
