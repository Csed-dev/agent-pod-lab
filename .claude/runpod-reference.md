# RunPod Reference

## Compute Adapter

```python
from orchestrator.adapters.runpod import RunPodCompute
compute = RunPodCompute()

# GPU instance
instance_id = compute.create_instance("exp-001", gpu_type="NVIDIA RTX A5000", image="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", disk_gb=20)

# Custom image/disk
instance_id = compute.create_instance("exp", gpu_type="NVIDIA RTX A5000", image="custom:latest", disk_gb=50)
```

## Instance Lifecycle

```python
conn = compute.wait_until_ready(instance_id)        # waits up to 2 min, auto-terminates on timeout
output = compute.run_command(conn, "nvidia-smi")     # run command via SSH
compute.upload_file(conn, "local.py", "/workspace/remote.py")  # copy file to instance
compute.terminate_instance(instance_id)              # destroy instance
```

## Available Resources

```python
# GPU list with live prices
gpus = compute.available_gpus(min_memory_gb=0)
for g in gpus:
    print(f"{g['id']:45s} {g['memory_gb']:>4}GB  ${g['price_per_hr']:.2f}/hr")

# Active instances
for inst in compute.list_instances():
    print(f"{inst.instance_id} | {inst.name} | {inst.status} | {inst.gpu_type} | ${inst.cost_per_hr}/hr")
```

## GPU Types

| GPU | VRAM | Price |
|-----|------|-------|
| NVIDIA RTX A5000 | 24GB | ~$0.16/hr |
| NVIDIA GeForce RTX 3090 | 24GB | ~$0.22/hr |
| NVIDIA RTX A6000 | 48GB | ~$0.33/hr |
| NVIDIA GeForce RTX 4090 | 24GB | ~$0.34/hr |
| NVIDIA A100 80GB PCIe | 80GB | ~$1.19/hr |

## CPU Pod Types

| ID | Tier | Use Case |
|----|------|----------|
| cpu3c | CPU3 Compute-Optimized | CPU-heavy tasks |
| cpu3g | CPU3 General Purpose | Default CPU |
| cpu3m | CPU3 Memory-Optimized | Large datasets |
| cpu5c | CPU5 Compute-Optimized | Fast CPU |
| cpu5g | CPU5 General Purpose | Fast general |
| cpu5m | CPU5 Memory-Optimized | Fast + large RAM |

## Error Handling

- Instance not ready in 2 min → auto-terminated, raises TimeoutError
- GPU unavailable → use `compute.available_gpus()` to find alternatives
- SSH fails → instance may have died, terminate and retry
- ALWAYS terminate instances in finally blocks
