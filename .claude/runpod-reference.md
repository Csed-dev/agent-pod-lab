# RunPod Reference

## Pod Creation

```python
from orchestrator import PodManager
pm = PodManager()

# GPU pod
pod_id = pm.create_pod("exp-001", gpu_type="NVIDIA RTX A5000")

# CPU pod (no GPU)
pod_id = pm.create_pod("cpu-task", gpu_type="cpu3g", gpu_count=0)

# Custom image/disk
pod_id = pm.create_pod("exp", gpu_type="NVIDIA RTX A5000", image="custom:latest", container_disk_gb=50)
```

## Pod Lifecycle

```python
conn = pm.wait_until_ready(pod_id)        # waits up to 2 min, auto-terminates on timeout
output = pm.ssh_run(conn, "nvidia-smi")   # run command via SSH
pm.scp_to_pod(conn, "local.py", "/workspace/remote.py")  # copy file to pod
pm.terminate_pod(pod_id)                  # destroy pod
pm.stop_pod(pod_id)                       # stop (preserves state)
```

## Available Resources

```python
# GPU list with live prices
gpus = pm.get_available_gpus(min_memory_gb=0)
for g in gpus:
    print(f"{g['id']:45s} {g['memory_gb']:>4}GB  ${g['price_per_hr']:.2f}/hr")

# Active pods
for pod in pm.list_pods():
    print(f"{pod.pod_id} | {pod.name} | {pod.status} | {pod.gpu_type} | ${pod.cost_per_hr}/hr")
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

```python
pod_id = pm.create_pod("data-processing", gpu_type="cpu3g", gpu_count=0)
```

## Error Handling

- Pod not ready in 2 min → auto-terminated, raises TimeoutError
- GPU unavailable → use `pm.get_available_gpus()` to find alternatives
- SSH fails → pod may have died, terminate and retry
- ALWAYS terminate pods in finally blocks
