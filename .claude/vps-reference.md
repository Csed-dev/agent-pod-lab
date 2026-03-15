# VPS Reference

## Connection

```bash
ssh root@<VPS_IP>
```

## Project Location

```
/root/agent-pod-lab/
```

## Pull Latest Changes

```bash
ssh root@<VPS_IP> "cd /root/agent-pod-lab && git pull"
```

## Run Orchestrator Code

```bash
cd /root/agent-pod-lab && /root/agent-pod-lab-env/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from orchestrator import PodManager
pm = PodManager()
for pod in pm.list_pods():
    print(f'{pod.pod_id} | {pod.name} | {pod.status} | {pod.gpu_type}')
"
```

## Start Claude Code on VPS

```bash
ssh root@<VPS_IP>
cd /root/agent-pod-lab
claude
```

## Environment Variables

Required in `/root/agent-pod-lab/.env`:
```
RUNPOD_API_KEY=<your-key>
SMTP_USER=<your-email>
SMTP_PASSWORD=<your-app-password>
```

## Git Setup

- SSH key on VPS for GitHub access
- Repo cloned at `/root/agent-pod-lab/`
