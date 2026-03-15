import time
from pathlib import Path

from fastapi import APIRouter, Request

from api.dependencies import get_manager

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_start_time = time.monotonic()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "uptime_s": round(time.monotonic() - _start_time)}


@router.get("/version")
def version(request: Request) -> dict:
    return {"version": request.app.version, "name": "agent-pod-lab"}


@router.get("/deep-health")
def deep_health() -> dict:
    checks: dict[str, str] = {}

    experiments_path = Path("experiments.yaml")
    checks["experiments_yaml"] = "ok" if experiments_path.exists() else "missing"

    config_path = Path("scheduler_config.yaml")
    checks["scheduler_config"] = "ok" if config_path.exists() else "missing"

    state_path = Path("scheduler_state.json")
    checks["scheduler_state"] = "ok" if state_path.exists() else "missing"

    results_dir = Path("results")
    checks["results_dir"] = "ok" if results_dir.is_dir() else "missing"

    try:
        mgr = get_manager()
        mgr.experiments_status()
        checks["orchestrator"] = "ok"
    except Exception as e:
        checks["orchestrator"] = str(e)[:200]

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
