from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from graphlib import TopologicalSorter, CycleError
from pathlib import Path

import yaml

from orchestrator.errors import (
    CyclicDependencyError,
    DependencyNotFoundError,
)


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ExperimentSpec:
    name: str
    command: str
    script: str = ""
    description: str = ""
    hypothesis: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    gpu_type: str | None = None
    timeout: int | None = None
    acceptance: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentState:
    name: str
    run_id: str = ""
    status: ExperimentStatus = ExperimentStatus.PENDING
    instance_id: str | None = None
    gpu_type_used: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    cost_usd: float | None = None
    gpu_utilization_pct: float | None = None


@dataclass
class SchedulerConfig:
    max_pods: int
    gpu_type: str
    image: str
    container_disk_gb: int
    repo_url: str
    workspace_dir: str
    setup_command: str
    sync_command: str
    experiment_timeout: int
    setup_timeout: int
    pod_ready_timeout: int
    log_dir: str
    results_dir: str
    poll_interval: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> SchedulerConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            max_pods=data["max_pods"],
            gpu_type=data["gpu_type"],
            image=data["image"],
            container_disk_gb=data.get("container_disk_gb", 20),
            repo_url=data["repo_url"],
            workspace_dir=data["workspace_dir"],
            setup_command=data["setup_command"],
            sync_command=data["sync_command"],
            experiment_timeout=data.get("experiment_timeout", 900),
            setup_timeout=data.get("setup_timeout", 600),
            pod_ready_timeout=data.get("pod_ready_timeout", 120),
            log_dir=data.get("log_dir", "logs"),
            results_dir=data.get("results_dir", "results"),
            poll_interval=data.get("poll_interval", 10),
        )


def load_experiments(path: str | Path, default_command: str = "cd {workspace} && PYTHONPATH={workspace} uv run python {script}") -> list[ExperimentSpec]:
    with open(path) as f:
        data = yaml.safe_load(f)

    experiment_names = [e["name"] for e in data.get("experiments", [])]

    experiments = []
    for exp_data in data.get("experiments", []):
        deps = exp_data.get("dependencies", [])
        for dep in deps:
            if dep not in experiment_names:
                raise DependencyNotFoundError(exp_data["name"], dep, experiment_names)

        script = exp_data.get("script", "")
        command = exp_data.get("command", default_command)
        if script:
            command = command.replace("{script}", script)

        readme = _load_experiment_readme(script)

        experiments.append(ExperimentSpec(
            name=exp_data["name"],
            command=command,
            script=script,
            description=readme.get("description", ""),
            hypothesis=readme.get("hypothesis", ""),
            tags=exp_data.get("tags", []),
            dependencies=deps,
            gpu_type=exp_data.get("gpu_type"),
            timeout=exp_data.get("timeout"),
            acceptance=readme.get("acceptance", {}),
        ))

    _validate_dag(experiments)
    return experiments


def _load_experiment_readme(script_path: str) -> dict:
    if not script_path:
        return {}
    readme_path = Path(script_path).parent / "README.md"
    if not readme_path.exists():
        return {}

    content = readme_path.read_text()
    result = {}

    for tag in ("description", "hypothesis", "acceptance", "context"):
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", content, re.DOTALL)
        if match:
            result[tag] = match.group(1).strip()

    if "acceptance" in result:
        acceptance = {}
        for line in result["acceptance"].split("\n"):
            line = line.strip().lstrip("- ")
            if not line:
                continue
            if "<=" in line:
                key, val = line.split("<=")
                acceptance[key.strip() + "_max"] = float(val.strip())
            elif ">=" in line:
                key, val = line.split(">=")
                acceptance[key.strip() + "_min"] = float(val.strip())
        result["acceptance"] = acceptance

    return result


def _validate_dag(experiments: list[ExperimentSpec]) -> None:
    graph = {}
    for exp in experiments:
        graph[exp.name] = set(exp.dependencies)
    try:
        ts = TopologicalSorter(graph)
        ts.prepare()
    except CycleError as e:
        raise CyclicDependencyError(str(e)) from e


def _get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def generate_run_id(results_dir: Path, name: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}-(\d+)$")
    max_num = 0
    if results_dir.exists():
        for entry in results_dir.iterdir():
            match = pattern.match(entry.name)
            if match:
                max_num = max(max_num, int(match.group(1)))
    return f"{name}-{max_num + 1:04d}"


def save_experiment_config(
    results_dir: Path,
    run_id: str,
    spec: ExperimentSpec,
    config: SchedulerConfig,
    gpu_type_used: str,
) -> None:
    exp_dir = results_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "name": spec.name,
        "run_id": run_id,
        "description": spec.description,
        "hypothesis": spec.hypothesis,
        "tags": spec.tags,
        "script": spec.script,
        "command": spec.command,
        "dependencies": spec.dependencies,
        "gpu_type": gpu_type_used,
        "timeout": spec.timeout or config.experiment_timeout,
        "image": config.image,
        "acceptance": spec.acceptance,
        "git_commit": _get_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)


def save_train_snapshot(results_dir: Path, run_id: str, script_path: str) -> None:
    if not script_path:
        return
    exp_dir = results_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    source = Path(script_path)
    shutil.copy2(source, exp_dir / "train.py")


def save_experiment_result(
    results_dir: Path,
    run_id: str,
    state: ExperimentState,
    log: str,
    metrics: dict[str, float],
) -> None:
    exp_dir = results_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "log.txt", "w") as f:
        f.write(log)

    if metrics:
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    status_data = {
        "name": state.name,
        "run_id": state.run_id,
        "status": state.status.value,
        "instance_id": state.instance_id,
        "gpu_type_used": state.gpu_type_used,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "exit_code": state.exit_code,
        "error": state.error,
        "cost_usd": state.cost_usd,
        "gpu_utilization_pct": state.gpu_utilization_pct,
    }
    with open(exp_dir / "status.json", "w") as f:
        json.dump(status_data, f, indent=2)


def save_interpretation(results_dir: Path, run_id: str, interpretation: str) -> None:
    exp_dir = results_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "interpretation.md", "w") as f:
        f.write(interpretation)


def load_all_results(results_dir: Path, tags: list[str] | None = None, status: str | None = None) -> list[dict]:
    results = []
    if not results_dir.exists():
        return results

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        entry = {"run_id": exp_dir.name}

        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                entry["config"] = yaml.safe_load(f)

        status_path = exp_dir / "status.json"
        if status_path.exists():
            with open(status_path) as f:
                entry["status"] = json.load(f)

        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                entry["metrics"] = json.load(f)

        interpretation_path = exp_dir / "interpretation.md"
        if interpretation_path.exists():
            entry["interpretation"] = interpretation_path.read_text()

        if tags:
            entry_tags = entry.get("config", {}).get("tags", [])
            if not any(t in entry_tags for t in tags):
                continue

        if status:
            entry_status = entry.get("status", {}).get("status", "")
            if entry_status != status:
                continue

        results.append(entry)

    return results


def export_csv(results_dir: Path, output_path: str | None = None) -> str:
    results = load_all_results(results_dir)
    if not results:
        return "No results to export."

    all_metric_keys = set()
    for r in results:
        all_metric_keys.update(r.get("metrics", {}).keys())
    all_metric_keys = sorted(all_metric_keys)

    fieldnames = [
        "run_id", "name", "status", "tags", "gpu_type",
        "started_at", "finished_at", "cost_usd", "gpu_utilization_pct",
        "exit_code", "description", "hypothesis",
    ] + all_metric_keys

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for r in results:
        config = r.get("config", {})
        status_data = r.get("status", {})
        metrics = r.get("metrics", {})

        row = {
            "run_id": r["run_id"],
            "name": config.get("name", ""),
            "status": status_data.get("status", ""),
            "tags": ",".join(config.get("tags", [])),
            "gpu_type": status_data.get("gpu_type_used", config.get("gpu_type", "")),
            "started_at": status_data.get("started_at", ""),
            "finished_at": status_data.get("finished_at", ""),
            "cost_usd": status_data.get("cost_usd", ""),
            "gpu_utilization_pct": status_data.get("gpu_utilization_pct", ""),
            "exit_code": status_data.get("exit_code", ""),
            "description": config.get("description", ""),
            "hypothesis": config.get("hypothesis", ""),
            **metrics,
        }
        writer.writerow(row)

    csv_content = output.getvalue()

    if output_path:
        with open(output_path, "w") as f:
            f.write(csv_content)
        return f"Exported {len(results)} results to {output_path}"

    return csv_content


def save_state(path: Path, experiments: dict[str, ExperimentState]) -> None:
    data = {}
    for name, state in experiments.items():
        data[name] = {
            "run_id": state.run_id,
            "status": state.status.value,
            "instance_id": state.instance_id,
            "gpu_type_used": state.gpu_type_used,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "exit_code": state.exit_code,
            "error": state.error,
            "cost_usd": state.cost_usd,
            "gpu_utilization_pct": state.gpu_utilization_pct,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state(path: Path) -> dict[str, ExperimentState]:
    with open(path) as f:
        data = json.load(f)
    states = {}
    for name, s in data.items():
        states[name] = ExperimentState(
            name=name,
            run_id=s.get("run_id", ""),
            status=ExperimentStatus(s["status"]),
            instance_id=s.get("instance_id") or s.get("pod_id"),
            gpu_type_used=s.get("gpu_type_used"),
            started_at=s.get("started_at"),
            finished_at=s.get("finished_at"),
            exit_code=s.get("exit_code"),
            error=s.get("error"),
            cost_usd=s.get("cost_usd"),
            gpu_utilization_pct=s.get("gpu_utilization_pct"),
        )
    return states


def estimate_cost(price_per_hour: float, started_at: str, finished_at: str) -> float:
    if not price_per_hour:
        return 0.0
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(finished_at)
    hours = (end - start).total_seconds() / 3600
    return round(price_per_hour * hours, 4)
