from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from orchestrator.errors import (
    DependencyNotFoundError,
    ExperimentNotFoundError,
    ExperimentNotReadyError,
    GpuUnavailableError,
    PodPoolFullError,
)
from orchestrator.models import (
    ExperimentSpec,
    ExperimentState,
    ExperimentStatus,
    SchedulerConfig,
    estimate_cost,
    export_csv,
    generate_run_id,
    load_all_results,
    load_experiments,
    load_state,
    save_experiment_config,
    save_experiment_result,
    save_interpretation,
    save_state,
    save_train_snapshot,
)
from orchestrator.notification import SmtpConfig, send_experiment_notification
from orchestrator.pod import PodConnection, PodManager


@dataclass(frozen=True)
class ExperimentStartResult:
    pod_id: str
    ssh_command: str
    run_id: str
    experiment_name: str
    description: str
    hypothesis: str
    acceptance: dict[str, float]
    interpretation_path: str
    readme_path: str

    def sub_agent_prompt(self) -> str:
        acceptance_str = "\n".join(f"- {k}: {v}" for k, v in self.acceptance.items()) if self.acceptance else "None defined"
        return (
            f"<task>\n"
            f"<experiment>{self.experiment_name}</experiment>\n"
            f"<run-id>{self.run_id}</run-id>\n"
            f"<description>{self.description}</description>\n"
            f"<hypothesis>{self.hypothesis}</hypothesis>\n"
            f"<acceptance>\n{acceptance_str}\n</acceptance>\n"
            f"<ssh-command>\n{self.ssh_command}\n</ssh-command>\n"
            f"<interpretation-path>{self.interpretation_path}</interpretation-path>\n"
            f"<readme-path>{self.readme_path}</readme-path>\n"
            f"</task>"
        )


class ExperimentManager:
    def __init__(self, experiments_path: str, config_path: str, state_path: str = "scheduler_state.json"):
        self._config = SchedulerConfig.from_yaml(config_path)
        self._experiments_path = experiments_path
        self._specs = load_experiments(experiments_path)
        self._spec_map = {s.name: s for s in self._specs}
        self._state_path = Path(state_path)
        self._results_dir = Path(self._config.results_dir)
        self._pm = PodManager()

        self._states: dict[str, ExperimentState] = {}
        if self._state_path.exists():
            self._states = load_state(self._state_path)

        for spec in self._specs:
            if spec.name not in self._states:
                self._states[spec.name] = ExperimentState(name=spec.name)

        self._active_pods: dict[str, PodConnection] = {}
        self._smtp_config = SmtpConfig.from_yaml(config_path)

    def validate_experiments(self) -> str:
        issues = []
        for spec in self._specs:
            for dep in spec.dependencies:
                if dep not in self._spec_map:
                    issues.append(f"  {spec.name}: dependency '{dep}' not found")
        if issues:
            return "Validation FAILED:\n" + "\n".join(issues)
        return f"Validation OK: {len(self._specs)} experiments, no issues."

    def experiments_status(self) -> str:
        self._propagate_blocked()
        lines = ["Experiment Status:"]
        for spec in self._specs:
            state = self._states[spec.name]
            deps_str = f" (deps: {', '.join(spec.dependencies)})" if spec.dependencies else ""
            gpu = spec.gpu_type or self._config.gpu_type
            desc = f" — {spec.description.strip()[:60]}" if spec.description else ""
            tags_str = f" [{','.join(spec.tags)}]" if spec.tags else ""
            run_str = f" run={state.run_id}" if state.run_id else ""
            lines.append(f"  {spec.name:30s} {state.status.value:10s}{run_str} gpu={gpu}{tags_str}{deps_str}{desc}")

        running = sum(1 for s in self._states.values() if s.status == ExperimentStatus.RUNNING)
        completed = sum(1 for s in self._states.values() if s.status == ExperimentStatus.COMPLETED)
        failed = sum(1 for s in self._states.values() if s.status == ExperimentStatus.FAILED)
        blocked = sum(1 for s in self._states.values() if s.status == ExperimentStatus.BLOCKED)
        ready = len(self.get_ready())

        lines.append(f"\nRunning: {running}/{self._config.max_pods}  Ready: {ready}  "
                      f"Completed: {completed}  Failed: {failed}  Blocked: {blocked}")
        return "\n".join(lines)

    def get_ready(self) -> list[str]:
        self._propagate_blocked()
        ready = []
        for spec in self._specs:
            state = self._states[spec.name]
            if state.status != ExperimentStatus.PENDING:
                continue
            dep_states = [self._states[d].status for d in spec.dependencies]
            if all(s == ExperimentStatus.COMPLETED for s in dep_states):
                ready.append(spec.name)
        return ready

    def start(self, experiment_name: str, gpu_type: str | None = None) -> ExperimentStartResult:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        state = self._states[experiment_name]
        if experiment_name not in self.get_ready():
            spec = self._spec_map[experiment_name]
            waiting = [d for d in spec.dependencies
                       if self._states[d].status != ExperimentStatus.COMPLETED]
            raise ExperimentNotReadyError(experiment_name, state.status.value, waiting)

        running_count = sum(1 for s in self._states.values() if s.status == ExperimentStatus.RUNNING)
        if running_count >= self._config.max_pods:
            running_names = [n for n, s in self._states.items() if s.status == ExperimentStatus.RUNNING]
            raise PodPoolFullError(self._config.max_pods, running_names)

        spec = self._spec_map[experiment_name]
        effective_gpu = gpu_type or spec.gpu_type or self._config.gpu_type

        try:
            pod_id = self._pm.create_pod(f"exp-{experiment_name}", gpu_type=effective_gpu)
        except Exception as e:
            if "no longer any instances" in str(e).lower() or "does not have the resources" in str(e).lower():
                gpus = self._pm.get_available_gpus()
                top5 = "\n".join(f"  {g['id']:40s} {g['memory_gb']:>4}GB  ${g['price_per_hr']:.2f}/hr" for g in gpus[:5])
                raise GpuUnavailableError(effective_gpu, f"Cheapest available:\n{top5}") from e
            raise

        run_id = generate_run_id(self._results_dir, experiment_name)

        state.status = ExperimentStatus.RUNNING
        state.run_id = run_id
        state.pod_id = pod_id
        state.gpu_type_used = effective_gpu
        state.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._persist()

        save_experiment_config(self._results_dir, run_id, spec, self._config, effective_gpu)
        save_train_snapshot(self._results_dir, run_id, spec.script)

        try:
            conn = self._pm.wait_until_ready(pod_id)
            self._active_pods[pod_id] = conn

            workspace = self._config.workspace_dir
            self._pm.ssh_run(conn, self._config.setup_command.format(workspace=workspace),
                             timeout=self._config.setup_timeout)

            if spec.script:
                self._pm.scp_to_pod(conn, spec.script, f"{workspace}/train.py")
        except Exception:
            self._pm.terminate_pod(pod_id)
            state.status = ExperimentStatus.PENDING
            state.run_id = ""
            state.pod_id = None
            state.gpu_type_used = None
            state.started_at = None
            self._persist()
            raise

        command = spec.command.format(workspace=workspace)
        timeout = spec.timeout or self._config.experiment_timeout

        ssh_command = self._build_ssh_command(conn, command, timeout)
        readme_path = str(Path(spec.script).parent / "README.md") if spec.script else ""
        return ExperimentStartResult(
            pod_id=pod_id,
            ssh_command=ssh_command,
            run_id=run_id,
            experiment_name=experiment_name,
            description=spec.description,
            hypothesis=spec.hypothesis,
            acceptance=spec.acceptance,
            interpretation_path=f"results/{run_id}/interpretation.md",
            readme_path=readme_path,
        )

    def finish(self, experiment_name: str, exit_code: int, output: str) -> None:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        state = self._states[experiment_name]
        state.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state.exit_code = exit_code

        if exit_code == 0:
            state.status = ExperimentStatus.COMPLETED
        else:
            state.status = ExperimentStatus.FAILED
            state.error = _extract_error(output)

        metrics = _extract_metrics(output)

        if state.gpu_type_used and state.started_at and state.finished_at:
            state.cost_usd = estimate_cost(state.gpu_type_used, state.started_at, state.finished_at)

        save_experiment_result(self._results_dir, state.run_id, state, output, metrics)

        if state.pod_id:
            conn = self._active_pods.get(state.pod_id)
            if conn:
                gpu_util = self._pm.ssh_run(conn, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo -1", timeout=10)
                util_val = float(gpu_util.strip())
                if util_val >= 0:
                    state.gpu_utilization_pct = util_val

            self._pm.terminate_pod(state.pod_id)
            self._active_pods.pop(state.pod_id, None)

        self._sync_to_cloud(state.run_id, metrics)

        if self._smtp_config:
            send_experiment_notification(self._smtp_config, state, metrics)

        self._propagate_blocked()
        self._persist()

    def cancel(self, experiment_name: str) -> str:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))
        state = self._states[experiment_name]
        if state.status != ExperimentStatus.RUNNING:
            return f"Experiment '{experiment_name}' is not running (status: {state.status.value})."
        if state.pod_id:
            self._pm.terminate_pod(state.pod_id)
            self._active_pods.pop(state.pod_id, None)
        state.status = ExperimentStatus.FAILED
        state.error = "Cancelled by user"
        state.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if state.gpu_type_used and state.started_at and state.finished_at:
            state.cost_usd = estimate_cost(state.gpu_type_used, state.started_at, state.finished_at)
        self._propagate_blocked()
        self._persist()
        return f"Experiment '{experiment_name}' cancelled. Pod terminated."

    def result(self, experiment_name: str) -> str:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        state = self._states[experiment_name]
        spec = self._spec_map[experiment_name]
        lines = [f"Experiment: {experiment_name}"]
        if state.run_id:
            lines.append(f"Run ID: {state.run_id}")
        lines.append(f"Status: {state.status.value}")
        if spec.tags:
            lines.append(f"Tags: {', '.join(spec.tags)}")
        if spec.description:
            lines.append(f"Description: {spec.description.strip()}")
        if spec.hypothesis:
            lines.append(f"Hypothesis: {spec.hypothesis.strip()}")

        if state.started_at:
            lines.append(f"Started: {state.started_at}")
        if state.finished_at:
            lines.append(f"Finished: {state.finished_at}")
        if state.gpu_type_used:
            lines.append(f"GPU: {state.gpu_type_used}")
        if state.cost_usd is not None:
            lines.append(f"Cost: ${state.cost_usd:.4f}")
        if state.gpu_utilization_pct is not None:
            lines.append(f"GPU Utilization: {state.gpu_utilization_pct:.0f}%")
        if state.exit_code is not None:
            lines.append(f"Exit code: {state.exit_code}")
        if state.error:
            lines.append(f"Error: {state.error}")

        exp_dir = self._results_dir / state.run_id if state.run_id else self._results_dir / experiment_name

        metrics_path = exp_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            lines.append("\nMetrics:")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")

        if spec.acceptance and metrics:
            lines.append("\nAcceptance Criteria:")
            for criterion, threshold in spec.acceptance.items():
                if criterion.endswith("_max"):
                    key = criterion[:-4]
                    val = metrics.get(key)
                    passed = val is not None and val <= threshold
                    lines.append(f"  {key} <= {threshold}: {'PASS' if passed else 'FAIL'} (actual: {val})")
                elif criterion.endswith("_min"):
                    key = criterion[:-4]
                    val = metrics.get(key)
                    passed = val is not None and val >= threshold
                    lines.append(f"  {key} >= {threshold}: {'PASS' if passed else 'FAIL'} (actual: {val})")

        interpretation_path = exp_dir / "interpretation.md"
        if interpretation_path.exists():
            lines.append(f"\nInterpretation:\n{interpretation_path.read_text().strip()}")

        log_path = exp_dir / "log.txt"
        if log_path.exists():
            log = log_path.read_text()
            lines.append(f"\nLog (last 20 lines):")
            for line in log.strip().split("\n")[-20:]:
                lines.append(f"  {line}")

        return "\n".join(lines)

    def meets_acceptance(self, experiment_name: str) -> bool | None:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        spec = self._spec_map[experiment_name]
        state = self._states[experiment_name]
        if not spec.acceptance or not state.run_id:
            return None

        metrics_path = self._results_dir / state.run_id / "metrics.json"
        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            metrics = json.load(f)

        for criterion, threshold in spec.acceptance.items():
            if criterion.endswith("_max"):
                key = criterion[:-4]
                val = metrics.get(key)
                if val is None or val > threshold:
                    return False
            elif criterion.endswith("_min"):
                key = criterion[:-4]
                val = metrics.get(key)
                if val is None or val < threshold:
                    return False
        return True

    def write_interpretation(self, experiment_name: str, interpretation: str) -> None:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))
        state = self._states[experiment_name]
        run_id = state.run_id or experiment_name
        save_interpretation(self._results_dir, run_id, interpretation)

    def add_experiment(
        self,
        name: str,
        script: str,
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
        gpu_type: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if name in self._spec_map:
            raise ValueError(f"Experiment '{name}' already exists.")

        if dependencies:
            for dep in dependencies:
                if dep not in self._spec_map:
                    raise DependencyNotFoundError(name, dep, list(self._spec_map.keys()))

        from orchestrator.models import _load_experiment_readme
        readme = _load_experiment_readme(script)

        command = f"cd {{workspace}} && uv run {script}"

        spec = ExperimentSpec(
            name=name,
            command=command,
            script=script,
            description=readme.get("description", ""),
            hypothesis=readme.get("hypothesis", ""),
            tags=tags or [],
            dependencies=dependencies or [],
            gpu_type=gpu_type,
            timeout=timeout,
            acceptance=readme.get("acceptance", {}),
        )

        self._specs.append(spec)
        self._spec_map[name] = spec
        self._states[name] = ExperimentState(name=name)
        self._persist()
        self._append_experiment_to_yaml(spec)

        return f"Experiment '{name}' added. Status: PENDING."

    def all_results(self, tags: list[str] | None = None, status: str | None = None) -> list[dict]:
        return load_all_results(self._results_dir, tags=tags, status=status)

    def export_csv(self, output_path: str | None = None) -> str:
        return export_csv(self._results_dir, output_path)

    def available_experiments_detail(self) -> str:
        lines = ["Experiments:"]
        for spec in self._specs:
            state = self._states[spec.name]
            lines.append(f"\n  {spec.name} [{state.status.value}]")
            lines.append(f"    Script: {spec.script}")
            if spec.description:
                lines.append(f"    Description: {spec.description.strip()[:100]}")
            if spec.hypothesis:
                lines.append(f"    Hypothesis: {spec.hypothesis.strip()[:100]}")
            if spec.dependencies:
                lines.append(f"    Dependencies: {', '.join(spec.dependencies)}")
            if spec.acceptance:
                criteria = ", ".join(f"{k}: {v}" for k, v in spec.acceptance.items())
                lines.append(f"    Acceptance: {criteria}")
        return "\n".join(lines)

    def available_gpus(self, min_memory_gb: int = 0) -> str:
        gpus = self._pm.get_available_gpus(min_memory_gb)
        lines = ["Available GPUs:"]
        for g in gpus:
            lines.append(f"  {g['id']:45s} {g['memory_gb']:>4}GB  ${g['price_per_hr']:.2f}/hr")
        return "\n".join(lines)

    def cleanup_orphaned_pods(self) -> str:
        pods = self._pm.list_pods()
        running_pod_ids = {s.pod_id for s in self._states.values()
                          if s.status == ExperimentStatus.RUNNING and s.pod_id}
        orphans = [p for p in pods if p.status in ("RUNNING", "STARTING")
                   and p.pod_id not in running_pod_ids]
        if not orphans:
            return "No orphaned pods found."
        lines = ["Terminating orphaned pods:"]
        for pod in orphans:
            self._pm.terminate_pod(pod.pod_id)
            lines.append(f"  {pod.pod_id} | {pod.name} | {pod.gpu_type} — terminated")
        return "\n".join(lines)

    def reset(self, experiment_name: str) -> str:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))
        state = self._states[experiment_name]
        if state.status == ExperimentStatus.RUNNING and state.pod_id:
            self._pm.terminate_pod(state.pod_id)
            self._active_pods.pop(state.pod_id, None)
        state.status = ExperimentStatus.PENDING
        state.run_id = ""
        state.pod_id = None
        state.gpu_type_used = None
        state.started_at = None
        state.finished_at = None
        state.exit_code = None
        state.error = None
        state.cost_usd = None
        state.gpu_utilization_pct = None
        self._persist()
        return f"Experiment '{experiment_name}' reset to PENDING."

    def _propagate_blocked(self) -> None:
        changed = True
        while changed:
            changed = False
            for spec in self._specs:
                state = self._states[spec.name]
                if state.status != ExperimentStatus.PENDING:
                    continue
                for dep in spec.dependencies:
                    dep_status = self._states[dep].status
                    if dep_status in (ExperimentStatus.FAILED, ExperimentStatus.BLOCKED):
                        state.status = ExperimentStatus.BLOCKED
                        state.error = f"Blocked by failed dependency: {dep}"
                        changed = True
                        break

    def _persist(self) -> None:
        save_state(self._state_path, self._states)

    def _append_experiment_to_yaml(self, spec: ExperimentSpec) -> None:
        with open(self._experiments_path) as f:
            data = yaml.safe_load(f) or {}
        if "experiments" not in data:
            data["experiments"] = []
        entry = {"name": spec.name, "script": spec.script}
        if spec.dependencies:
            entry["dependencies"] = spec.dependencies
        if spec.tags:
            entry["tags"] = spec.tags
        if spec.gpu_type:
            entry["gpu_type"] = spec.gpu_type
        if spec.timeout:
            entry["timeout"] = spec.timeout
        data["experiments"].append(entry)
        with open(self._experiments_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _sync_to_cloud(self, run_id: str, metrics: dict[str, float]) -> None:
        import os
        if not os.environ.get("CF_ACCOUNT_ID"):
            return
        from orchestrator.cloudflare_storage import D1Client, R2Client
        config_path = self._results_dir / run_id / "config.yaml"
        status_path = self._results_dir / run_id / "status.json"
        if not config_path.exists() or not status_path.exists():
            raise FileNotFoundError(f"Missing config.yaml or status.json in results/{run_id}/")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        with open(status_path) as f:
            status_data = json.load(f)
        d1 = D1Client()
        d1.save_experiment(run_id, config, status_data)
        if metrics:
            d1.save_metrics(run_id, metrics)
        r2 = R2Client()
        r2.upload_experiment_files(self._results_dir, run_id)

    def _build_ssh_command(self, conn: PodConnection, command: str, timeout: int) -> str:
        inner = f"timeout {timeout} bash -c {_shell_quote(command)}"
        return (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"-p {conn.port} root@{conn.ip} "
            f"{_shell_quote(inner)}"
        )


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _extract_error(output: str) -> str:
    lines = output.strip().split("\n")
    error_lines = [l for l in lines[-20:] if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()]
    if error_lines:
        return error_lines[-1][:200]
    return lines[-1][:200] if lines else "Unknown error"


def _extract_metrics(log: str) -> dict[str, float]:
    metrics = {}
    for match in re.finditer(r"^([a-z_]+):\s+([0-9.eE+\-]+)%?", log, re.MULTILINE):
        key = match.group(1)
        try:
            metrics[key] = float(match.group(2))
        except ValueError:
            pass
    return metrics
