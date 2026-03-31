import os
import subprocess
import time

import runpod

from orchestrator.ports.compute import ComputePort, Connection, InstanceInfo

POD_READY_TIMEOUT_S = 120
POD_POLL_INTERVAL_S = 10
SSH_CONNECT_TIMEOUT_S = 10


def _ssh_base_args(conn: Connection) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=5",
        "-p", str(conn.port),
        f"root@{conn.ip}",
    ]


def _scp_base_args(conn: Connection) -> list[str]:
    return [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=5",
        "-P", str(conn.port),
    ]


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


class RunPodCompute(ComputePort):
    def __init__(self) -> None:
        api_key = os.environ["RUNPOD_API_KEY"]
        runpod.api_key = api_key
        self._api_key = api_key

    def create_instance(
        self, name: str, gpu_type: str, image: str, disk_gb: int
    ) -> str:
        response = runpod.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type,
            gpu_count=1,
            volume_in_gb=0,
            container_disk_in_gb=disk_gb,
            ports="22/tcp",
            start_ssh=True,
        )
        return response["id"]

    def wait_until_ready(self, instance_id: str) -> Connection:
        deadline = time.monotonic() + POD_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            pod = runpod.get_pod(instance_id)
            if pod is None:
                time.sleep(POD_POLL_INTERVAL_S)
                continue
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            for port_info in ports:
                if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                    conn = Connection(
                        instance_id=instance_id,
                        ip=port_info["ip"],
                        port=port_info["publicPort"],
                    )
                    self._wait_ssh_ready(conn)
                    return conn
            time.sleep(POD_POLL_INTERVAL_S)
        self.terminate_instance(instance_id)
        raise TimeoutError(
            f"Pod {instance_id} not ready within {POD_READY_TIMEOUT_S}s — terminated. "
            f"Check available GPUs with available_gpus() and try a different gpu_type."
        )

    def run_command(self, conn: Connection, command: str, timeout: int = 120) -> str:
        result = subprocess.run(
            _ssh_base_args(conn) + [command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    def upload_file(self, conn: Connection, local_path: str, remote_path: str) -> None:
        result = subprocess.run(
            _scp_base_args(conn) + [local_path, f"root@{conn.ip}:{remote_path}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

    def download_file(self, conn: Connection, remote_path: str, local_path: str) -> None:
        result = subprocess.run(
            _scp_base_args(conn) + [f"root@{conn.ip}:{remote_path}", local_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP from pod failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

    def terminate_instance(self, instance_id: str) -> None:
        runpod.terminate_pod(instance_id)

    def list_instances(self) -> list[InstanceInfo]:
        pods = runpod.get_pods()
        return [
            InstanceInfo(
                instance_id=p["id"],
                name=p.get("name", ""),
                status=p.get("desiredStatus", "UNKNOWN"),
                gpu_type=p.get("machine", {}).get("gpuDisplayName", ""),
                cost_per_hr=p.get("costPerHr", 0.0),
            )
            for p in pods
        ]

    def available_gpus(self, min_memory_gb: int = 0) -> list[dict]:
        import requests

        response = requests.post(
            f"https://api.runpod.io/graphql?api_key={self._api_key}",
            json={"query": """
                query { gpuTypes {
                    id displayName memoryInGb
                    secureCloud communityCloud
                    lowestPrice(input: { gpuCount: 1 }) {
                        uninterruptablePrice
                    }
                }}
            """},
        )
        response.raise_for_status()
        gpus = response.json()["data"]["gpuTypes"]
        available = []
        for g in gpus:
            mem = g.get("memoryInGb", 0)
            if mem < min_memory_gb:
                continue
            if not (g.get("secureCloud") or g.get("communityCloud")):
                continue
            price = (g.get("lowestPrice") or {}).get("uninterruptablePrice") or 0
            if price == 0:
                continue
            available.append({
                "id": g["id"],
                "memory_gb": mem,
                "price_per_hr": price,
                "secure": bool(g.get("secureCloud")),
                "community": bool(g.get("communityCloud")),
            })
        return sorted(available, key=lambda x: x["price_per_hr"])

    def gpu_prices(self) -> dict[str, float]:
        gpus = self.available_gpus(min_memory_gb=0)
        return {g["id"]: g["price_per_hr"] for g in gpus}

    def build_exec_command(self, conn: Connection, command: str, timeout: int) -> str:
        inner = f"timeout {timeout} bash -c {_shell_quote(command)}"
        return (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"-o ServerAliveInterval=30 -o ServerAliveCountMax=5 "
            f"-p {conn.port} root@{conn.ip} "
            f"{_shell_quote(inner)}"
        )

    def start_background_job(
        self, conn: Connection, command: str, log_path: str, pid_path: str, exit_code_path: str
    ) -> None:
        wrapper = (
            f"nohup bash -c {_shell_quote(command + f'; echo $? > {exit_code_path}')} "
            f"> {log_path} 2>&1 & echo $! > {pid_path}"
        )
        self.run_command(conn, wrapper, timeout=30)

    def poll_job(
        self, conn: Connection, pid_path: str, log_path: str, exit_code_path: str
    ) -> tuple[bool, int | None, str]:
        check_cmd = (
            f"pid=$(cat {pid_path} 2>/dev/null || echo 0); "
            f"if [ \"$pid\" -ne 0 ] && kill -0 \"$pid\" 2>/dev/null; then echo RUNNING; "
            f"else cat {exit_code_path} 2>/dev/null || echo -1; fi; "
            f"echo '---LOG---'; tail -50 {log_path} 2>/dev/null || true"
        )
        output = self.run_command(conn, check_cmd, timeout=30)
        parts = output.split("---LOG---", 1)
        status_line = parts[0].strip()
        log_tail = parts[1].strip() if len(parts) > 1 else ""
        if status_line == "RUNNING":
            return False, None, log_tail
        exit_code = int(status_line)
        return True, exit_code, log_tail

    def _wait_ssh_ready(self, conn: Connection, retries: int = 10) -> None:
        for attempt in range(retries):
            try:
                self.run_command(conn, "echo ready", timeout=15)
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == retries - 1:
                    raise
                time.sleep(5)
