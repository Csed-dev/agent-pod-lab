import os
import subprocess
import time

import runpod

from orchestrator.ports.compute import ComputePort, Connection, InstanceInfo

POD_READY_TIMEOUT_S = 120
POD_POLL_INTERVAL_S = 10
SSH_CONNECT_TIMEOUT_S = 10


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
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
                "-p", str(conn.port),
                f"root@{conn.ip}",
                command,
            ],
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
            [
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
                "-P", str(conn.port),
                local_path,
                f"root@{conn.ip}:{remote_path}",
            ],
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
            [
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
                "-P", str(conn.port),
                f"root@{conn.ip}:{remote_path}",
                local_path,
            ],
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
            f"-p {conn.port} root@{conn.ip} "
            f"{_shell_quote(inner)}"
        )

    def _wait_ssh_ready(self, conn: Connection, retries: int = 10) -> None:
        for attempt in range(retries):
            try:
                self.run_command(conn, "echo ready", timeout=15)
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == retries - 1:
                    raise
                time.sleep(5)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"
