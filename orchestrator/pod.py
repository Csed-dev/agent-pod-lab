import os
import subprocess
import time
from dataclasses import dataclass

import runpod

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
POD_READY_TIMEOUT_S = 120
POD_POLL_INTERVAL_S = 10
SSH_CONNECT_TIMEOUT_S = 10

runpod.api_key = RUNPOD_API_KEY


@dataclass(frozen=True)
class PodConnection:
    pod_id: str
    ip: str
    port: int


@dataclass(frozen=True)
class PodInfo:
    pod_id: str
    name: str
    status: str
    gpu_type: str
    cost_per_hr: float


class PodManager:
    def create_pod(
        self,
        name: str,
        gpu_type: str,
        gpu_count: int = 1,
        image: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        container_disk_gb: int = 20,
    ) -> str:
        response = runpod.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type,
            gpu_count=gpu_count,
            volume_in_gb=0,
            container_disk_in_gb=container_disk_gb,
            ports="22/tcp",
            start_ssh=True,
        )
        return response["id"]

    def wait_until_ready(self, pod_id: str) -> PodConnection:
        deadline = time.monotonic() + POD_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            pod = runpod.get_pod(pod_id)
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            for port_info in ports:
                if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                    conn = PodConnection(
                        pod_id=pod_id,
                        ip=port_info["ip"],
                        port=port_info["publicPort"],
                    )
                    self._wait_ssh_ready(conn)
                    return conn
            time.sleep(POD_POLL_INTERVAL_S)
        self.terminate_pod(pod_id)
        raise TimeoutError(
            f"Pod {pod_id} not ready within {POD_READY_TIMEOUT_S}s — terminated. "
            f"Check available GPUs with get_available_gpus() and try a different gpu_type."
        )

    def ssh_run(self, conn: PodConnection, command: str, timeout: int = 120) -> str:
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

    def scp_to_pod(self, conn: PodConnection, local_path: str, remote_path: str, timeout: int = 30) -> None:
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
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

    def scp_from_pod(self, conn: PodConnection, remote_path: str, local_path: str, timeout: int = 60) -> None:
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
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP from pod failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

    def terminate_pod(self, pod_id: str) -> None:
        runpod.terminate_pod(pod_id)

    def stop_pod(self, pod_id: str) -> None:
        runpod.stop_pod(pod_id)

    def list_pods(self) -> list[PodInfo]:
        pods = runpod.get_pods()
        return [
            PodInfo(
                pod_id=p["id"],
                name=p.get("name", ""),
                status=p.get("desiredStatus", "UNKNOWN"),
                gpu_type=p.get("machine", {}).get("gpuDisplayName", ""),
                cost_per_hr=p.get("costPerHr", 0.0),
            )
            for p in pods
        ]

    def get_available_gpus(self, min_memory_gb: int = 0) -> list[dict]:
        import requests

        response = requests.post(
            f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}",
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

    def _wait_ssh_ready(self, conn: PodConnection, retries: int = 10) -> None:
        for attempt in range(retries):
            try:
                self.ssh_run(conn, "echo ready", timeout=15)
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == retries - 1:
                    raise
                time.sleep(5)
