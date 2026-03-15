from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Connection:
    instance_id: str
    ip: str
    port: int


@dataclass(frozen=True)
class InstanceInfo:
    instance_id: str
    name: str
    status: str
    gpu_type: str
    cost_per_hr: float


class ComputePort(Protocol):
    def create_instance(
        self, name: str, gpu_type: str, image: str, disk_gb: int
    ) -> str: ...

    def wait_until_ready(self, instance_id: str) -> Connection: ...

    def run_command(
        self, conn: Connection, command: str, timeout: int = 120
    ) -> str: ...

    def upload_file(
        self, conn: Connection, local_path: str, remote_path: str
    ) -> None: ...

    def download_file(
        self, conn: Connection, remote_path: str, local_path: str
    ) -> None: ...

    def terminate_instance(self, instance_id: str) -> None: ...

    def list_instances(self) -> list[InstanceInfo]: ...

    def available_gpus(self, min_memory_gb: int = 0) -> list[dict]: ...

    def gpu_prices(self) -> dict[str, float]: ...

    def build_exec_command(
        self, conn: Connection, command: str, timeout: int
    ) -> str: ...
