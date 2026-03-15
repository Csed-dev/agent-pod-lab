from __future__ import annotations

from pathlib import Path
from typing import Protocol


class CloudSyncPort(Protocol):
    def sync_experiment(
        self,
        results_dir: Path,
        run_id: str,
        config: dict,
        status_data: dict,
        metrics: dict[str, float],
    ) -> None: ...
