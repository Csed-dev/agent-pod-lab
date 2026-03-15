from __future__ import annotations

from typing import Protocol

from orchestrator.models import ExperimentState


class NotificationPort(Protocol):
    def notify(
        self, state: ExperimentState, metrics: dict[str, float]
    ) -> None: ...
