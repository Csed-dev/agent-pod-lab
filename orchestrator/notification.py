from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path

import yaml

from orchestrator.models import ExperimentState, ExperimentStatus


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    user: str
    password: str
    to: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> SmtpConfig | None:
        with open(path) as f:
            data = yaml.safe_load(f)
        smtp = data.get("smtp")
        if not smtp:
            return None
        return cls(
            host=smtp["host"],
            port=smtp["port"],
            user=os.environ["SMTP_USER"],
            password=os.environ["SMTP_PASSWORD"],
            to=smtp["to"],
        )


def send_experiment_notification(
    config: SmtpConfig,
    state: ExperimentState,
    metrics: dict[str, float],
) -> None:
    status_label = "COMPLETED" if state.status == ExperimentStatus.COMPLETED else "FAILED"
    subject = f"[Experiment {status_label}] {state.name} ({state.run_id})"

    lines = [
        f"Experiment: {state.name}",
        f"Run ID:     {state.run_id}",
        f"Status:     {status_label}",
        f"Started:    {state.started_at}",
        f"Finished:   {state.finished_at}",
    ]

    if state.cost_usd is not None:
        lines.append(f"Cost:       ${state.cost_usd:.2f}")

    if state.error:
        lines.append(f"Error:      {state.error}")

    if metrics:
        lines.append("")
        lines.append("Metrics:")
        for key, value in metrics.items():
            lines.append(f"  {key}: {value}")

    body = "\n".join(lines)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config.user
    msg["To"] = config.to

    with smtplib.SMTP(config.host, config.port) as server:
        server.starttls()
        server.login(config.user, config.password)
        server.send_message(msg)
