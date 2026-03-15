from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from orchestrator.ports.storage import CloudSyncPort

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
CF_D1_DATABASE_ID = os.environ.get("CF_D1_DATABASE_ID", "")
CF_R2_ACCESS_KEY = os.environ.get("CF_R2_ACCESS_KEY", "")
CF_R2_SECRET_KEY = os.environ.get("CF_R2_SECRET_KEY", "")
CF_R2_BUCKET = os.environ.get("CF_R2_BUCKET", "agent-pod-lab")

D1_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database"


class CloudflareSync(CloudSyncPort):
    def __init__(self) -> None:
        if not CF_ACCOUNT_ID or not CF_API_TOKEN or not CF_D1_DATABASE_ID:
            raise EnvironmentError(
                "Missing Cloudflare D1 env vars. Required:\n"
                "  CF_ACCOUNT_ID, CF_API_TOKEN, CF_D1_DATABASE_ID"
            )
        if not CF_R2_ACCESS_KEY or not CF_R2_SECRET_KEY:
            raise EnvironmentError(
                "Missing Cloudflare R2 env vars. Required:\n"
                "  CF_R2_ACCESS_KEY, CF_R2_SECRET_KEY"
            )
        self._d1 = _D1Client()
        self._r2 = _R2Client()

    def sync_experiment(
        self,
        results_dir: Path,
        run_id: str,
        config: dict,
        status_data: dict,
        metrics: dict[str, float],
    ) -> None:
        self._d1.save_experiment(run_id, config, status_data)
        if metrics:
            self._d1.save_metrics(run_id, metrics)
        self._r2.upload_experiment_files(results_dir, run_id)


class _D1Client:
    def __init__(self) -> None:
        self._url = f"{D1_BASE_URL}/{CF_D1_DATABASE_ID}/query"
        self._headers = {
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        }

    def execute(self, sql: str, params: list | None = None) -> list[dict]:
        body: dict = {"sql": sql}
        if params:
            body["params"] = params
        response = requests.post(self._url, headers=self._headers, json=body)
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            errors = data.get("errors", [])
            raise RuntimeError(f"D1 query failed: {errors}")
        return data["result"][0].get("results", [])

    def init_schema(self) -> None:
        self.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT DEFAULT '',
                hypothesis TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                command TEXT DEFAULT '',
                gpu_type TEXT DEFAULT '',
                image TEXT DEFAULT '',
                git_commit TEXT DEFAULT '',
                acceptance TEXT DEFAULT '{}',
                started_at TEXT,
                finished_at TEXT,
                exit_code INTEGER,
                error TEXT,
                cost_usd REAL,
                gpu_utilization_pct REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (run_id, key),
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)

    def save_experiment(self, run_id: str, config: dict, status_data: dict) -> None:
        self.execute(
            """INSERT OR REPLACE INTO experiments
                (run_id, name, status, description, hypothesis, tags,
                 command, gpu_type, image, git_commit, acceptance,
                 started_at, finished_at, exit_code, error, cost_usd, gpu_utilization_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                run_id,
                config.get("name", ""),
                status_data.get("status", ""),
                config.get("description", ""),
                config.get("hypothesis", ""),
                ",".join(config.get("tags", [])),
                config.get("command", ""),
                status_data.get("gpu_type_used", config.get("gpu_type", "")),
                config.get("image", ""),
                config.get("git_commit", ""),
                json.dumps(config.get("acceptance", {})),
                status_data.get("started_at"),
                status_data.get("finished_at"),
                status_data.get("exit_code"),
                status_data.get("error"),
                status_data.get("cost_usd"),
                status_data.get("gpu_utilization_pct"),
            ],
        )

    def save_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.execute(
                "INSERT OR REPLACE INTO metrics (run_id, key, value) VALUES (?, ?, ?)",
                [run_id, key, value],
            )

    def query_experiments(self, where: str = "1=1", params: list | None = None) -> list[dict]:
        return self.execute(f"SELECT * FROM experiments WHERE {where} ORDER BY created_at DESC", params)

    def query_metrics(self, run_id: str) -> dict[str, float]:
        rows = self.execute("SELECT key, value FROM metrics WHERE run_id = ?", [run_id])
        return {r["key"]: r["value"] for r in rows}

    def query_all_with_metrics(self, where: str = "1=1", params: list | None = None) -> list[dict]:
        experiments = self.query_experiments(where, params)
        if not experiments:
            return experiments
        run_ids = [e["run_id"] for e in experiments]
        placeholders = ",".join("?" for _ in run_ids)
        all_metrics = self.execute(
            f"SELECT run_id, key, value FROM metrics WHERE run_id IN ({placeholders})",
            run_ids,
        )
        metrics_by_run: dict[str, dict[str, float]] = {}
        for row in all_metrics:
            metrics_by_run.setdefault(row["run_id"], {})[row["key"]] = row["value"]
        for exp in experiments:
            exp["metrics"] = metrics_by_run.get(exp["run_id"], {})
        return experiments


class _R2Client:
    def __init__(self) -> None:
        import boto3
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=CF_R2_ACCESS_KEY,
            aws_secret_access_key=CF_R2_SECRET_KEY,
            region_name="auto",
        )
        self._bucket = CF_R2_BUCKET

    def upload_file(self, local_path: str | Path, remote_key: str) -> None:
        self._s3.upload_file(str(local_path), self._bucket, remote_key)

    def upload_experiment_files(self, results_dir: Path, run_id: str) -> int:
        exp_dir = results_dir / run_id
        if not exp_dir.exists():
            return 0
        count = 0
        for file_path in exp_dir.iterdir():
            if file_path.is_file():
                remote_key = f"results/{run_id}/{file_path.name}"
                self.upload_file(file_path, remote_key)
                count += 1
        return count
