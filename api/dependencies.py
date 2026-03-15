from functools import lru_cache

from orchestrator import ExperimentManager


@lru_cache(maxsize=1)
def get_manager() -> ExperimentManager:
    return ExperimentManager("experiments.yaml", "scheduler_config.yaml")
