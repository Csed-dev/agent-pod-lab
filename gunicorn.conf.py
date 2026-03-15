# Single worker: ExperimentManager holds in-memory state synced to
# scheduler_state.json without file locking. Multiple workers would
# race on that file and serve stale state.
bind = "0.0.0.0:8000"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
accesslog = "-"
