import sys

sys.path.insert(0, ".")

from orchestrator import PodManager


def test_pod_lifecycle():
    pm = PodManager()

    print("1. Checking for orphaned pods...")
    for pod in pm.list_pods():
        print(f"   {pod.pod_id} | {pod.name} | {pod.status} | {pod.gpu_type} | ${pod.cost_per_hr}/hr")

    print("2. Checking available GPUs...")
    gpus = pm.get_available_gpus(min_memory_gb=20)
    gpu_type = gpus[0]["id"]
    print(f"   Cheapest: {gpu_type} ${gpus[0]['price_per_hr']:.2f}/hr")

    print(f"3. Creating pod ({gpu_type})...")
    pod_id = pm.create_pod("test-lifecycle", gpu_type=gpu_type)
    print(f"   Pod ID: {pod_id}")

    try:
        print("4. Waiting for pod to be ready...")
        conn = pm.wait_until_ready(pod_id)
        print(f"   Connected: {conn.ip}:{conn.port}")

        print("5. Testing SSH...")
        gpu_info = pm.ssh_run(conn, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        print(f"   GPU: {gpu_info.strip()}")

        python_version = pm.ssh_run(conn, "python3 --version")
        print(f"   {python_version.strip()}")

        print("6. Testing SCP...")
        pm.scp_to_pod(conn, "train.py", "/tmp/test_upload.py")
        verify = pm.ssh_run(conn, "ls -la /tmp/test_upload.py")
        print(f"   {verify.strip()}")

        print("ALL TESTS PASSED")
    finally:
        print("7. Terminating pod...")
        pm.terminate_pod(pod_id)
        print("   Terminated")


def test_experiment_manager():
    from orchestrator import ExperimentManager

    print("1. Loading experiments...")
    mgr = ExperimentManager("experiments.yaml", "scheduler_config.yaml")

    print("2. Validating...")
    print(f"   {mgr.validate_experiments()}")

    print("3. Status...")
    print(mgr.experiments_status())

    print("4. Ready experiments...")
    ready = mgr.get_ready()
    print(f"   Ready: {ready}")

    print("5. Available GPUs...")
    print(mgr.available_gpus())

    print("6. Experiment details...")
    print(mgr.available_experiments_detail())

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test",
        choices=["lifecycle", "manager"],
        help="lifecycle: pod create/ssh/terminate. manager: ExperimentManager validation.",
    )
    args = parser.parse_args()

    {"lifecycle": test_pod_lifecycle, "manager": test_experiment_manager}[args.test]()
