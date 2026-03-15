import sys

sys.path.insert(0, ".")

from orchestrator.adapters.runpod import RunPodCompute


def test_pod_lifecycle():
    compute = RunPodCompute()

    print("1. Checking for orphaned pods...")
    for inst in compute.list_instances():
        print(f"   {inst.instance_id} | {inst.name} | {inst.status} | {inst.gpu_type} | ${inst.cost_per_hr}/hr")

    print("2. Checking available GPUs...")
    gpus = compute.available_gpus(min_memory_gb=20)
    gpu_type = gpus[0]["id"]
    print(f"   Cheapest: {gpu_type} ${gpus[0]['price_per_hr']:.2f}/hr")

    print(f"3. Creating instance ({gpu_type})...")
    instance_id = compute.create_instance("test-lifecycle", gpu_type=gpu_type, image="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", disk_gb=20)
    print(f"   Instance ID: {instance_id}")

    try:
        print("4. Waiting for instance to be ready...")
        conn = compute.wait_until_ready(instance_id)
        print(f"   Connected: {conn.ip}:{conn.port}")

        print("5. Testing command execution...")
        gpu_info = compute.run_command(conn, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        print(f"   GPU: {gpu_info.strip()}")

        python_version = compute.run_command(conn, "python3 --version")
        print(f"   {python_version.strip()}")

        print("6. Testing file upload...")
        compute.upload_file(conn, "train.py", "/tmp/test_upload.py")
        verify = compute.run_command(conn, "ls -la /tmp/test_upload.py")
        print(f"   {verify.strip()}")

        print("ALL TESTS PASSED")
    finally:
        print("7. Terminating instance...")
        compute.terminate_instance(instance_id)
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
