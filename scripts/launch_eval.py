#!/usr/bin/env python3
"""Parallel evaluation launcher for GR00T on RoboCasa365.

Submits a single SLURM job (via submitit) that spans N nodes.
Each node uses ThreadPoolExecutor to run 8 GPU workers in parallel.
"""

import argparse
import os
import platform
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import submitit
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = str(REPO_ROOT / "scripts" / "run_eval_distributed.py")


def resolve_tasks(task_sets: list[str]) -> list[str]:
    tasks = []
    for ts in task_sets:
        tasks.extend(TASK_SET_REGISTRY[ts])
    return sorted(set(tasks))


def partition_tasks(tasks: list[str], n_workers: int) -> list[list[str]]:
    """Greedy bin-packing by task horizon."""
    if n_workers > len(tasks):
        raise ValueError(f"n_workers ({n_workers}) > n_tasks ({len(tasks)})")
    tasks_with_horizon = [(t, get_task_horizon(t)) for t in tasks]
    tasks_with_horizon.sort(key=lambda x: x[1], reverse=True)
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for task, horizon in tasks_with_horizon:
        min_idx = loads.index(min(loads))
        buckets[min_idx].append(task)
        loads[min_idx] += horizon
    return buckets


def run_gpu(gpu_id, tasks, model_path, split, run_id, base_port,
            n_episodes, n_envs, n_action_steps, video_dir, embodiment_tag, data_config):
    """Run evaluation on a single GPU via subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "egl"
    conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:/usr/lib64:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        sys.executable, WORKER_SCRIPT,
        "--tasks", *tasks,
        "--port", str(base_port + gpu_id),
        "--run_id", run_id,
        "--model_path", model_path,
        "--split", split,
        "--n_episodes", str(n_episodes),
        "--n_envs", str(n_envs),
        "--n_action_steps", str(n_action_steps),
        "--embodiment_tag", embodiment_tag,
        "--data_config", data_config,
        "--video_dir", video_dir,
    ]
    print(f"[GPU {gpu_id}] port={base_port + gpu_id} tasks={tasks}")
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return gpu_id, result.returncode


def node_eval(all_partitions, gpus_per_node, model_path, split, run_id,
              base_port, n_episodes, n_envs, n_action_steps, video_dir,
              embodiment_tag, data_config):
    """Run on each node. Detects SLURM_NODEID, picks its partition, runs 8 GPUs."""
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    hostname = platform.node()
    start = node_id * gpus_per_node
    my_partitions = all_partitions[start:start + gpus_per_node]
    node_port = base_port + node_id * 100

    print(f"[Node {node_id}] hostname={hostname} "
          f"{len(my_partitions)} GPUs (partitions {start}-{start + len(my_partitions) - 1})")

    # GPU health check
    try:
        gpu_check = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,ecc.errors.uncorrected.volatile.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        print(f"[Node {node_id}] GPU info:\n{gpu_check.stdout.strip()}")
        if gpu_check.stderr.strip():
            print(f"[Node {node_id}] nvidia-smi stderr: {gpu_check.stderr.strip()}")
    except Exception as e:
        print(f"[Node {node_id}] nvidia-smi failed: {e}")

    with ThreadPoolExecutor(max_workers=gpus_per_node) as pool:
        futures = [
            pool.submit(run_gpu, gpu_id, tasks, model_path, split, run_id,
                        node_port, n_episodes, n_envs, n_action_steps,
                        video_dir, embodiment_tag, data_config)
            for gpu_id, tasks in enumerate(my_partitions) if tasks
        ]
        for f in futures:
            gpu_id, rc = f.result()
            print(f"[GPU {gpu_id}] {'OK' if rc == 0 else f'FAILED (rc={rc})'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target"])
    parser.add_argument("--task_set", type=str, nargs="+", default=["atomic_seen", "composite_seen", "composite_unseen"])
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--n_envs", type=int, default=5)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--qos", type=str, default="h200_unicorns_high")
    parser.add_argument("--time_hours", type=int, default=13)
    parser.add_argument("--cpus_per_gpu", type=int, default=20)
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="panda_omron")
    parser.add_argument("--exclude_nodes", type=str, default=None,
                        help="Comma-separated list of nodes to exclude (e.g. h200-085-038,h200-xxx).")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).resolve())

    tasks = resolve_tasks(args.task_set)
    n_gpus = args.n_nodes * args.gpus_per_node
    partitions = partition_tasks(tasks, n_gpus)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_port = random.randint(10000, 50000)

    print(f"Tasks: {len(tasks)}, Nodes: {args.n_nodes}, GPUs: {n_gpus}, Episodes: {args.n_episodes}")
    for i, part in enumerate(partitions):
        h = sum(get_task_horizon(t) for t in part)
        print(f"  GPU {i:2d}: {len(part):2d} tasks, horizon={h:6d}  {part}")

    log_dir = os.path.join(args.video_dir, run_id, "logs")
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_partition="learn",
        slurm_qos=args.qos,
        slurm_account="unicorns",
        slurm_gres=f"gpu:{args.gpus_per_node}",
        slurm_cpus_per_task=args.gpus_per_node * args.cpus_per_gpu,
        slurm_mem="256G",
        timeout_min=args.time_hours * 60,
        slurm_job_name=f"gr00t_eval_{run_id}",
        slurm_additional_parameters={
            "nodes": args.n_nodes,
            "ntasks-per-node": 1,
            **({
                "exclude": args.exclude_nodes,
            } if args.exclude_nodes else {}),
        },
    )

    job = executor.submit(
        node_eval, partitions, args.gpus_per_node, model_path, args.split,
        run_id, base_port, args.n_episodes, args.n_envs, args.n_action_steps,
        args.video_dir, args.embodiment_tag, args.data_config,
    )
    print(f"Submitted SLURM job {job.job_id}")


if __name__ == "__main__":
    main()
