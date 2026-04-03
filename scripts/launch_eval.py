#!/usr/bin/env python3
"""Parallel evaluation launcher for GR00T on RoboCasa365.

Distributes tasks across multiple GPUs (and optionally multiple SLURM nodes).
Each GPU runs an independent simulation pipeline via run_eval_distributed.py.

Modes:
  --local        Run on current node using available GPUs (default for 1 node).
  (default)      Submit a single multi-node SLURM job via sbatch.
  --status       Print progress from stats.json files and exit.
"""

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

TARGET_SPLITS = ["atomic_seen", "composite_seen", "composite_unseen"]

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = str(REPO_ROOT / "scripts" / "run_eval_distributed.py")
VENV_PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")


def resolve_tasks(task_sets: list[str]) -> list[str]:
    """Resolve task set names to a sorted, deduplicated list of task names."""
    tasks = []
    for ts in task_sets:
        tasks.extend(TASK_SET_REGISTRY[ts])
    return sorted(set(tasks))


def partition_tasks(tasks: list[str], n_workers: int) -> list[list[str]]:
    """Greedy bin-packing by estimated horizon.

    Sorts tasks by horizon descending, assigns each to the worker with the
    lowest cumulative horizon. Balances wall-clock time despite 25x variance
    in task durations (200-4800 steps).
    """
    if n_workers > len(tasks):
        raise ValueError(
            f"n_workers ({n_workers}) > n_tasks ({len(tasks)}). Reduce --gpus_per_node or provide more tasks."
        )

    tasks_with_horizon = [(t, get_task_horizon(t)) for t in tasks]
    tasks_with_horizon.sort(key=lambda x: x[1], reverse=True)

    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers

    for task, horizon in tasks_with_horizon:
        min_idx = loads.index(min(loads))
        buckets[min_idx].append(task)
        loads[min_idx] += horizon

    return buckets


def gpu_worker(
    worker_id: int,
    tasks: list[str],
    model_path: str,
    split: str,
    run_id: str,
    base_port: int,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    video_dir: str | None,
    embodiment_tag: str,
    data_config: str,
) -> tuple[int, int, list[str]]:
    """Run one GPU's evaluation as a subprocess."""
    port = base_port + worker_id
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    env["MUJOCO_GL"] = "egl"
    ld_path = "/home/francoisporcher/miniforge3/lib:/usr/lib64"
    if env.get("LD_LIBRARY_PATH"):
        ld_path += ":" + env["LD_LIBRARY_PATH"]
    env["LD_LIBRARY_PATH"] = ld_path

    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

    cmd = [
        python,
        WORKER_SCRIPT,
        "--tasks",
        *tasks,
        "--port",
        str(port),
        "--run_id",
        run_id,
        "--model_path",
        model_path,
        "--split",
        split,
        "--n_episodes",
        str(n_episodes),
        "--n_envs",
        str(n_envs),
        "--n_action_steps",
        str(n_action_steps),
        "--embodiment_tag",
        embodiment_tag,
        "--data_config",
        data_config,
    ]
    if video_dir:
        cmd.extend(["--video_dir", video_dir])

    print(f"[Worker {worker_id}] GPU={worker_id} port={port} tasks={tasks}")
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return (worker_id, result.returncode, tasks)


def node_job(
    gpu_task_partitions: list[list[str]],
    model_path: str,
    split: str,
    run_id: str,
    base_port: int,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    video_dir: str | None,
    embodiment_tag: str,
    data_config: str,
) -> dict[int, dict]:
    """Run all GPU workers on one node via ThreadPoolExecutor."""
    non_empty = [(i, parts) for i, parts in enumerate(gpu_task_partitions) if parts]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(non_empty)) as pool:
        futures = {
            pool.submit(
                gpu_worker,
                worker_id=i,
                tasks=parts,
                model_path=model_path,
                split=split,
                run_id=run_id,
                base_port=base_port,
                n_episodes=n_episodes,
                n_envs=n_envs,
                n_action_steps=n_action_steps,
                video_dir=video_dir,
                embodiment_tag=embodiment_tag,
                data_config=data_config,
            ): i
            for i, parts in non_empty
        }

        results = {}
        for future in concurrent.futures.as_completed(futures):
            wid, rc, tasks = future.result()
            results[wid] = {"tasks": tasks, "returncode": rc}
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[Worker {wid}] {status}")

    return results


def print_status(video_dir: str, split: str):
    """Scan stats.json files and print evaluation progress."""
    evals_dir = os.path.join(video_dir, "evals", split)
    if not os.path.isdir(evals_dir):
        print(f"No evals directory found at {evals_dir}")
        return

    run_dirs = sorted(os.listdir(evals_dir))
    if not run_dirs:
        print("No evaluation runs found.")
        return

    for run_id in run_dirs:
        run_path = os.path.join(evals_dir, run_id)
        if not os.path.isdir(run_path):
            continue
        task_dirs = sorted(os.listdir(run_path))
        completed = 0
        total = len(task_dirs)
        successes = []
        for task_dir in task_dirs:
            stats_path = os.path.join(run_path, task_dir, "stats.json")
            if os.path.exists(stats_path):
                completed += 1
                try:
                    with open(stats_path) as f:
                        stats = json.load(f)
                    successes.append((task_dir, stats.get("success_rate", -1)))
                except Exception:
                    pass

        print(f"\nRun: {run_id}  [{completed}/{total} tasks completed]")
        for task, sr in successes:
            print(f"  {task}: {sr:.2f}")
        if successes:
            avg = sum(s for _, s in successes) / len(successes)
            print(f"  --- Average: {avg:.2f} ---")


def run_slurm_worker():
    """Entry point when running inside a SLURM job (called via srun on each node).

    Reads partitions from the JSON file passed via EVAL_PARTITIONS_FILE env var,
    selects this node's partition using SLURM_NODEID, and runs gpu workers.
    """
    partitions_file = os.environ["EVAL_PARTITIONS_FILE"]
    with open(partitions_file) as f:
        config = json.load(f)

    node_id = int(os.environ.get("SLURM_NODEID", 0))
    gpus_per_node = config["gpus_per_node"]
    start = node_id * gpus_per_node
    end = start + gpus_per_node
    node_partitions = config["partitions"][start:end]

    job_id = os.environ.get("SLURM_JOB_ID", "0")
    base_port = int(job_id) % 10000 + 10000 + node_id * 100

    print(f"[Node {node_id}] Running {len(node_partitions)} GPU workers (GPUs {start}-{end - 1})")

    results = node_job(
        gpu_task_partitions=node_partitions,
        model_path=config["model_path"],
        split=config["split"],
        run_id=config["run_id"],
        base_port=base_port,
        n_episodes=config["n_episodes"],
        n_envs=config["n_envs"],
        n_action_steps=config["n_action_steps"],
        video_dir=config["video_dir"],
        embodiment_tag=config["embodiment_tag"],
        data_config=config["data_config"],
    )

    failed = [wid for wid, info in results.items() if info["returncode"] != 0]
    if failed:
        print(f"[Node {node_id}] Failed workers: {failed}")
        sys.exit(1)
    print(f"[Node {node_id}] All workers completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation launcher for GR00T on RoboCasa365.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--task_set",
        type=str,
        nargs="+",
        default=None,
        help="Task set name(s) to evaluate. Defaults to all 3 target splits.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="target",
        choices=["pretrain", "target"],
        help="Split to evaluate on.",
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="Number of SLURM nodes.")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="GPUs per node.")
    parser.add_argument("--n_episodes", type=int, default=50, help="Episodes per task.")
    parser.add_argument("--n_envs", type=int, default=5, help="Parallel environments per task.")
    parser.add_argument("--n_action_steps", type=int, default=16, help="Action steps per env step.")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save videos.")
    parser.add_argument("--qos", type=str, default="h200_unicorns_high", help="SLURM QoS.")
    parser.add_argument("--time_hours", type=int, default=13, help="SLURM time limit (hours).")
    parser.add_argument("--local", action="store_true", help="Run locally on current node (skip SLURM).")
    parser.add_argument("--status", action="store_true", help="Print progress from stats.json files and exit.")
    parser.add_argument("--slurm_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="panda_omron")

    args = parser.parse_args()

    if args.slurm_worker:
        run_slurm_worker()
        return

    model_path = str(Path(args.model_path).resolve())
    video_dir = args.video_dir or model_path

    if args.status:
        print_status(video_dir, args.split)
        return

    task_sets = args.task_set or TARGET_SPLITS
    tasks = resolve_tasks(task_sets)
    n_total_gpus = args.n_nodes * args.gpus_per_node

    if n_total_gpus > len(tasks):
        n_total_gpus = len(tasks)
        print(f"Warning: more GPUs than tasks. Using {n_total_gpus} GPUs.")

    partitions = partition_tasks(tasks, n_total_gpus)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_port = random.randint(10000, 60000)

    print(f"\n{'=' * 60}")
    print("GR00T Parallel Evaluation")
    print(f"{'=' * 60}")
    print(f"Model:     {model_path}")
    print(f"Video dir: {video_dir}")
    print(f"Split:     {args.split}")
    print(f"Task sets: {task_sets}")
    print(f"Tasks:     {len(tasks)}")
    print(f"Episodes:  {args.n_episodes}")
    print(f"Workers:   {n_total_gpus} ({args.n_nodes} node(s) x {args.gpus_per_node} GPU(s))")
    print(f"Run ID:    {run_id}")
    print(f"Base port: {base_port}")
    print("\nTask assignment:")
    for i, part in enumerate(partitions):
        horizons = [get_task_horizon(t) for t in part]
        total_h = sum(horizons)
        print(f"  GPU {i:2d}: {len(part):2d} tasks, horizon={total_h:6d}  {part}")
    print(f"{'=' * 60}\n")

    if args.local:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            base_port = int(slurm_job_id) % 10000 + 10000

        node_partitions = partitions[: args.gpus_per_node]
        results = node_job(
            gpu_task_partitions=node_partitions,
            model_path=model_path,
            split=args.split,
            run_id=run_id,
            base_port=base_port,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            video_dir=video_dir,
            embodiment_tag=args.embodiment_tag,
            data_config=args.data_config,
        )
        failed = [wid for wid, info in results.items() if info["returncode"] != 0]
        if failed:
            print(f"\nFailed workers: {failed}")
            sys.exit(1)
        print("\nAll workers completed successfully.")
    else:
        log_dir = os.path.join(video_dir, "slurm_logs", run_id)
        os.makedirs(log_dir, exist_ok=True)

        config = {
            "partitions": [p for p in partitions],
            "gpus_per_node": args.gpus_per_node,
            "model_path": model_path,
            "split": args.split,
            "run_id": run_id,
            "n_episodes": args.n_episodes,
            "n_envs": args.n_envs,
            "n_action_steps": args.n_action_steps,
            "video_dir": video_dir,
            "embodiment_tag": args.embodiment_tag,
            "data_config": args.data_config,
        }
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
        launcher_script = str(Path(__file__).resolve())

        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=gr00t_eval_{run_id}
#SBATCH --nodes={args.n_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={args.gpus_per_node}
#SBATCH --cpus-per-task={args.gpus_per_node * 10}
#SBATCH --mem=256G
#SBATCH --time={args.time_hours}:00:00
#SBATCH --qos={args.qos}
#SBATCH --account=unicorns
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err

export EVAL_PARTITIONS_FILE="{config_path}"
export LD_LIBRARY_PATH="/home/francoisporcher/miniforge3/lib:/usr/lib64:$LD_LIBRARY_PATH"
export MUJOCO_GL=egl

srun {python} {launcher_script} --model_path {model_path} --split {args.split} --slurm_worker
"""
        sbatch_path = os.path.join(log_dir, "submit.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(sbatch_script)

        result = subprocess.run(
            ["sbatch", sbatch_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"sbatch failed: {result.stderr}")
            sys.exit(1)

        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted SLURM job {job_id} ({args.n_nodes} node(s), {n_total_gpus} GPUs)")
        print(f"Logs: {log_dir}/slurm_{job_id}.out")
        print(
            f"\nMonitor: {python} {launcher_script} --model_path {model_path} --split {args.split} --video_dir {video_dir} --status"
        )


if __name__ == "__main__":
    main()
