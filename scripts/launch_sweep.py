#!/usr/bin/env python3
"""Launch a sweep of TurnOnMicrowave fine-tuning runs over learning rate × batch size × tune_visual."""

import subprocess
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = "/checkpoint/unicorns/shared/datasets/original_robocasa/v1.0/target/atomic/TurnOnMicrowave/20250813/lerobot"
BASE_MODEL = str(
    REPO_ROOT / "checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000"
)
OUTPUT_ROOT = REPO_ROOT / "gr00t_checkpoints" / "fine_tuning"

SWEEP_CONFIG = {
    "learning_rate": [1e-5, 3e-5, 1e-4],
    "batch_size_per_gpu": [8, 16, 32],
    "tune_visual": [False, True],
}

FIXED_CONFIG = {
    "max_steps": 60000,
    "save_steps": 10000,
    "num_gpus": 8,
    "eval_tasks": [
        "TurnOnMicrowave",
        "CloseFridge",
        "OpenCabinet",
        "PickPlaceSinkToCounter",
        "PickPlaceCounterToStove",
    ],
    "eval_n_episodes": 30,
    "eval_n_envs": 5,
    "eval_qos": "h200_unicorns_high",
    "report_to": "wandb",
    "wandb_project": "gr00t-finetuning",
}

SLURM_CONFIG = {
    "qos": "h200_unicorns_high",
    "account": "unicorns",
    "gres": "gpu:8",
    "cpus_per_task": 160,
    "mem": "256G",
    "time_hours": 24,
}


def build_run_name(lr: float, bs: int, tune_vis: bool) -> str:
    vis_tag = "vis" if tune_vis else "novis"
    return f"TurnOnMicrowave_lr{lr:.0e}_bs{bs}_{vis_tag}"


def launch_run(lr: float, bs: int, tune_vis: bool, dry_run: bool = False):
    run_name = build_run_name(lr, bs, tune_vis)
    output_dir = OUTPUT_ROOT / run_name

    vis_flag = "--tune-visual" if tune_vis else "--no-tune-visual"

    cmd = [
        "srun",
        f"--qos={SLURM_CONFIG['qos']}",
        f"--account={SLURM_CONFIG['account']}",
        f"--gres={SLURM_CONFIG['gres']}",
        f"--cpus-per-task={SLURM_CONFIG['cpus_per_task']}",
        f"--mem={SLURM_CONFIG['mem']}",
        f"--time={SLURM_CONFIG['time_hours']}:00:00",
        f"--job-name={run_name}",
        "bash",
        "-c",
        (
            f"export MUJOCO_GL=egl && "
            f"{sys.executable} {REPO_ROOT / 'scripts/gr00t_finetune.py'} "
            f"--dataset-path {DATASET_PATH} "
            f"--base-model-path {BASE_MODEL} "
            f"--output-dir {output_dir} "
            f"--learning-rate {lr} "
            f"--batch-size {bs} "
            f"{vis_flag} "
            f"--num-gpus {FIXED_CONFIG['num_gpus']} "
            f"--max-steps {FIXED_CONFIG['max_steps']} "
            f"--save-steps {FIXED_CONFIG['save_steps']} "
            f"--report-to {FIXED_CONFIG['report_to']} "
            f"--wandb-project {FIXED_CONFIG['wandb_project']} "
            f"--wandb-run-name {run_name} "
            f"--eval-tasks {' '.join(FIXED_CONFIG['eval_tasks'])} "
            f"--eval-n-episodes {FIXED_CONFIG['eval_n_episodes']} "
            f"--eval-n-envs {FIXED_CONFIG['eval_n_envs']} "
            f"--eval-qos {FIXED_CONFIG['eval_qos']}"
        ),
    ]

    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"  lr={lr}, bs/gpu={bs}, effective_bs={bs * 8}, tune_visual={tune_vis}")
    print(f"  output: {output_dir}")

    if dry_run:
        print(f"  [DRY RUN] Would submit: {' '.join(cmd[:8])} ...")
        return

    result = subprocess.Popen(cmd)
    print(f"  Submitted (PID {result.pid})")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without submitting"
    )
    args = parser.parse_args()

    lrs = SWEEP_CONFIG["learning_rate"]
    bss = SWEEP_CONFIG["batch_size_per_gpu"]
    visuals = SWEEP_CONFIG["tune_visual"]

    total = len(lrs) * len(bss) * len(visuals)
    print(
        f"Sweep: {len(lrs)} lr × {len(bss)} bs × {len(visuals)} tune_visual = {total} runs"
    )
    print(f"LRs: {lrs}")
    print(f"BSs: {bss} (effective: {[b * 8 for b in bss]})")
    print(f"tune_visual: {visuals}")

    processes = []
    for lr, bs, tune_vis in product(lrs, bss, visuals):
        p = launch_run(lr, bs, tune_vis, dry_run=args.dry_run)
        if p:
            processes.append(p)

    if not args.dry_run and processes:
        print(f"\n{len(processes)} runs submitted. Monitor via: squeue -u $USER")
