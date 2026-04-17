#!/usr/bin/env python3
"""Launch a sweep of fine-tuning runs.

Usage:
    # Single-task sweep (TurnOnMicrowave)
    python scripts/launch_sweep.py --mode single-task --dry-run
    python scripts/launch_sweep.py --mode single-task

    # Multitask sweep (target50)
    python scripts/launch_sweep.py --mode multitask --dry-run
    python scripts/launch_sweep.py --mode multitask

    # Multitask v2: lower LR + weight decay sweep
    python scripts/launch_sweep.py --mode multitask-v2 --dry-run
    python scripts/launch_sweep.py --mode multitask-v2

    # Launch a single run for testing
    python scripts/launch_sweep.py --mode multitask --only target50_lr1e-05_bs8_novis
"""

import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import submitit

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = str(
    REPO_ROOT / "checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000"
)
OUTPUT_ROOT = REPO_ROOT / "gr00t_checkpoints" / "fine_tuning"

ALL_50_TASKS = [
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "CuttingToolSelection",
    "DeliverStraw",
    "GarnishPancake",
    "GatherTableware",
    "GetToastedBread",
    "HeatKebabSandwich",
    "KettleBoiling",
    "LoadDishwasher",
    "MakeIceLemonade",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PackIdenticalLunches",
    "PanTransfer",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "PortionHotDogs",
    "PreSoakPan",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SeparateFreezerRack",
    "SetUpCuttingStation",
    "SlideDishwasherRack",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "WaffleReheat",
    "WashFruitColander",
    "WashLettuce",
    "WeighIngredients",
]

SWEEP_CONFIGS = {
    "multitask": {
        "learning_rate": [1e-5, 3e-5],
        "batch_size_per_gpu": [8, 16],
        "tune_visual": [False, True],
        "weight_decay": [1e-5],
    },
    "multitask-v2": {
        "learning_rate": [5e-6],
        "batch_size_per_gpu": [8, 16],
        "tune_visual": [False, True],
        "weight_decay": [0.0, 1e-5],
    },
    "single-task": {
        "learning_rate": [1e-5, 3e-5, 1e-4],
        "batch_size_per_gpu": [8, 16],
        "tune_visual": [False, True],
        "weight_decay": [1e-5],
    },
}

MODE_CONFIGS = {
    "single-task": {
        "prefix": "TurnOnMicrowave",
        "dataset_flag": "--dataset-path /checkpoint/unicorns/shared/datasets/original_robocasa/v1.0/target/atomic/TurnOnMicrowave/20250813/lerobot",
        "eval_tasks": [
            "TurnOnMicrowave",
            "CloseFridge",
            "OpenCabinet",
            "PickPlaceSinkToCounter",
            "PickPlaceCounterToStove",
        ],
        "max_steps": 60000,
        "save_steps": 10000,
        "eval_time_hours": 4,
    },
    "multitask": {
        "prefix": "target50",
        "dataset_flag": "--dataset-soup target50",
        "eval_tasks": ALL_50_TASKS,
        "max_steps": 120000,
        "save_steps": 20000,
        "eval_time_hours": 24,
    },
    "multitask-v2": {
        "prefix": "target50",
        "dataset_flag": "--dataset-soup target50",
        "eval_tasks": ALL_50_TASKS,
        "max_steps": 120000,
        "save_steps": 20000,
        "eval_time_hours": 24,
    },
}

SLURM_CONFIG = {
    "qos": "h200_unicorns_high",
    "account": "unicorns",
    "gpus_per_node": 8,
    "cpus_per_task": 192,
    "mem_gb": 1800,
    "time_hours": 48,
}

FIXED_CONFIG = {
    "num_gpus": 8,
    "eval_n_episodes": 30,
    "eval_n_envs": 5,
    "eval_qos": "h200_unicorns_high",
    "report_to": "wandb",
    "wandb_project": "gr00t-finetuning",
}


def build_run_name(prefix: str, lr: float, bs: int, tune_vis: bool, wd: float) -> str:
    vis_tag = "vis" if tune_vis else "novis"
    wd_tag = f"_wd{wd:.0e}" if wd != 1e-5 else ""
    return f"{prefix}_lr{lr:.0e}_bs{bs}{wd_tag}_{vis_tag}"


def run_training(
    run_name: str, mode_cfg: dict, lr: float, bs: int, tune_vis: bool, wd: float
):
    """Function executed inside the SLURM job by submitit."""
    output_dir = OUTPUT_ROOT / run_name
    vis_flag = "--tune-visual" if tune_vis else "--no-tune-visual"

    cmd = (
        f"export MUJOCO_GL=egl && "
        f"{sys.executable} {REPO_ROOT / 'scripts/gr00t_finetune.py'} "
        f"{mode_cfg['dataset_flag']} "
        f"--base-model-path {BASE_MODEL} "
        f"--output-dir {output_dir} "
        f"--learning-rate {lr} "
        f"--weight-decay {wd} "
        f"--batch-size {bs} "
        f"{vis_flag} "
        f"--num-gpus {FIXED_CONFIG['num_gpus']} "
        f"--max-steps {mode_cfg['max_steps']} "
        f"--save-steps {mode_cfg['save_steps']} "
        f"--report-to {FIXED_CONFIG['report_to']} "
        f"--wandb-project {FIXED_CONFIG['wandb_project']} "
        f"--wandb-run-name {run_name} "
        f"--eval-tasks {' '.join(mode_cfg['eval_tasks'])} "
        f"--eval-n-episodes {FIXED_CONFIG['eval_n_episodes']} "
        f"--eval-n-envs {FIXED_CONFIG['eval_n_envs']} "
        f"--eval-qos {FIXED_CONFIG['eval_qos']} "
        f"--eval-time-hours {mode_cfg['eval_time_hours']}"
    )

    print(f"[{run_name}] Starting training on {os.environ.get('SLURMD_NODENAME', '?')}")
    print(f"[{run_name}] Command: bash -c '{cmd}'")
    result = subprocess.run(["bash", "-c", cmd])
    print(f"[{run_name}] Finished with return code {result.returncode}")
    return result.returncode


def launch_run(
    executor: submitit.AutoExecutor,
    mode_cfg: dict,
    lr: float,
    bs: int,
    tune_vis: bool,
    wd: float,
    dry_run: bool = False,
):
    run_name = build_run_name(mode_cfg["prefix"], lr, bs, tune_vis, wd)
    output_dir = OUTPUT_ROOT / run_name

    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(
        f"  lr={lr}, bs/gpu={bs}, effective_bs={bs * 8}, tune_visual={tune_vis}, weight_decay={wd}"
    )
    print(f"  dataset: {mode_cfg['dataset_flag']}")
    print(
        f"  eval: {len(mode_cfg['eval_tasks'])} tasks, {FIXED_CONFIG['eval_n_episodes']} episodes"
    )
    print(f"  output: {output_dir}")

    if dry_run:
        print("  [DRY RUN] Would submit via submitit")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    executor.folder = str(output_dir)
    executor.update_parameters(slurm_job_name=run_name)
    job = executor.submit(run_training, run_name, mode_cfg, lr, bs, tune_vis, wd)
    print(f"  Submitted: job_id={job.job_id}, logs={job.paths.stdout}")
    return job


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODE_CONFIGS.keys()), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only", type=str, default=None, help="Launch only this run name"
    )
    args = parser.parse_args()

    mode_cfg = MODE_CONFIGS[args.mode]
    sweep = SWEEP_CONFIGS[args.mode]
    lrs = sweep["learning_rate"]
    bss = sweep["batch_size_per_gpu"]
    visuals = sweep["tune_visual"]
    wds = sweep["weight_decay"]

    total = len(lrs) * len(bss) * len(visuals) * len(wds)
    print(f"Mode: {args.mode} ({mode_cfg['prefix']})")
    print(
        f"Sweep: {len(lrs)} lr × {len(bss)} bs × {len(visuals)} tune_visual × {len(wds)} wd = {total} runs"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(OUTPUT_ROOT))
    executor.update_parameters(
        slurm_partition=None,
        slurm_qos=SLURM_CONFIG["qos"],
        slurm_account=SLURM_CONFIG["account"],
        gpus_per_node=SLURM_CONFIG["gpus_per_node"],
        slurm_cpus_per_task=SLURM_CONFIG["cpus_per_task"],
        slurm_mem=f"{SLURM_CONFIG['mem_gb']}G",
        timeout_min=SLURM_CONFIG["time_hours"] * 60,
        slurm_additional_parameters={"gres": f"gpu:{SLURM_CONFIG['gpus_per_node']}"},
    )

    jobs = []
    for lr, bs, tune_vis, wd in product(lrs, bss, visuals, wds):
        run_name = build_run_name(mode_cfg["prefix"], lr, bs, tune_vis, wd)
        if args.only and run_name != args.only:
            continue
        job = launch_run(executor, mode_cfg, lr, bs, tune_vis, wd, dry_run=args.dry_run)
        if job:
            jobs.append((run_name, job))

    if jobs:
        print(f"\n{'=' * 60}")
        print(f"{len(jobs)} jobs submitted. Monitor via: squeue -u $USER")
        for name, job in jobs:
            print(f"  {name}: job_id={job.job_id}")
