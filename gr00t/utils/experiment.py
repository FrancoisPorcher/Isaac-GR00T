# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(self, run_name: str, exp_cfg_dir: Path | None = None):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)


def _partition_tasks(tasks: list[str], n_workers: int) -> list[list[str]]:
    """Greedy bin-packing of tasks by horizon across workers."""
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    tasks_with_horizon = [(t, get_task_horizon(t)) for t in tasks]
    tasks_with_horizon.sort(key=lambda x: x[1], reverse=True)
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for task, horizon in tasks_with_horizon:
        min_idx = loads.index(min(loads))
        buckets[min_idx].append(task)
        loads[min_idx] += horizon
    return buckets


def _run_eval_node(
    partitions,
    gpus_per_node,
    model_path,
    split,
    base_port,
    n_episodes,
    n_envs,
    video_dir,
    embodiment_tag,
    data_config,
):
    """Run distributed eval on one node with ThreadPoolExecutor. Called by submitit."""
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    worker_script = str(repo_root / "scripts" / "run_eval_distributed.py")

    def run_gpu(gpu_id, tasks):
        if not tasks:
            return gpu_id, 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["MUJOCO_GL"] = "egl"
        cmd = [
            sys.executable,
            worker_script,
            "--tasks",
            *tasks,
            "--port",
            str(base_port + gpu_id),
            "--run_id",
            "eval",
            "--model_path",
            model_path,
            "--split",
            split,
            "--n_episodes",
            str(n_episodes),
            "--n_envs",
            str(n_envs),
            "--n_action_steps",
            "16",
            "--embodiment_tag",
            embodiment_tag,
            "--data_config",
            data_config,
            "--video_dir",
            video_dir,
        ]
        result = subprocess.run(cmd, env=env, cwd=str(repo_root))
        return gpu_id, result.returncode

    with ThreadPoolExecutor(max_workers=gpus_per_node) as pool:
        futures = [
            pool.submit(run_gpu, gpu_id, partitions[gpu_id])
            for gpu_id in range(gpus_per_node)
        ]
        for f in futures:
            gpu_id, rc = f.result()
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[Eval GPU {gpu_id}] {status}")


class SimulationEvalCallback(TrainerCallback):
    """Asynchronously launches simulation evaluations on a separate node via submitit
    whenever a checkpoint is saved. Polls for completion and logs results to wandb."""

    def __init__(
        self,
        tasks: list[str],
        eval_every_n_saves: int = 1,
        n_episodes: int = 10,
        n_envs: int = 5,
        split: str = "target",
        data_config: str = "panda_omron",
        embodiment_tag: str = "new_embodiment",
        eval_qos: str = "h200_dev",
        gpus_per_node: int = 8,
        time_hours: int = 4,
    ):
        self.tasks = tasks
        self.eval_every_n_saves = eval_every_n_saves
        self.n_episodes = n_episodes
        self.n_envs = n_envs
        self.split = split
        self.data_config = data_config
        self.embodiment_tag = embodiment_tag
        self.eval_qos = eval_qos
        self.gpus_per_node = gpus_per_node
        self.time_hours = time_hours

        self._save_count = 0
        self._active_eval = None  # {"job": submitit.Job, "step": int, "eval_dir": Path, "start_time": float}
        self._wandb_metric_defined = False

        import wandb as _wandb

        self._wandb = _wandb

    def _define_wandb_metric(self):
        if self._wandb_metric_defined:
            return
        if self._wandb.run is not None:
            self._wandb.define_metric("eval/*", step_metric="eval_step")
            self._wandb_metric_defined = True

    def _parse_eval_results(self, eval_dir: Path) -> dict[str, float] | None:
        """Parse stats.json from eval outputs. Returns None if any task is missing."""
        evals_dir = eval_dir / "eval" / "evals"
        if not evals_dir.exists():
            return None
        results = {}
        for task in self.tasks:
            stats_path = evals_dir / task / "stats.json"
            if not stats_path.exists():
                return None
            with open(stats_path) as f:
                data = json.load(f)
            results[f"eval/{task}_sr"] = data["success_rate"]
        if results:
            results["eval/mean_sr"] = sum(results.values()) / len(results)
        return results

    def _log_results(self, results: dict[str, float], step: int):
        """Log eval results to wandb and stdout."""
        self._define_wandb_metric()
        print(f"\n[SimulationEvalCallback] Results for step {step}:")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.3f}")
        if self._wandb.run is not None:
            self._wandb.log({"eval_step": step, **results})

    def _try_harvest(self) -> bool:
        """Check if active eval finished. If so, parse and log results. Returns True if harvested."""
        if self._active_eval is None:
            return False

        job = self._active_eval["job"]
        step = self._active_eval["step"]

        # Check if job failed
        if job.done():
            exception = job.exception()
            if exception is not None:
                print(
                    f"[SimulationEvalCallback] Eval job for step {step} FAILED: {exception}"
                )
                self._active_eval = None
                return True

        results = self._parse_eval_results(self._active_eval["eval_dir"])
        if results is not None:
            self._log_results(results, step)
            elapsed = time.time() - self._active_eval["start_time"]
            print(
                f"[SimulationEvalCallback] Eval for step {step} completed in {elapsed:.0f}s"
            )
            self._active_eval = None
            return True
        return False

    def _submit_eval(self, checkpoint_dir: Path, eval_dir: Path, step: int):
        """Submit a distributed eval job via submitit."""
        import random

        import submitit

        eval_dir.mkdir(parents=True, exist_ok=True)
        log_dir = eval_dir / "logs"

        n_workers = min(self.gpus_per_node, len(self.tasks))
        partitions = _partition_tasks(self.tasks, n_workers)
        partitions.extend([] for _ in range(self.gpus_per_node - len(partitions)))

        base_port = random.randint(10000, 50000)

        executor = submitit.AutoExecutor(folder=str(log_dir))
        executor.update_parameters(
            slurm_qos=self.eval_qos,
            slurm_account="unicorns",
            slurm_gres=f"gpu:{self.gpus_per_node}",
            slurm_cpus_per_task=self.gpus_per_node * 20,
            slurm_mem="256G",
            timeout_min=self.time_hours * 60,
            slurm_job_name=f"eval-step-{step}",
            slurm_additional_parameters={"nodes": 1, "ntasks-per-node": 1},
        )

        job = executor.submit(
            _run_eval_node,
            partitions,
            self.gpus_per_node,
            str(checkpoint_dir),
            self.split,
            base_port,
            self.n_episodes,
            self.n_envs,
            str(eval_dir),
            self.embodiment_tag,
            self.data_config,
        )
        return job

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self._save_count += 1
        if self._save_count % self.eval_every_n_saves != 0:
            return

        self._try_harvest()
        if self._active_eval is not None:
            print(
                f"[SimulationEvalCallback] Eval for step {self._active_eval['step']} still running "
                f"({time.time() - self._active_eval['start_time']:.0f}s elapsed). Skipping eval for step {state.global_step}."
            )
            return

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        eval_dir = Path(args.output_dir) / "evals" / f"step-{state.global_step}"

        job = self._submit_eval(checkpoint_dir, eval_dir, state.global_step)

        self._active_eval = {
            "job": job,
            "step": state.global_step,
            "eval_dir": eval_dir,
            "start_time": time.time(),
        }
        print(
            f"[SimulationEvalCallback] Submitted eval job {job.job_id} for step {state.global_step} "
            f"({len(self.tasks)} tasks, {self.n_episodes} episodes each, {self.gpus_per_node} GPUs)"
        )

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._try_harvest()

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for any active eval, then eval the final checkpoint if it wasn't evaluated."""
        if not state.is_world_process_zero:
            return

        # Wait for active eval to finish
        if self._active_eval is not None:
            print(
                f"[SimulationEvalCallback] Waiting for eval job {self._active_eval['job'].job_id} to finish..."
            )
            deadline = time.time() + self.time_hours * 3600
            while time.time() < deadline:
                if self._try_harvest():
                    break
                time.sleep(30)
            else:
                print(
                    f"[SimulationEvalCallback] Eval job {self._active_eval['job'].job_id} did not finish. "
                    f"Check manually: eval_dir={self._active_eval['eval_dir']}"
                )
                return

        # Eval the final checkpoint if it wasn't already evaluated
        final_checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        final_eval_dir = Path(args.output_dir) / "evals" / f"step-{state.global_step}"
        if final_checkpoint.exists() and not self._parse_eval_results(final_eval_dir):
            print(
                f"[SimulationEvalCallback] Launching final eval for step {state.global_step}..."
            )
            job = self._submit_eval(final_checkpoint, final_eval_dir, state.global_step)
            self._active_eval = {
                "job": job,
                "step": state.global_step,
                "eval_dir": final_eval_dir,
                "start_time": time.time(),
            }
            print(
                f"[SimulationEvalCallback] Final eval job {job.job_id} submitted. Waiting..."
            )
            deadline = time.time() + self.time_hours * 3600
            while time.time() < deadline:
                if self._try_harvest():
                    return
                time.sleep(30)
            print(
                f"[SimulationEvalCallback] Final eval job {job.job_id} did not finish. "
                f"Check manually: eval_dir={final_eval_dir}"
            )
