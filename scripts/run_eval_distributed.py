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

import argparse
import os
import json
import numpy as np
import threading
import time
from datetime import datetime
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def run_server(data_config, model_path, embodiment_tag, port):
    data_config = DATA_CONFIG_MAP[data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )

    server = RobotInferenceServer(policy, port=port)
    server.run()


def run_client(host, port, env_names, video_dir, split, n_episodes, n_envs, n_action_steps, run_id):
    simulation_client = SimulationInferenceClient(host=host, port=port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    for env_name in env_names:
        this_video_dir = os.path.join(video_dir, "evals", split, run_id, env_name)

        stats_path = os.path.join(this_video_dir, "stats.json")
        if os.path.exists(stats_path):
            print(f"{env_name} stats already exists. skipping.")
            continue
        horizon = get_task_horizon(env_name)
        config = SimulationConfig(
            env_name=f"robocasa/{env_name}",
            split=split,
            n_episodes=n_episodes,
            n_envs=n_envs,
            video=VideoConfig(video_dir=this_video_dir, env_name=env_name, n_envs=n_envs),
            multistep=MultiStepConfig(
                n_action_steps=n_action_steps,
                max_episode_steps=horizon,
            ),
        )

        print(f"Running simulation for {env_name}...")
        try:
            env_name, episode_successes = simulation_client.run_simulation(config)
        except Exception as e:
            print("Exception!", e)
            continue

        success_rate = np.mean(episode_successes)

        print(f"Results for {env_name}:")
        print(f"Success rate: {success_rate:.2f}")

        with open(stats_path, "w") as f:
            stats = {
                "num_episodes": len(episode_successes),
                "success_rate": success_rate,
            }
            json.dump(stats, f, indent=4)
        print(f"saved stats to {stats_path}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        default="panda_omron",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        nargs="+",
        help="Name of the task soup(s). Mutually exclusive with --tasks.",
        default=None,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Individual task names. Mutually exclusive with --task_set.",
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to evaluate on. Can be either pretrain or target.",
        choices=["pretrain", "target"],
        required=True,
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="Shared run ID across workers. Auto-generated if not provided.",
        default=None,
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument("--host", type=str, help="Host address for the server.", default="localhost")
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=50)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=5)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    args = parser.parse_args()

    if args.task_set and args.tasks:
        parser.error("--task_set and --tasks are mutually exclusive.")
    if not args.task_set and not args.tasks:
        parser.error("Exactly one of --task_set or --tasks must be provided.")

    if args.task_set:
        env_names = []
        for task_set in args.task_set:
            env_names += TASK_SET_REGISTRY[task_set]
        env_names = sorted(set(env_names))
    else:
        env_names = args.tasks

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    server_thread = threading.Thread(
        target=run_server,
        args=(args.data_config, args.model_path, args.embodiment_tag, args.port),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)
    run_client(
        host=args.host,
        port=args.port,
        env_names=env_names,
        video_dir=args.video_dir or args.model_path,
        split=args.split,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        run_id=run_id,
    )
