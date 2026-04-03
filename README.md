## NVIDIA Isaac GR00T
This is the NVIDIA Isaac GR00T fork repo for running RoboCasa benchmark experiments. This fork is based on the original [GR00T code](https://github.com/NVIDIA/Isaac-GR00T) from NVIDIA. Our fork supports training for **GR00T N1.5**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd groot
pip install -e .
```

### Key files
- Training: [scripts/gr00t_finetune.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py)
- Evaluation: [scripts/run_eval.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/run_eval.py)

### Experiment workflow
```
# train model
python scripts/gr00t_finetune.py \
--output-dir <experiment-path> \
--dataset_soup <dataset-soup> \
--max_steps <num-training-steps>

# evaluate model
python scripts/run_eval.py \
--model_path <checkpoint-path> \
--task_set <task-set> \
--split <split>

# report evaluation results
python gr00t/eval/get_eval_stats.py \
--dir <checkpoint-path>
```


# Francois modifications

## Installation of Robocasa365

```
uv sync --group robocasa365
```

## Robocasa datasets

This env uses **lerobot 0.4.0** which uses dataset format **v3.0** ([release blog](https://huggingface.co/blog/lerobot-release-v040)).

```
/checkpoint/unicorns/shared/datasets/original_robocasa_v30
```

Older v2.1 datasets (for lerobot <0.4.0):

```
/checkpoint/unicorns/shared/datasets/original_robocasa
```

## Where to save checkpoints

Save all large artifacts (model checkpoints, evaluation videos, training outputs) to the checkpoint partition rather than home:

```
/checkpoint/unicorns/francoisporcher/gr00t/
├── debug/                # debug/dev training runs
├── full_eval_single_node/ # 1-node (8 GPU) evaluation outputs + videos
└── full_eval_2_nodes/     # 2-node (16 GPU) evaluation outputs + videos
```

**Why?** Home (`/storage/home`) is backed by shared NFS and currently holds 785G. The checkpoint partition has more capacity and is designed for large transient artifacts.

**Evaluation example** — point `--video_dir` and `--model_path` to checkpoint storage:

```bash
uv run python scripts/run_eval.py \
  --model_path checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000 \
  --task_set atomic_seen composite_seen composite_unseen \
  --split target \
  --n_episodes 30 \
  --n_envs 5 \
  --video_dir /checkpoint/unicorns/francoisporcher/gr00t/full_eval_single_node
```

**Storage estimate per full evaluation run** (50 tasks × 30 trials = 1,500 rollouts): ~3–5 GB with videos.

## Model checkpoint

The pretrained GR00T N1.5 checkpoint used for evaluation is at:

```
checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000
```

## Parallel Evaluation

The launcher distributes tasks across GPUs using greedy bin-packing by task horizon for balanced wall-clock time.

```bash
# Single node (8 GPUs)
uv run scripts/launch_eval.py --model_path <path> --split target

# Multi-node (2 nodes × 8 GPUs)
uv run scripts/launch_eval.py --model_path <path> --split target --n_nodes 2

# Local debug (1 GPU, 2 episodes)
uv run scripts/launch_eval.py --model_path <path> --split target --local --gpus_per_node 1 --n_episodes 2

# Check progress
uv run scripts/launch_eval.py --model_path <path> --split target --status

# Collect results
uv run gr00t/eval/get_eval_stats.py --dir <model_path>
```
