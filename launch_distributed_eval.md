```bash
cd /storage/home/francoisporcher/Isaac-GR00T && .venv/bin/python scripts/launch_eval.py \
  --model_path /storage/home/francoisporcher/Isaac-GR00T/checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000 \
  --video_dir /checkpoint/unicorns/francoisporcher/gr00t/full_eval_4_nodes \
  --n_nodes 4 \
  --split target \
  --n_episodes 30 \
  --n_envs 5 \
  --qos h200_unicorns_high \
  --time_hours 16 \
  --exclude_nodes h200-085-038
```
