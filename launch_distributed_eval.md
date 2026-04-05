```bash
cd <REPO_ROOT> && uv run scripts/launch_eval.py \
  --model_path <MODEL_CHECKPOINT_PATH> \
  --video_dir <OUTPUT_DIR> \
  --n_nodes 4 \
  --split target \
  --n_episodes 30 \
  --n_envs 5 \
  --qos h200_unicorns_high \
  --time_hours 16 \
  --exclude_nodes <NODES_TO_EXCLUDE>
```
