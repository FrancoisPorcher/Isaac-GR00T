"""
Gemma 4 Failure Analysis — Analyze robot rollout videos with Gemma 4 26B-A4B.

Usage:
    uv run python gemma4/analyze_failures.py --video_path <path_to_video.mp4>
    uv run python gemma4/analyze_failures.py --video_dir <dir_with_videos> --filter failure
"""

import argparse
import json
import re
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "google/gemma-4-26B-A4B-it"

ANALYSIS_PROMPT = """\
You are analyzing a robot manipulation rollout in a kitchen simulation (RoboCasa).
The video shows 3 camera views side-by-side: left view, right view, and wrist camera (eye-in-hand).

**Task instruction**: {task_description}

Analyze this rollout and answer:

1. **Success or failure?** Did the robot complete the task?
2. **Progress score** (0.0 to 1.0): How far did the robot get toward completing the task?
3. **Failure point**: At what moment did things go wrong?
4. **Root cause**: What specifically went wrong? Choose from:
   - Object distraction (attended to wrong object)
   - Grasp failure (missed the grasp)
   - Wrong interaction point (e.g., pressed wrong button)
   - No retry after failure (repeated the same failing strategy)
   - Spatial error (wrong position/angle of approach)
   - Other (describe)
5. **Spatial detail**: Describe the PRECISE spatial relationship between the robot's gripper and the target at the moment of failure. Be quantitative:
   - Estimate the gripper's approach angle in degrees (e.g., "~30° from perpendicular")
   - Specify the offset direction and approximate magnitude (e.g., "gripper tip is ~2cm to the left and ~1cm below the button center")
   - Describe the gripper orientation (e.g., "gripper is rotated ~45° clockwise relative to the panel surface")
   - Reference specific visual landmarks (e.g., "the gripper is level with the '5' key, but the START button is 3 keys to the right")
6. **Corrective suggestion**: What EXACT motion correction should the robot apply? Be specific about direction, magnitude, and angle changes (e.g., "rotate wrist 30° counterclockwise, then translate 2cm right and 1cm up").
7. **Dense progress timeline**: Break the video into ~5 phases and rate progress at each phase (0.0-1.0).

Respond in JSON format:
{{
    "success": false,
    "progress_score": 0.3,
    "failure_point": "Around 5 seconds in, when the robot...",
    "root_cause": "wrong_interaction_point",
    "root_cause_detail": "The robot pressed the left button instead of the right one",
    "spatial_detail": {{
        "approach_angle_deg": 30,
        "approach_angle_description": "Gripper approaches ~30° from perpendicular to the panel, tilted downward from the left",
        "offset_from_target": "Gripper tip is ~2cm to the left and ~1cm below the START button",
        "gripper_orientation": "Gripper fingers are horizontal, parallel to the counter surface",
        "visual_landmarks": "Gripper is level with the number '5' key; START button is in the bottom-right of the keypad"
    }},
    "corrective_suggestion": "Rotate wrist 30° counterclockwise to face the panel perpendicularly, then translate 2cm right and 1cm up to center on the START button",
    "progress_timeline": [
        {{"phase": "0-2s", "description": "Robot navigates toward the target object", "score": 0.2}},
        {{"phase": "2-4s", "description": "Robot reaches for the object", "score": 0.4}}
    ]
}}
"""


def extract_task_description(video_path: Path) -> str:
    """Extract task description from filename: {success|failure}_{TaskName}_{desc}_try_{N}.mp4"""
    match = re.match(r"(?:success|failure)_(\w+?)_(.*?)_try_\d+", video_path.stem)
    if match:
        return f"{match.group(1)}: {match.group(2).replace('_', ' ')}"
    return video_path.stem.replace("_", " ")


def sample_frames(video_path: str, n_frames: int = 16) -> list[Image.Image]:
    """Sample evenly-spaced frames from a video using PyAV."""
    container = av.open(video_path)
    total = container.streams.video[0].frames or 250
    targets = set(np.linspace(0, total - 1, n_frames, dtype=int).tolist())
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in targets:
            frames.append(frame.to_image())
        if i > max(targets):
            break
    container.close()
    return frames


def generate(model, processor, messages: list[dict], max_new_tokens: int = 1024) -> str:
    """Run inference on a list of chat messages and return generated text."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def analyze_video(model, processor, video_path: Path, n_frames: int = 16) -> dict:
    """Analyze a rollout video: sample frames with PyAV, send as images to Gemma 4."""
    task_description = extract_task_description(video_path)
    frames = sample_frames(str(video_path), n_frames)
    prompt = ANALYSIS_PROMPT.format(task_description=task_description)

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": prompt}],
    }]

    raw = generate(model, processor, messages)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    result = json.loads(json_match.group()) if json_match else {"raw_response": raw}
    try_match = re.search(r"_try_(\d+)$", video_path.stem)
    try_number = int(try_match.group(1)) if try_match else None
    result["video_path"] = str(video_path)
    result["try_id"] = f"try_{try_number:03d}" if try_number is not None else None
    result["try_number"] = try_number
    result["task_description"] = task_description
    result["n_frames"] = len(frames)
    return result


def find_videos(video_dir: str, filter_type: str = "failure") -> list[Path]:
    patterns = {"failure": "failure_*.mp4", "success": "success_*.mp4", "all": "*.mp4"}
    return sorted(Path(video_dir).rglob(patterns[filter_type]))


def main():
    parser = argparse.ArgumentParser(description="Analyze robot rollout videos with Gemma 4")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--filter", type=str, default="failure", choices=["failure", "success", "all"])
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--max_videos", type=int, default=10)
    parser.add_argument("--output", type=str, default="gemma4/analysis_results.json")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    args = parser.parse_args()

    assert args.video_path or args.video_dir, "Provide --video_path or --video_dir"

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
    )
    print("Model loaded on", model.device)

    if args.video_path:
        videos = [Path(args.video_path)]
    else:
        videos = find_videos(args.video_dir, args.filter)[:args.max_videos]
        assert videos, f"No {args.filter} videos found in {args.video_dir}"
    print(f"Analyzing {len(videos)} video(s)")

    results = []
    for i, video_path in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {video_path.name}")
        result = analyze_video(model, processor, video_path, args.n_frames)
        results.append(result)
        print(f"  Task: {result['task_description']}")
        score = result.get("progress_score")
        if score is not None:
            print(f"  Progress: {score} | Cause: {result.get('root_cause', '?')}")
            print(f"  Spatial: {result.get('spatial_detail', '?')}")
            print(f"  Fix: {result.get('corrective_suggestion', '?')}")
        else:
            print(f"  Response: {result.get('raw_response', '')[:200]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
