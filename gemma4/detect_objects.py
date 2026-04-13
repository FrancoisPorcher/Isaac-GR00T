"""
Gemma 4 Object Detection — Detect objects with bounding boxes in rollout frames.

Crops each camera view (left/right/wrist) from the 768x256 concatenated frame,
prompts Gemma 4 for JSON bounding-box detection, and visualizes results.

Usage:
    uv run python gemma4/detect_objects.py \
        --video_path gemma4/failure_TurnOnMicrowave_*.mp4

    uv run python gemma4/detect_objects.py \
        --video_dir gemma4 --max_videos 5
"""

import argparse
import json
import re
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "google/gemma-4-26B-A4B-it"

VIEW_NAMES = ["left", "right", "wrist"]
VIEW_WIDTH = 256

DETECTION_PROMPT = """\
Detect the {object_name} in this image.
Return JSON: {{"label": "{object_name}", "box_2d": [y_min, x_min, y_max, x_max]}}
box_2d coordinates are integers in [0, 1000] normalized space.
If the object is not visible at all in the image, return {{"label": "{object_name}", "box_2d": null}}
"""


def crop_views(frame: Image.Image) -> dict[str, Image.Image]:
    """Split a 768x256 concatenated frame into three 256x256 camera views."""
    w, h = frame.size
    assert w == 3 * VIEW_WIDTH, f"Expected width {3 * VIEW_WIDTH}, got {w}"
    return {
        name: frame.crop((i * VIEW_WIDTH, 0, (i + 1) * VIEW_WIDTH, h))
        for i, name in enumerate(VIEW_NAMES)
    }


def sample_frames(video_path: str, frame_indices: list[int] | None = None) -> list[Image.Image]:
    """Sample specific frames from a video. Default: last frame only."""
    container = av.open(video_path)
    total = container.streams.video[0].frames or 250

    if frame_indices is None:
        frame_indices = [total - 1]
    resolved = [i % total for i in frame_indices]
    targets = set(resolved)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in targets:
            frames.append(frame.to_image())
        if i > max(targets):
            break
    container.close()
    return frames


def generate(model, processor, messages: list[dict], max_new_tokens: int = 256) -> str:
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def detect(model, processor, image: Image.Image, object_name: str) -> dict:
    """Prompt Gemma 4 to detect an object and return parsed bounding box."""
    prompt = DETECTION_PROMPT.format(object_name=object_name)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    raw = generate(model, processor, messages)
    print(f"    Raw: {raw[:300]}")

    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not json_match:
        return {"label": object_name, "box_2d": None, "raw": raw}

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"label": object_name, "box_2d": None, "raw": raw}

    return result


def box_to_pixels(box_2d: list[int], height: int, width: int) -> list[float]:
    """Convert [y_min, x_min, y_max, x_max] from [0,1000] to pixel coords."""
    y_min, x_min, y_max, x_max = box_2d
    return [
        y_min / 1000 * height,
        x_min / 1000 * width,
        y_max / 1000 * height,
        x_max / 1000 * width,
    ]


def visualize_detections(
    frame: Image.Image,
    views: dict[str, Image.Image],
    view_detections: dict[str, dict],
    output_path: str,
    title: str = "",
):
    """Draw bounding boxes on each cropped view individually + the full frame."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Full concatenated frame
    axes[0].imshow(frame)
    axes[0].set_title("Full frame (768×256)", fontsize=9)
    for i in range(1, 3):
        axes[0].axvline(x=i * VIEW_WIDTH, color="white", linewidth=1, linestyle="--", alpha=0.5)

    colors = {"left": "lime", "right": "cyan", "wrist": "red"}

    for ax_idx, view_name in enumerate(VIEW_NAMES):
        ax = axes[ax_idx + 1]
        view_img = views[view_name]
        ax.imshow(view_img)

        det = view_detections.get(view_name, {})
        box = det.get("box_2d")

        if box is not None:
            h, w = view_img.size[1], view_img.size[0]
            py_min, px_min, py_max, px_max = box_to_pixels(box, h, w)
            rect = patches.Rectangle(
                (px_min, py_min), px_max - px_min, py_max - py_min,
                linewidth=2, edgecolor=colors[view_name], facecolor="none",
            )
            ax.add_patch(rect)
            status = "detected"
        else:
            status = "not found"

        ax.set_title(f"{view_name} — {status}", fontsize=9, color=colors[view_name])
        ax.axis("off")

    axes[0].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def find_videos(video_dir: str, pattern: str = "failure_*.mp4") -> list[Path]:
    return sorted(Path(video_dir).glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Detect objects in rollout frames with Gemma 4")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--object", type=str, default="START ENTER button on the microwave keypad")
    parser.add_argument("--frame_indices", type=int, nargs="+", default=[-1],
                        help="Frame indices to analyze (negative = from end). Default: last frame.")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="gemma4/detections")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    args = parser.parse_args()

    assert args.video_path or args.video_dir, "Provide --video_path or --video_dir"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Model loaded on {model.device}")

    if args.video_path:
        videos = [Path(args.video_path)]
    else:
        videos = find_videos(args.video_dir)[:args.max_videos]
        assert videos, f"No videos found in {args.video_dir}"
    print(f"Processing {len(videos)} video(s), detecting: '{args.object}'")

    all_results = []

    for vi, video_path in enumerate(videos):
        print(f"\n[{vi+1}/{len(videos)}] {video_path.name}")
        frames = sample_frames(str(video_path), frame_indices=args.frame_indices)

        for fi, frame in enumerate(frames):
            frame_idx = args.frame_indices[fi]
            print(f"  Frame {frame_idx} ({frame.size[0]}x{frame.size[1]})")

            views = crop_views(frame)
            view_detections = {}

            for view_name, view_img in views.items():
                print(f"  [{view_name}]")
                det = detect(model, processor, view_img, args.object)
                view_detections[view_name] = det
                box = det.get("box_2d")
                if box:
                    print(f"    ✓ box={box}")
                else:
                    print(f"    ✗ not found")

            out_name = f"{video_path.stem}_frame{frame_idx}.png"
            visualize_detections(
                frame, views, view_detections,
                str(output_dir / out_name),
                title=f"{video_path.name} — frame {frame_idx}",
            )

            all_results.append({
                "video": str(video_path),
                "frame_idx": frame_idx,
                "detections": {
                    k: {kk: vv for kk, vv in v.items() if kk != "raw"}
                    for k, v in view_detections.items()
                },
            })

    results_path = output_dir / "detection_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
