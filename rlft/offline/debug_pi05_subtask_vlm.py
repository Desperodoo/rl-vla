from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import tyro
from PIL import Image, ImageDraw

from rlft.offline.pi05_bridge.subtask_annotations import (
    export_review_video,
    parse_vlm_boundary_frame,
    read_episode_info,
    refine_boundary_from_robot_signals,
)


DEFAULT_EPISODE = "recorded_data/fixed_dual_light/episode_0005_20260319_235708.hdf5"
DEFAULT_ANNOTATIONS = (
    "/mnt/disk_2/wjz/runs/pi05_subtask_annotations/"
    "pick_and_place_tape_into_cup_full_qwen25vl7b_local_cup_visible_boundary/annotations.json"
)
DEFAULT_OUTPUT = (
    "/mnt/disk_2/wjz/runs/pi05_subtask_vlm_debug/"
    "fixed_dual_light_episode_0005_20260319_235708"
)


@dataclass
class Args:
    episode_path: str = DEFAULT_EPISODE
    output_dir: str = DEFAULT_OUTPUT
    annotations_path: Optional[str] = DEFAULT_ANNOTATIONS
    sample_fps_values: list[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])
    current_vlm_boundary_frame: Optional[int] = None
    manual_gold_boundary_frame: Optional[int] = 420
    max_tiles_per_sheet: int = 30
    sheet_columns: int = 3
    tile_height: int = 180
    export_videos: bool = True
    write_prompts_only: bool = True
    ablation_prompt_names: Optional[list[str]] = None
    run_local_qwen: bool = False
    qwen_model: str = "/mnt/disk_2/wjz/models_local/Qwen2.5-VL-7B-Instruct-local"
    device: str = "cuda"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    run_openai: bool = False
    openai_model: str = "gpt-4.1"


def _configure_large_cache() -> None:
    cache_root = Path("/mnt/disk_2/wjz/.cache/huggingface")
    tmp_dir = Path("/mnt/disk_2/wjz/tmp")
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "hub").mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", "/mnt/disk_2/wjz/.cache")
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")


def _load_current_vlm_boundary(annotations_path: str | None, episode_id: str) -> int | None:
    if not annotations_path:
        return None
    path = Path(annotations_path).expanduser()
    if not path.exists():
        return None
    with open(path) as handle:
        payload = json.load(handle)
    for record in payload.get("episodes", []):
        if record.get("episode_id") == episode_id:
            boundary = record.get("vlm", {}).get("boundary_frame", record.get("boundary_frame"))
            return int(boundary)
    return None


def _dual_view_tile(
    third: np.ndarray,
    wrist: np.ndarray,
    *,
    frame_index: int,
    fps: float,
    tile_height: int,
    markers: dict[str, int],
) -> Image.Image:
    left = Image.fromarray(third)
    right = Image.fromarray(wrist)
    if left.height != tile_height:
        width = int(round(left.width * tile_height / left.height))
        left = left.resize((width, tile_height), Image.Resampling.BILINEAR)
    if right.height != tile_height:
        width = int(round(right.width * tile_height / right.height))
        right = right.resize((width, tile_height), Image.Resampling.BILINEAR)

    header = 30
    width = left.width + right.width
    tile = Image.new("RGB", (width, tile_height + header + 6), (245, 245, 245))
    tile.paste(left, (0, header))
    tile.paste(right, (left.width, header))
    draw = ImageDraw.Draw(tile)
    tags = [name for name, marker_frame in markers.items() if marker_frame == frame_index]
    tag_text = f"  [{' | '.join(tags)}]" if tags else ""
    draw.rectangle([0, 0, width - 1, header - 1], fill=(18, 18, 18))
    draw.text(
        (8, 8),
        f"frame {frame_index}  t={frame_index / fps:.2f}s  third_person | wrist{tag_text}",
        fill=(255, 255, 255),
    )

    border = (170, 170, 170)
    if "manual_gold" in tags:
        border = (20, 160, 80)
    elif "current_vlm" in tags:
        border = (220, 70, 40)
    elif "robot_signal" in tags:
        border = (50, 110, 220)
    draw.rectangle([0, 0, width - 1, tile.height - 1], outline=border, width=4)
    return tile


def _sheet_frame_indices(
    num_frames: int,
    fps: float,
    sample_fps: float,
    markers: dict[str, int],
) -> list[int]:
    stride = max(1, int(round(fps / sample_fps)))
    indices = set(range(0, num_frames, stride))
    indices.add(num_frames - 1)
    for frame in markers.values():
        if 0 <= frame < num_frames:
            indices.add(int(frame))
    return sorted(indices)


def _write_contact_sheets(
    episode_path: Path,
    output_dir: Path,
    *,
    sample_fps: float,
    markers: dict[str, int],
    max_tiles_per_sheet: int,
    sheet_columns: int,
    tile_height: int,
) -> list[Path]:
    with h5py.File(episode_path, "r") as handle:
        fps = float(handle.attrs.get("record_freq", 30.0))
        num_frames = int(handle.attrs.get("num_steps", handle["action"].shape[0]))
        by_camera = handle.get("observations/images_by_camera")
        if by_camera is not None and "third_person" in by_camera:
            third = by_camera["third_person"]
        elif "observations/images" in handle:
            third = handle["observations/images"]
        else:
            raise KeyError(f"No third-person images found in {episode_path}")
        if by_camera is not None and "wrist" in by_camera:
            wrist = by_camera["wrist"]
        elif "observations/images" in handle:
            wrist = handle["observations/images"]
        else:
            wrist = third

        indices = _sheet_frame_indices(num_frames, fps, sample_fps, markers)
        sheet_paths: list[Path] = []
        sheet_dir = output_dir / "contact_sheets" / f"{sample_fps:g}fps"
        sheet_dir.mkdir(parents=True, exist_ok=True)

        for page, start in enumerate(range(0, len(indices), max_tiles_per_sheet), start=1):
            page_indices = indices[start : start + max_tiles_per_sheet]
            tiles = [
                _dual_view_tile(
                    np.asarray(third[idx]),
                    np.asarray(wrist[idx]),
                    frame_index=idx,
                    fps=fps,
                    tile_height=tile_height,
                    markers=markers,
                )
                for idx in page_indices
            ]
            tile_w = max(tile.width for tile in tiles)
            tile_h = max(tile.height for tile in tiles)
            rows = int(np.ceil(len(tiles) / sheet_columns))
            sheet = Image.new("RGB", (sheet_columns * tile_w, rows * tile_h), (230, 230, 230))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i % sheet_columns) * tile_w, (i // sheet_columns) * tile_h))
            path = sheet_dir / f"sheet_{page:02d}.jpg"
            sheet.save(path, quality=92)
            sheet_paths.append(path)
    return sheet_paths


def _write_prompt_ablations(output_dir: Path, *, fps: float, num_frames: int) -> dict[str, str]:
    prompts = {
        "baseline_timestamp": (
            "Segment this robot manipulation demonstration into pick_tape and place_tape_in_cup. "
            "Return JSON timestamps for the boundary."
        ),
        "layout_event_frame_index": (
            "The image/video shows two synchronized camera views concatenated horizontally: "
            "LEFT is third_person, RIGHT is wrist/first-person. Find the first frame where all conditions "
            "are true: (1) the gripper is already holding the black tape roll, (2) the tape roll has been "
            "lifted off the table, and (3) after that lift, the blue cup first appears in the RIGHT wrist view. "
            "Ignore any blue cup visibility before the tape is grasped and lifted. Return only JSON with "
            "{\"boundary_frame\": <original_frame_index>, \"events\": {\"stable_grasp_frame\": <int>, "
            "\"lifted_frame\": <int>, \"cup_visible_after_lift_frame\": <int>}, \"reason\": <short string>}."
        ),
        "numbered_contact_sheet": (
            "Use the visible frame labels printed at the top of each tile. The boundary is not first contact "
            "or first stable grasp. The boundary is the first labeled frame, after grasp and lift, where the "
            "blue cup is visible in the wrist/first-person view on the RIGHT. If the exact frame lies between "
            "two tiles, choose the first later labeled frame. Return only JSON with boundary_frame and reason."
        ),
    }
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name, prompt in prompts.items():
        (prompt_dir / f"{name}.txt").write_text(
            prompt + f"\n\nOriginal FPS: {fps:.3f}; original num_frames: {num_frames}.\n"
        )
    return prompts


def _run_local_qwen_on_images(
    image_paths: list[Path],
    prompt: str,
    *,
    model_name: str,
    device: str,
    dtype: str,
    local_files_only: bool,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype_map[dtype],
        device_map="auto" if device.startswith("cuda") else device,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a precise robot video annotator."}]},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": str(image_path)} for image_path in image_paths],
                {
                    "type": "text",
                    "text": (
                        "The attached images are consecutive numbered contact-sheet pages from the same episode. "
                        "Read the frame labels printed in each tile.\n\n"
                        f"{prompt}"
                    ),
                },
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return processor.batch_decode(
        [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=True)],
        skip_special_tokens=True,
    )[0].strip()


def _run_openai_on_images(image_paths: list[Path], prompt: str, *, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The attached images are consecutive numbered contact-sheet pages from the same episode. "
                "Read the frame labels printed in each tile.\n\n"
                f"{prompt}"
            ),
        }
    ]
    for image_path in image_paths:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed: HTTP {exc.code}: {body}") from exc
    return data["choices"][0]["message"]["content"]


def _write_report(
    output_dir: Path,
    *,
    episode_id: str,
    info: dict[str, Any],
    markers: dict[str, int],
    refinement: dict[str, Any],
    contact_sheets: dict[str, list[str]],
    model_results: dict[str, Any],
) -> None:
    current = markers.get("current_vlm")
    gold = markers.get("manual_gold")
    robot = markers.get("robot_signal")
    lines = [
        f"# PI05 VLM Subtask Debug: {episode_id}",
        "",
        "## References",
        "",
        f"- FPS: {info['fps']}",
        f"- Frames: {info['num_frames']}",
        f"- Current VLM boundary: frame {current} ({current / info['fps']:.2f}s)" if current is not None else "- Current VLM boundary: unknown",
        f"- Robot signal peak near current VLM: frame {robot} ({robot / info['fps']:.2f}s)" if robot is not None else "- Robot signal peak: unknown",
        f"- Codex/manual gold estimate: frame {gold} ({gold / info['fps']:.2f}s)" if gold is not None else "- Codex/manual gold estimate: unset",
        "",
        "## Initial Diagnosis",
        "",
        "- The current VLM output is an early cut for this debug sample: at frame 188 the gripper has not completed the intended pick phase.",
        "- The desired boundary is visual and phase-based, so robot gripper/speed peaks should validate grasp/lift context but should not replace the visual cup-appearance boundary.",
        "- If local VLM keeps collapsing to 2-5 second timestamps on the numbered contact sheet, treat that as a base-model/input-format failure rather than a simple snapping bug.",
        "",
        "## Contact Sheets",
        "",
    ]
    for fps, paths in contact_sheets.items():
        lines.append(f"- {fps}:")
        lines.extend(f"  - `{path}`" for path in paths)
    lines.extend(
        [
            "",
            "## Robot Signal Validation",
            "",
            "```json",
            json.dumps(refinement, indent=2),
            "```",
            "",
            "## Model Results",
            "",
            "```json",
            json.dumps(model_results, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = tyro.cli(Args)
    _configure_large_cache()

    episode_path = Path(args.episode_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = read_episode_info(episode_path)
    episode_id = f"{info['source_subset']}/{info['episode_filename']}"

    current_boundary = args.current_vlm_boundary_frame
    if current_boundary is None:
        current_boundary = _load_current_vlm_boundary(args.annotations_path, episode_id)
    if current_boundary is None:
        current_boundary = int(round(info["num_frames"] * 0.5))

    refinement = refine_boundary_from_robot_signals(episode_path, current_boundary)
    robot_frame = int(refinement["frame"])
    markers = {"current_vlm": int(current_boundary), "robot_signal": robot_frame}
    if args.manual_gold_boundary_frame is not None:
        markers["manual_gold"] = int(args.manual_gold_boundary_frame)

    contact_sheets: dict[str, list[str]] = {}
    for sample_fps in args.sample_fps_values:
        paths = _write_contact_sheets(
            episode_path,
            output_dir,
            sample_fps=sample_fps,
            markers=markers,
            max_tiles_per_sheet=args.max_tiles_per_sheet,
            sheet_columns=args.sheet_columns,
            tile_height=args.tile_height,
        )
        contact_sheets[f"{sample_fps:g}fps"] = [str(path) for path in paths]

    if args.export_videos:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        for sample_fps in args.sample_fps_values:
            export_review_video(
                episode_path,
                video_dir / f"{episode_path.stem}_dual_view_{sample_fps:g}fps.mp4",
                target_fps=sample_fps,
            )

    prompts = _write_prompt_ablations(output_dir, fps=float(info["fps"]), num_frames=int(info["num_frames"]))
    if args.ablation_prompt_names is not None:
        unknown = sorted(set(args.ablation_prompt_names) - set(prompts))
        if unknown:
            raise ValueError(f"Unknown prompt names: {unknown}; choose from {sorted(prompts)}")
        prompts = {name: prompts[name] for name in args.ablation_prompt_names}
    model_results: dict[str, Any] = {
        "local_qwen": "skipped",
        "openai": "skipped",
        "parse_checks": {},
    }

    sheets_for_models = [
        Path(path)
        for path in (contact_sheets["4fps"] if "4fps" in contact_sheets else next(iter(contact_sheets.values())))
    ]
    if args.run_local_qwen:
        model_results["local_qwen"] = {}
        for name, prompt in prompts.items():
            raw = _run_local_qwen_on_images(
                sheets_for_models,
                prompt,
                model_name=args.qwen_model,
                device=args.device,
                dtype=args.dtype,
                local_files_only=args.local_files_only,
            )
            model_results["local_qwen"][name] = raw
            try:
                boundary, parsed = parse_vlm_boundary_frame(
                    raw,
                    fps=float(info["fps"]),
                    expected_subtasks=("pick_tape", "place_tape_in_cup"),
                )
                model_results["parse_checks"][f"local_qwen/{name}"] = {
                    "boundary_frame": boundary,
                    "parsed": parsed,
                }
            except Exception as exc:
                model_results["parse_checks"][f"local_qwen/{name}"] = {
                    "error": type(exc).__name__,
                    "message": str(exc),
                }

    if args.run_openai:
        model_results["openai"] = {}
        for name in [prompt_name for prompt_name in ["layout_event_frame_index", "numbered_contact_sheet"] if prompt_name in prompts]:
            prompt = prompts[name]
            raw = _run_openai_on_images(sheets_for_models, prompt, model=args.openai_model)
            model_results["openai"][name] = raw
            try:
                boundary, parsed = parse_vlm_boundary_frame(
                    raw,
                    fps=float(info["fps"]),
                    expected_subtasks=("pick_tape", "place_tape_in_cup"),
                )
                model_results["parse_checks"][f"openai/{name}"] = {
                    "boundary_frame": boundary,
                    "parsed": parsed,
                }
            except Exception as exc:
                model_results["parse_checks"][f"openai/{name}"] = {
                    "error": type(exc).__name__,
                    "message": str(exc),
                }

    summary = {
        "episode_id": episode_id,
        "episode_path": str(episode_path),
        "info": info,
        "markers": markers,
        "refinement": refinement,
        "contact_sheets": contact_sheets,
        "prompts": {name: str(output_dir / "prompts" / f"{name}.txt") for name in prompts},
        "model_results": model_results,
    }
    (output_dir / "debug_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_report(
        output_dir,
        episode_id=episode_id,
        info=info,
        markers=markers,
        refinement=refinement,
        contact_sheets=contact_sheets,
        model_results=model_results,
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
