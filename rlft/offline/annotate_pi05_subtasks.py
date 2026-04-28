from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tyro

from rlft.offline.pi05_bridge.subtask_annotations import (
    DEFAULT_SUBSETS,
    QwenVideoAnnotator,
    build_episode_annotation,
    detect_rule_based_subtask_boundary,
    export_review_video,
    parse_vlm_boundary_frame,
    read_episode_info,
    validate_annotation_record,
    write_annotation_sidecar,
    write_boundary_contact_sheet,
    write_review_html,
)
from rlft.offline.pi05_bridge.task_semantics import load_pi05_task_semantics


@dataclass
class Args:
    recorded_root: str = "recorded_data"
    output_dir: str = "runs/pi05_subtask_annotations/pick_and_place_tape_into_cup"
    task_semantics_path: str = "configs/pi05_task_semantics/pick_and_place_tape_into_cup.json"
    subsets: list[str] = field(default_factory=lambda: list(DEFAULT_SUBSETS))
    pilot_episodes_per_subset: Optional[int] = None
    run_vlm: bool = False
    boundary_source: str = "vlm"
    skip_existing: bool = True
    review_video_fps: float = 2.0
    qwen_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    allow_stub_annotations: bool = False
    stub_boundary_fraction: float = 0.5
    hf_cache_root: Optional[str] = "/mnt/disk_2/wjz/.cache/huggingface"
    disable_hf_xet: bool = True
    tmp_dir: Optional[str] = "/mnt/disk_2/wjz/tmp"
    local_files_only: bool = False
    rule_min_blue_threshold: float = 0.012
    rule_stable_seconds: float = 0.12
    rule_lower_bound_margin_seconds: float = 0.20
    write_contact_sheets: bool = True


def _select_paths(args: Args) -> list[Path]:
    recorded_root = Path(args.recorded_root).expanduser().resolve()
    selected: list[Path] = []
    for subset in args.subsets:
        paths = sorted((recorded_root / subset).glob("episode_*.hdf5"))
        if args.pilot_episodes_per_subset is not None:
            paths = paths[: args.pilot_episodes_per_subset]
        selected.extend(paths)
    return selected


def _existing_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as handle:
        data = json.load(handle)
    return {record["episode_id"]: record for record in data.get("episodes", [])}


def _configure_hf_cache(args: Args) -> None:
    if args.hf_cache_root:
        cache_root = Path(args.hf_cache_root).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "hub").mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    if args.disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir).expanduser().resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)
        os.environ["TMP"] = str(tmp_dir)
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")


def main() -> None:
    args = tyro.cli(Args)
    _configure_hf_cache(args)
    semantics = load_pi05_task_semantics(args.task_semantics_path)
    if len(semantics.subtasks) != 2:
        raise ValueError("This annotator expects exactly two subtasks")
    if args.boundary_source not in {"vlm", "rule_detector", "stub"}:
        raise ValueError("--boundary-source must be one of: vlm, rule_detector, stub")

    recorded_root = Path(args.recorded_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    videos_dir = output_dir / "videos"
    contact_sheets_dir = output_dir / "contact_sheets"
    raw_vlm_dir = output_dir / "raw_vlm"
    annotations_path = output_dir / "annotations.json"
    review_path = output_dir / "review.html"

    episode_paths = _select_paths(args)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {recorded_root}")

    existing = _existing_records(annotations_path) if args.skip_existing else {}
    records: list[dict] = list(existing.values())
    selected_ids = {
        f"{path.parent.name}/{path.name}"
        for path in episode_paths
    }
    records = [record for record in records if record.get("episode_id") in selected_ids]
    existing = {record["episode_id"]: record for record in records}

    manifest_records: list[dict] = []
    annotator: QwenVideoAnnotator | None = None
    for index, episode_path in enumerate(episode_paths, start=1):
        episode_id = f"{episode_path.parent.name}/{episode_path.name}"
        if episode_id in existing:
            print(f"[SKIP] {episode_id} already annotated")
            continue

        info = read_episode_info(episode_path)
        review_video = videos_dir / info["source_subset"] / f"{episode_path.stem}_dual_view_2fps.mp4"
        export_review_video(episode_path, review_video, target_fps=args.review_video_fps)
        rel_video = review_video.relative_to(output_dir).as_posix()
        manifest_record = {
            **info,
            "review_video": rel_video,
        }
        manifest_records.append(manifest_record)
        print(f"[VIDEO] {index}/{len(episode_paths)} {episode_id} -> {review_video}")

        if args.boundary_source == "vlm" and not args.run_vlm and not args.allow_stub_annotations:
            continue

        detector_output = None
        refine = True
        if args.boundary_source == "rule_detector":
            detector_output = detect_rule_based_subtask_boundary(
                episode_path,
                min_blue_threshold=args.rule_min_blue_threshold,
                stable_seconds=args.rule_stable_seconds,
                lower_bound_margin_seconds=args.rule_lower_bound_margin_seconds,
            )
            vlm_boundary_frame = int(detector_output["boundary_frame"])
            vlm_raw = {
                "boundary_source": "rule_detector",
                "rule_detector": detector_output,
            }
            parse_error = None
            refine = False
        elif args.run_vlm:
            if annotator is None:
                annotator = QwenVideoAnnotator(
                    semantics=semantics,
                    model_name=args.qwen_model,
                    device=args.device,
                    dtype=args.dtype,
                    local_files_only=args.local_files_only,
                )
            raw_text = annotator.annotate(review_video)
            raw_vlm_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_vlm_dir / info["source_subset"] / f"{episode_path.stem}.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_text)
            try:
                vlm_boundary_frame, vlm_raw = parse_vlm_boundary_frame(
                    raw_text,
                    fps=float(info["fps"]),
                    expected_subtasks=semantics.subtask_names,
                )
                parse_error = None
            except Exception as exc:
                vlm_boundary_frame = int(round(float(info["num_frames"]) * args.stub_boundary_fraction))
                vlm_raw = {
                    "parse_error": type(exc).__name__,
                    "parse_error_message": str(exc),
                    "raw_text": raw_text,
                    "fallback_boundary_frame": vlm_boundary_frame,
                }
                parse_error = exc
        else:
            vlm_boundary_frame = int(round(float(info["num_frames"]) * args.stub_boundary_fraction))
            vlm_raw = {
                "stub": True,
                "reason": "allow_stub_annotations was enabled for smoke testing",
                "boundary_frame": vlm_boundary_frame,
            }
            parse_error = None

        record = build_episode_annotation(
            episode_path,
            semantics,
            vlm_boundary_frame=vlm_boundary_frame,
            vlm_raw_output=vlm_raw,
            recorded_root=recorded_root,
            refine=refine,
            boundary_source=args.boundary_source,
            detector_output=detector_output,
        )
        record["review_video"] = rel_video
        if args.write_contact_sheets:
            markers = {"boundary": int(record["boundary_frame"])}
            if detector_output is not None:
                markers = {
                    "rule": int(detector_output["boundary_frame"]),
                    "grasp_lift": int(detector_output["grasp_lift_frame"]),
                    "lower_bound": int(detector_output["grasp_lift_lower_bound_frame"]),
                }
            elif record.get("refinement"):
                markers["robot"] = int(record["refinement"]["frame"])
            sheet_path = contact_sheets_dir / info["source_subset"] / f"{episode_path.stem}_boundary_sheet.jpg"
            write_boundary_contact_sheet(episode_path, sheet_path, markers=markers)
            record["contact_sheet"] = sheet_path.relative_to(output_dir).as_posix()
        if parse_error is not None:
            record["flags"] = sorted(set(record.get("flags", []) + ["needs_review_vlm_parse_error"]))
            record["confidence"] = 0.1
            record["review_status"] = "needs_review"
        if args.allow_stub_annotations and not args.run_vlm:
            record["flags"] = sorted(set(record.get("flags", []) + ["needs_review_stub_annotation"]))
            record["confidence"] = 0.1
            record["review_status"] = "needs_review"

        errors = validate_annotation_record(record, semantics)
        if errors:
            raise ValueError(f"Invalid annotation for {episode_id}: {errors}")
        records.append(record)
        write_annotation_sidecar(
            annotations_path,
            semantics=semantics,
            records=records,
            recorded_root=recorded_root,
        )
        write_review_html(review_path, records)
        print(f"[ANNOTATED] {episode_id} boundary={record['boundary_frame']} flags={record['flags']}")

    manifest_path = output_dir / "episode_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_records, indent=2) + "\n")

    if records:
        write_annotation_sidecar(
            annotations_path,
            semantics=semantics,
            records=records,
            recorded_root=recorded_root,
        )
        write_review_html(review_path, records)
        num_review = sum(1 for record in records if record.get("review_status") == "needs_review")
        print(f"[DONE] wrote {len(records)} annotations to {annotations_path} ({num_review} need review)")
    else:
        print(f"[DONE] wrote review videos only. Re-run with --run-vlm to create {annotations_path}")


if __name__ == "__main__":
    main()
