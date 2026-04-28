from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from rlft.offline.pi05_bridge.subtask_annotations import (
    iter_recorded_episode_paths,
    load_subtask_annotation_sidecar,
)
from rlft.offline.pi05_bridge.task_semantics import load_pi05_task_semantics


@dataclass
class Args:
    subtask_annotations_path: str
    recorded_root: str = "recorded_data"
    task_semantics_path: str = "configs/pi05_task_semantics/pick_and_place_tape_into_cup.json"
    subsets: Optional[list[str]] = None
    require_all_recorded: bool = False
    review_queue_path: Optional[str] = None


def main() -> None:
    args = tyro.cli(Args)
    semantics = load_pi05_task_semantics(args.task_semantics_path)
    annotations = load_subtask_annotation_sidecar(args.subtask_annotations_path, semantics)

    subsets = tuple(args.subsets) if args.subsets else None
    recorded_paths = iter_recorded_episode_paths(args.recorded_root, subsets=subsets) if subsets else iter_recorded_episode_paths(args.recorded_root)
    expected_ids = {f"{path.parent.name}/{path.name}" for path in recorded_paths}
    annotated_ids = set(annotations)

    missing_ids = sorted(expected_ids - annotated_ids)
    extra_ids = sorted(annotated_ids - expected_ids)
    needs_review = sorted(
        annotation.episode_id
        for annotation in annotations.values()
        if annotation.review_status == "needs_review" or any(flag.startswith("needs_review") for flag in annotation.flags)
    )

    by_subset: dict[str, dict[str, int]] = {}
    for annotation in annotations.values():
        stats = by_subset.setdefault(annotation.source_subset, {"total": 0, "needs_review": 0})
        stats["total"] += 1
        if annotation.episode_id in needs_review:
            stats["needs_review"] += 1

    if args.review_queue_path:
        path = Path(args.review_queue_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(needs_review) + ("\n" if needs_review else ""))

    ok = not needs_review and (not args.require_all_recorded or not missing_ids)
    payload = {
        "ok": ok,
        "num_annotations": len(annotations),
        "num_recorded_expected": len(expected_ids),
        "num_missing": len(missing_ids),
        "num_extra": len(extra_ids),
        "num_needs_review": len(needs_review),
        "by_subset": by_subset,
        "missing_ids": missing_ids[:50],
        "extra_ids": extra_ids[:50],
        "needs_review": needs_review,
        "review_queue_path": str(Path(args.review_queue_path).expanduser().resolve()) if args.review_queue_path else None,
    }
    print(json.dumps(payload, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
