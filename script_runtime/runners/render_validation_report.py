"""Render visual report for existing validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from script_runtime.validation.report import render_validation_report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render charts and markdown report from validation summary.")
    parser.add_argument(
        "--summary",
        default="script_runtime/artifacts/maniskill_validation_summary.json",
        help="Path to validation summary JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="script_runtime/artifacts/maniskill_report",
        help="Directory for generated charts and markdown report.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode timeline to render.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    outputs = render_validation_report(
        summary_path=Path(args.summary),
        output_dir=Path(args.out_dir),
        timeline_episode_index=args.episode_index,
    )
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
