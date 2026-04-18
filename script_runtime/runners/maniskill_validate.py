"""Batch validation runner for script_runtime in ManiSkill."""

from __future__ import annotations

import argparse
from pathlib import Path

from script_runtime.session import load_runtime_config
from script_runtime.validation.maniskill_rollout import run_maniskill_validation, save_validation_summary
from script_runtime.validation.report import render_validation_report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-episode script_runtime validation in ManiSkill.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks" / "pick_cube_maniskill.yaml"),
        help="Path to YAML/JSON validation config.",
    )
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional override for validation summary JSON path.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    config = load_runtime_config(args.config)
    try:
        summary = run_maniskill_validation(config)
    except ModuleNotFoundError as exc:
        print("ManiSkill validation dependencies are not installed in the current environment.")
        print("Install it first, for example:")
        print("  conda activate maniskill")
        print("  pip install gymnasium")
        print("  pip install mani-skill")
        print(f"details={exc}")
        return 2

    output_path = args.summary_out or config.get("validation", {}).get(
        "summary_path",
        "script_runtime/artifacts/maniskill_validation_summary.json",
    )
    saved = save_validation_summary(summary, output_path)
    report_dir = Path(config.get("validation", {}).get("report_dir", "script_runtime/artifacts/maniskill_report"))
    report_outputs = render_validation_report(saved, report_dir, timeline_episode_index=0)

    print(f"episodes={summary.num_episodes}")
    print(f"runtime_successes={summary.runtime_successes}")
    print(f"sim_successes={summary.sim_successes}")
    print(f"failure_code_counts={summary.failure_code_counts}")
    print(f"failed_skill_counts={summary.failed_skill_counts}")
    print(f"summary_path={saved}")
    for key, value in report_outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
