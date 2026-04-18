"""Standalone SDK-first pick-place runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from script_runtime.session import build_pick_place_session, load_runtime_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run script_runtime pick-place directly on arm_control_sdk.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks" / "pick_place_sdk.yaml"),
        help="Path to YAML/JSON runtime config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use MockSDKBridge instead of arm_control_sdk. Useful before the pybind SDK is built.",
    )
    parser.add_argument(
        "--trace-out",
        default="",
        help="Optional trace output override. Writes JSONL trace after execution.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    config = load_runtime_config(args.config)
    if args.trace_out:
        config.setdefault("runtime", {})["trace_path"] = args.trace_out

    session = build_pick_place_session(config, use_mock=args.dry_run)
    try:
        result = session.run()
    finally:
        shutdown_result = session.shutdown()

    print(f"task={session.task.name}")
    print(f"status={result.status.value}")
    print(f"failure_code={result.failure_code.value}")
    print(f"message={result.message}")
    print(f"trace_events={len(session.trace_recorder.events)}")
    if session.trace_path is not None:
        print(f"trace_path={session.trace_path}")
    print(f"shutdown_ok={shutdown_result.get('ok', False)}")
    return 0 if result.status.value == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
