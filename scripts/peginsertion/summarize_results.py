#!/usr/bin/env python3
"""Summarize PegInsertionSide-v1 best-run results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_ALGORITHMS = [
    "diffusion_policy",
    "flow_matching",
    "consistency_flow",
    "shortcut_flow",
    "reflected_flow",
    "cpql",
    "awcp",
    "aw_shortcut_flow",
    "sac",
    "dqc",
    "rlpd_sac",
    "awsc",
    "pld",
    "dsrl",
]


def latest_run(root: Path, exp_name: str, algorithm: str) -> Path | None:
    parent = root / "runs" / exp_name / algorithm
    candidates = sorted(parent.glob("best__*"), reverse=True)
    if candidates:
        return candidates[0]
    fallback = parent / "best"
    return fallback if fallback.exists() else None


def find_checkpoint(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    ckpt_dir = run_dir / "checkpoints"
    preferred = [
        "best_eval_success_once.pt",
        "best_eval_success_at_end.pt",
        "best.pt",
        "best_sae.pt",
        "final.pt",
    ]
    for name in preferred:
        path = ckpt_dir / name
        if path.exists():
            return str(path)
    numeric = sorted(ckpt_dir.glob("[0-9]*.pt"), reverse=True)
    if numeric:
        return str(numeric[0])
    step = sorted(ckpt_dir.glob("step_*.pt"), reverse=True)
    if step:
        return str(step[0])
    return None


def read_log(run_dir: Path | None, root: Path, exp_name: str, algorithm: str) -> str:
    paths = []
    if run_dir is not None:
        paths.append(run_dir / "train.log")
    paths.append(root / "runs" / exp_name / algorithm / "best" / "train.log")
    for path in paths:
        if path.exists():
            return path.read_text(errors="ignore")
    return ""


def last_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            try:
                value = float(matches[-1])
            except ValueError:
                continue
            if value > 1.0 and "%" in pattern:
                value /= 100.0
            return value
    return None


def parse_metrics(text: str) -> dict:
    return {
        "success_once": last_float(
            [
                r"final/best_success_(?:rate|once)\s+([0-9.]+)",
                r"eval/success_once\s+([0-9.]+)",
                r"success_once:\s+([0-9.]+)",
                r"Best SO:\s+([0-9.]+)%",
            ],
            text,
        ),
        "success_at_end": last_float(
            [
                r"final/best_success_at_end\s+([0-9.]+)",
                r"eval/success_at_end\s+([0-9.]+)",
                r"success_at_end:\s+([0-9.]+)",
                r"Best SAE:\s+([0-9.]+)%",
            ],
            text,
        ),
    }


def status_for(run_dir: Path | None, checkpoint: str | None, log_text: str) -> str:
    if checkpoint:
        return "success"
    if run_dir is None and not log_text:
        return "not_started"
    if re.search(r"(error|exception|traceback|out of memory|oom|failed)", log_text, re.I):
        return "failed"
    return "running_or_incomplete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--exp-name", default="peginsertion_best1")
    parser.add_argument("--output", default="results/peginsertion_best1/summary.json")
    parser.add_argument("--algorithms", nargs="*", default=DEFAULT_ALGORITHMS)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary = {
        "exp_name": args.exp_name,
        "task": "PegInsertionSide-v1",
        "runs": {},
    }

    for algo in args.algorithms:
        run_dir = latest_run(root, args.exp_name, algo)
        checkpoint = find_checkpoint(run_dir)
        log_text = read_log(run_dir, root, args.exp_name, algo)
        metrics = parse_metrics(log_text)
        summary["runs"][algo] = {
            "status": status_for(run_dir, checkpoint, log_text),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "checkpoint": checkpoint,
            **metrics,
        }

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
