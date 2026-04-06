#!/usr/bin/env python3
"""Run a baseline timeline analysis for a CARM inference run.

This is a thin wrapper around analyze_timeline.py that:
1. picks a timeline JSONL
2. optionally reads the paired run_info JSON
3. writes a small baseline summary JSON for iterative comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _find_latest(pattern: str) -> str:
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return str(matches[-1])


def _load_json(path: str) -> dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARM inference timeline baseline analysis")
    parser.add_argument('--timeline', type=str, default='', help='Path to timeline JSONL (default: latest in inference_logs)')
    parser.add_argument('--run_info', type=str, default='', help='Path to paired run_info JSON (optional)')
    parser.add_argument('--out', type=str, default='', help='Output summary JSON path')
    args = parser.parse_args()

    timeline_path = args.timeline or _find_latest('inference_logs/timeline_*.jsonl')
    run_info_path = args.run_info or _find_latest('inference_logs/run_info_*.json')
    out_path = args.out or str(Path(timeline_path).with_name(Path(timeline_path).stem + '_baseline.json'))

    analyzer = Path('carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py')
    subprocess.run([
        'python',
        str(analyzer),
        '--logs',
        timeline_path,
        '--out',
        out_path,
    ], check=True)

    summary = _load_json(out_path)
    run_info = _load_json(run_info_path)
    baseline = {
        'timeline': os.path.basename(timeline_path),
        'run_info': os.path.basename(run_info_path),
        'execution': run_info.get('execution', {}),
        'control': run_info.get('control', {}),
        'summary': run_info.get('summary', {}),
        'timeline_stats': summary,
        'event_counts': summary.get('basic', {}).get('event_counts', {}),
    }

    with open(out_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f'Wrote baseline summary to {out_path}')


if __name__ == '__main__':
    main()
