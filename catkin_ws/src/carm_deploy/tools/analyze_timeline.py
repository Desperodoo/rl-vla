#!/usr/bin/env python3
"""
时间线分析脚本

解析 timeline JSONL 日志，输出关键时间线统计，用于对比采集与推理部署。
"""

import argparse
import json
import os
from collections import defaultdict
import numpy as np


def load_events(paths):
    events = []
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def stats(arr):
    if len(arr) == 0:
        return None
    arr = np.array(arr, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def analyze(events):
    result = {}
    groups = defaultdict(list)
    for e in events:
        groups[e.get('event', 'unknown')].append(e)

    # obs / record_step
    delta_obs = []
    for e in groups.get('obs', []) + groups.get('record_step', []):
        d = e.get('delta_obs')
        if d is not None:
            delta_obs.append(d)
    result['delta_obs'] = stats(delta_obs)

    # inference
    inference_time = []
    for e in groups.get('inference', []):
        t = e.get('inference_time')
        if t is not None:
            inference_time.append(t)
    result['inference_time'] = stats(inference_time)

    # chunk
    delta_chunk_obs = []
    chunk_len = []
    chunk_lead = []  # chunk_targets[0] - chunk_base_time (should be 0)
    chunk_target_offsets = []  # chunk_targets - chunk_base_time
    for e in groups.get('chunk', []):
        d = e.get('delta_chunk_obs')
        if d is not None:
            delta_chunk_obs.append(d)
        targets = e.get('chunk_targets') or []
        base = e.get('chunk_base_time')
        if base is not None and targets:
            chunk_len.append(len(targets))
            chunk_lead.append(targets[0] - base)
            chunk_target_offsets.extend([t - base for t in targets])
    result['delta_chunk_obs'] = stats(delta_chunk_obs)
    result['chunk_len'] = stats(chunk_len)
    result['chunk_lead'] = stats(chunk_lead)
    result['chunk_target_offsets'] = stats(chunk_target_offsets)

    # control
    control_lag = []  # t_send_sys - query_time
    candidate_age = []  # query_time - latest candidate timestamp
    candidate_span = []  # max - min candidate timestamp
    for e in groups.get('control', []):
        t_send = e.get('t_send_sys')
        t_query = e.get('query_time')
        if t_send is not None and t_query is not None:
            control_lag.append(t_send - t_query)
        cands = e.get('candidate_timestamps') or []
        if t_query is not None and cands:
            candidate_age.append(t_query - max(cands))
            candidate_span.append(max(cands) - min(cands))
    result['control_lag'] = stats(control_lag)
    result['candidate_age'] = stats(candidate_age)
    result['candidate_span'] = stats(candidate_span)

    # record action present rate
    if groups.get('record_step'):
        present = [1 if e.get('action_present') else 0 for e in groups['record_step']]
        result['record_action_present_rate'] = float(np.mean(present))

    result['event_counts'] = {k: len(v) for k, v in groups.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze timeline JSONL logs')
    parser.add_argument('--logs', type=str, nargs='+', required=True, help='Timeline JSONL log paths')
    parser.add_argument('--out', type=str, default='', help='Output JSON path (optional)')
    args = parser.parse_args()

    events = load_events(args.logs)
    summary = analyze(events)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
