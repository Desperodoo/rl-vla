#!/usr/bin/env python3
"""合并多 GPU 并行生成的合成轨迹 HDF5 文件."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


def merge_h5_files(input_dirs: list[str], output_path: str) -> dict:
    """合并多个 GPU 输出目录中的 final h5 文件."""
    all_h5: list[str] = []
    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            print(f"⚠️ 跳过不存在的目录: {d}")
            continue
        # 找 final h5
        finals = sorted(p.glob("synthetic_final_*.h5"))
        batches = sorted(p.glob("synthetic_batch*.h5"))
        if finals:
            all_h5.append(str(finals[-1]))  # 取最新的 final
        elif batches:
            all_h5.append(str(batches[-1]))  # fallback: 最后一个 batch
        else:
            print(f"⚠️ {d} 中无 h5 文件")

    if not all_h5:
        raise FileNotFoundError("没有找到任何 h5 文件可合并")

    print(f"合并 {len(all_h5)} 个文件:")
    for f in all_h5:
        print(f"  - {f}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_trajs = 0
    with h5py.File(str(out_path), "w") as fout:
        meta = fout.create_group("meta")
        meta.attrs["source"] = "merged_parallel_imagination"
        meta.attrs["merge_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta.attrs["source_files"] = json.dumps(all_h5)

        for h5_path in all_h5:
            with h5py.File(h5_path, "r") as fin:
                for key in sorted(k for k in fin.keys() if k.startswith("traj_")):
                    new_key = f"traj_{total_trajs:04d}"
                    fin.copy(key, fout, name=new_key)
                    total_trajs += 1

        meta.attrs["num_trajectories"] = total_trajs

    print(f"✅ 合并完成: {total_trajs} 条轨迹 → {out_path}")
    print(f"   文件大小: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 合并 summary
    summaries = []
    for d in input_dirs:
        sj = Path(d) / "generation_summary.json"
        if sj.exists():
            with open(sj) as f:
                summaries.append(json.load(f))

    merged_summary = {
        "task_id": summaries[0]["task_id"] if summaries else "unknown",
        "num_target": sum(s.get("num_target", 0) for s in summaries),
        "num_generated": total_trajs,
        "total_time_min": max((s.get("total_time_min", 0) for s in summaries), default=0),
        "source_files": all_h5,
        "merged_file": str(out_path),
    }

    summary_path = out_path.parent / "generation_summary_merged.json"
    with open(summary_path, "w") as f:
        json.dump(merged_summary, f, indent=2, ensure_ascii=False)
    print(f"   Summary: {summary_path}")

    return merged_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    merge_h5_files(args.input_dirs, args.output)


if __name__ == "__main__":
    main()
