#!/usr/bin/env python3
"""
run_policy_update.py — P5.2 策略更新包装脚本

从 train_vlaw.py Step 7 调用，使用 PolicyUpdater 在
D_real+ ∪ D_syn+ ∪ D_demo 上更新 ShortCut Flow 策略。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(WORKSPACE))


def collect_hdf5_paths(labeled_dir: Path, task_list: list[str]) -> list[str]:
    """收集标注目录中所有任务的 HDF5 路径列表。"""
    paths = []
    for task_id in task_list:
        task_dir = labeled_dir / task_id
        paths.extend(str(p) for p in sorted(task_dir.glob("*.h5")))
    return paths


def collect_demo_paths(demo_dir: Path, task_list: list[str]) -> list[str]:
    """收集演示数据的 HDF5 路径。"""
    paths = []
    for task_id in task_list:
        task_dir = demo_dir / task_id
        if task_dir.exists():
            paths.extend(str(p) for p in sorted(task_dir.glob("*.h5")))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_real_dir", type=str, required=True,
                        help="VLM 标注 D_real 目录")
    parser.add_argument("--labeled_syn_dir", type=str, required=True,
                        help="VLM 标注 D_syn 目录")
    parser.add_argument("--demo_dir", type=str, required=True,
                        help="演示数据目录")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="起点策略 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 checkpoint 目录")
    parser.add_argument("--tasks", type=str,
                        default="LiftPegUpright-v1",
                        help="任务列表（默认 Lift-only；PickCube/StackCube deferred）")
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--iter_id", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    labeled_real = Path(args.labeled_real_dir)
    labeled_syn = Path(args.labeled_syn_dir)
    demo_dir = Path(args.demo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Step7] run_policy_update iter={args.iter_id}")
    print(f"  checkpoint : {args.checkpoint_path}")
    print(f"  tasks      : {task_list}")
    print(f"  num_steps  : {args.num_steps}")

    # 收集数据路径
    real_paths = collect_hdf5_paths(labeled_real, task_list) if labeled_real.exists() else []
    syn_paths  = collect_hdf5_paths(labeled_syn, task_list) if labeled_syn.exists() else []
    demo_paths = collect_demo_paths(demo_dir, task_list)

    print(f"  D_real HDF5: {len(real_paths)} 文件")
    print(f"  D_syn  HDF5: {len(syn_paths)} 文件")
    print(f"  D_demo HDF5: {len(demo_paths)} 文件")

    if args.dry_run:
        print("[DRY RUN] 跳过实际策略更新")
        return

    if not demo_paths and not syn_paths and not real_paths:
        print("[WARN] 无可用训练数据，跳过策略更新")
        return

    from rlft.vlaw.policy_updater import PolicyUpdaterConfig, PolicyUpdater

    cfg = PolicyUpdaterConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=str(output_dir),
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        iter_id=args.iter_id,
    )

    updater = PolicyUpdater(cfg)
    metrics = updater.update(
        real_hdf5_paths=real_paths,
        syn_hdf5_paths=syn_paths,
        demo_hdf5_paths=demo_paths,
    )

    # 保存训练指标
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Step7] ✅ 策略更新完成 → {output_dir}")
    print(f"  指标: {metrics}")


if __name__ == "__main__":
    main()
