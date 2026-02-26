#!/usr/bin/env python3
"""
collect_rollouts_iter.py — P1.1 单次迭代 Rollout 收集包装脚本

从 train_vlaw.py Step 1 调用，调用 data_collector.CollectorConfig.run()
依次收集所有任务的 D_real 数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str,
                        default="LiftPegUpright-v1,PickCube-v1,StackCube-v1")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--iter_id", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_base = Path(args.output_dir)

    print(f"[Step1] collect_rollouts_iter iter={args.iter_id} tasks={task_list}")

    if args.dry_run:
        print("[DRY RUN] 跳过实际收集")
        return

    from rlft.vlaw.data import CollectorConfig, VLAWDataCollector

    for task_id in task_list:
        print(f"\n[Step1] 收集 {task_id} × {args.num_episodes} episodes ...")
        cfg = CollectorConfig(
            env_id=task_id,
            num_envs=args.num_envs,
            num_episodes=args.num_episodes,
            checkpoint_path=args.policy_ckpt,
            use_random_policy=(args.policy_ckpt == ""),
            output_dir=str(output_base / task_id),
        )
        collector = VLAWDataCollector(cfg)
        out_path = collector.run()
        print(f"[Step1] ✅ {task_id} → {out_path}")

    print(f"\n[Step1] 全部任务收集完成 → {output_base}")


if __name__ == "__main__":
    main()
