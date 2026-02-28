#!/usr/bin/env python3
"""
evaluate_policy.py — P7 策略评估脚本

从 train_vlaw.py Step 8 调用，在 ManiSkill 仿真中评估当前策略。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tasks", type=str,
                        default="LiftPegUpright-v1",
                        help="任务列表（默认 Lift-only；PickCube/StackCube deferred）")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--iter_id", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Step8] evaluate_policy iter={args.iter_id}")
    print(f"  policy_ckpt : {args.policy_ckpt}")
    print(f"  tasks       : {task_list}")
    print(f"  num_episodes: {args.num_episodes}")

    if args.dry_run:
        results = {t: {"success_rate": 0.0, "dry_run": True} for t in task_list}
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print("[DRY RUN] 跳过实际评估")
        return

    from rlft.vlaw.data_collector import CollectorConfig, RolloutCollector

    results: dict = {}

    for task_id in task_list:
        print(f"\n[Step8] 评估 {task_id}...")
        cfg = CollectorConfig(
            env_id=task_id,
            num_envs=args.num_envs,
            num_episodes=args.num_episodes,
            checkpoint_path=args.policy_ckpt,
            use_random_policy=False,
            output_dir="/tmp/eval_temp",  # 临时目录，不保存
        )
        collector = RolloutCollector(cfg)

        # RolloutCollector 的 run() 返回 HDF5 路径，我们从 meta 读取成功率
        import h5py
        out_path = collector.run()
        with h5py.File(out_path) as f:
            meta = f.get("meta", {})
            total = meta.get("total_episodes", args.num_episodes)
            success = meta.get("success_count", 0)
            success_rate = float(success) / max(int(total), 1)

        results[task_id] = {
            "success_rate": success_rate,
            "num_episodes": int(total),
            "num_success": int(success),
        }
        print(f"  {task_id}: success_rate={success_rate:.2%}")

    # 保存结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    overall_sr = sum(r["success_rate"] for r in results.values()) / max(len(results), 1)
    print(f"\n[Step8] ✅ 评估完成 | 平均成功率={overall_sr:.2%} → {output_path}")


if __name__ == "__main__":
    main()
