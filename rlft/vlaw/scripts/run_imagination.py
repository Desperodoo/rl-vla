#!/usr/bin/env python3
"""
run_imagination.py — P4.3 Imagination 包装脚本

从 train_vlaw.py Step 5 调用，使用 ImaginationEnvEngine 生成 D_syn。
使用 ManiSkill env.step() 做精确状态转移（不依赖 State Predictor MLP）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parents[3].resolve()
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_ckpt", type=str, required=True,
                        help="Ctrl-World checkpoint 路径")
    parser.add_argument("--policy_ckpt", type=str, required=True,
                        help="ShortCut Flow checkpoint 路径")
    parser.add_argument("--real_data_dir", type=str, required=True,
                        help="D_real rollout 目录（提供初始状态）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="D_syn 输出目录")
    parser.add_argument("--tasks", type=str,
                        default="LiftPegUpright-v1",
                        help="任务列表（默认 Lift-only；PickCube/StackCube deferred）")
    parser.add_argument("--iter_id", type=int, default=1)
    parser.add_argument("--num_trajectories", type=int, default=200,
                        help="每任务合成轨迹数")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="并行仿真环境数")
    parser.add_argument("--gpu_ids", type=str, default="4,5,6,7",
                        help="可用 GPU IDs")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_base = Path(args.output_dir)
    real_data_base = Path(args.real_data_dir)

    print(f"[Step5] run_imagination iter={args.iter_id} tasks={task_list}")
    print(f"  wm_ckpt     : {args.wm_ckpt}")
    print(f"  policy_ckpt : {args.policy_ckpt}")
    print(f"  num_trajs   : {args.num_trajectories}")

    if args.dry_run:
        print("[DRY RUN] 跳过实际 Imagination 运行")
        return

    from rlft.vlaw.imagination_env import ImaginationEnvConfig, ImaginationEnvEngine

    for task_id in task_list:
        print(f"\n[Step5] {task_id}: 生成 {args.num_trajectories} 条合成轨迹...")

        cfg = ImaginationEnvConfig(
            task_id=task_id,
            num_envs=args.num_envs,
            wm_checkpoint_path=args.wm_ckpt,
            policy_checkpoint_path=args.policy_ckpt,
        )
        engine = ImaginationEnvEngine(cfg)

        real_task_dir = real_data_base / task_id
        out_task_dir = output_base / task_id

        result = engine.run(
            iter_id=args.iter_id,
            real_data_dir=str(real_task_dir),
            output_dir=str(out_task_dir),
            num_trajectories=args.num_trajectories,
        )
        print(f"[Step5] ✅ {task_id}: {result}")

    print(f"\n[Step5] Imagination 全部完成 → {output_base}")


if __name__ == "__main__":
    main()
