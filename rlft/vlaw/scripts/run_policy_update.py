#!/usr/bin/env python3
"""Policy 更新 — 稳定入口脚本 (Weighted Filtered BC).

合并自: run_policy_iter1.py, run_b3_policy_train.py
已包含:
  - BUG-018: EMA checkpoint 格式 (已在 PolicyUpdater._save_checkpoint 中修复)
  - ADR-012: 反灾难性遗忘默认配置 (demo_replay, 低 LR, 减少 steps)

用法:
    # Iter-1 (仅 D_real + D_demo, 无合成数据)
    CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 python \\
        rlft/vlaw/scripts/run_policy_update.py \\
        --iter_id 1 \\
        --real_dirs data/vlaw/rollouts/iter1/LiftPegUpright-v1 \\
        --demo_dirs data/vlaw/demos/LiftPegUpright-v1

    # Iter-2 (D_real + D_syn + D_demo)
    CUDA_VISIBLE_DEVICES=8 conda run -n rlft_ms3 python \\
        rlft/vlaw/scripts/run_policy_update.py \\
        --iter_id 2 \\
        --real_dirs data/vlaw/rollouts/iter2/LiftPegUpright-v1 \\
        --syn_dirs data/vlaw/synthetic/iter2/LiftPegUpright-v1 \\
        --demo_dirs data/vlaw/demos/LiftPegUpright-v1 \\
        --checkpoint_path checkpoints/vlaw/policy/iter1/policy_iter1.pt

    # Dry-run
    conda run -n rlft_ms3 python rlft/vlaw/scripts/run_policy_update.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser(description="VLAW Policy 更新 (稳定版)")

    # 数据目录 (新 API: 直接传 dir 列表)
    parser.add_argument("--real_dirs", type=str, nargs="*", default=[],
                        help="D_real+ HDF5 目录列表 (env_success / vlm_reward 过滤)")
    parser.add_argument("--syn_dirs", type=str, nargs="*", default=[],
                        help="D_syn+ HDF5 目录列表 (Imagination 合成数据)")
    parser.add_argument("--demo_dirs", type=str, nargs="*", default=[],
                        help="D_demo HDF5 目录列表 (高质量演示, weight=1.0)")

    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str,
                        default=str(WORKSPACE / "checkpoints/il/best_eval_success_once.pt"),
                        help="起点策略 checkpoint")
    parser.add_argument("--output_dir", type=str, default="",
                        help="输出目录 (默认: checkpoints/vlaw/policy/iter{iter_id})")

    # 训练超参 — ADR-012 防灾难性遗忘默认值
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="训练步数 (ADR-012: iter-1 用 2000, 后续可减至 1500)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="LR (ADR-012: 1e-5 已较保守, 不建议更高)")
    parser.add_argument("--warmup_steps", type=int, default=100)

    # 模型架构 (与 checkpoint 一致)
    parser.add_argument("--state_dim", type=int, default=25)
    parser.add_argument("--visual_feature_dim", type=int, default=256)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--use_visual_obs", action="store_true", default=True)

    # 运行控制
    parser.add_argument("--iter_id", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="逻辑 GPU ID (用 CUDA_VISIBLE_DEVICES 映射)")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="生成 loss 曲线 PNG")
    args = parser.parse_args()

    # 默认输出目录
    if not args.output_dir:
        args.output_dir = str(WORKSPACE / f"checkpoints/vlaw/policy/iter{args.iter_id}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ADR-012 警告: 无 demo_dirs 时提示灾难性遗忘风险
    if not args.demo_dirs:
        print("[WARN] ADR-012: 未提供 --demo_dirs, 策略可能遗忘预训练能力！")
        print("       建议: 添加 --demo_dirs data/vlaw/demos/LiftPegUpright-v1")

    print(f"[POLICY-UPDATE] iter={args.iter_id}")
    print(f"  checkpoint : {args.checkpoint_path}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  real_dirs  : {args.real_dirs}")
    print(f"  syn_dirs   : {args.syn_dirs}")
    print(f"  demo_dirs  : {args.demo_dirs}")
    print(f"  num_steps={args.num_steps}, bs={args.batch_size}, lr={args.learning_rate}")

    if args.dry_run:
        print("[DRY RUN] ✅ 配置验证通过, 跳过实际训练")
        return

    from rlft.vlaw.policy.policy_updater import PolicyUpdaterConfig, VLAWPolicyUpdater

    cfg = PolicyUpdaterConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gpu_id=args.gpu_id,
        use_wandb=args.use_wandb,
        wandb_run_name=args.wandb_run_name or f"vlaw_policy_iter{args.iter_id}",
        use_visual_obs=args.use_visual_obs,
        state_dim=args.state_dim,
        visual_feature_dim=args.visual_feature_dim,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        dry_run=False,
        iter_id=args.iter_id,
    )

    updater = VLAWPolicyUpdater(cfg)
    metrics = updater.update(
        real_success_dirs=args.real_dirs,
        syn_success_dirs=args.syn_dirs,
        demo_dirs=args.demo_dirs if args.demo_dirs else None,
    )

    # 保存训练指标
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.visualize:
        _visualize_loss(metrics, args.output_dir)

    print(f"\n[POLICY-UPDATE] ✅ 完成 → {args.output_dir}")
    print(f"  指标: {metrics}")


# ── 可视化 ──────────────────────────────────────────────────────────────────

def _visualize_loss(metrics: dict, output_dir: str) -> None:
    """生成 loss 曲线 PNG (x=step, y=loss).

    保存到 {output_dir}/viz/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 尝试从 metrics 中提取 loss 历史
    loss_history = metrics.get("loss_history", [])
    if not loss_history:
        # 尝试 step_losses 或其他格式
        loss_history = metrics.get("step_losses", [])
    if not loss_history:
        # 如果只有 final_loss, 画单点
        final = metrics.get("final_loss", metrics.get("loss", None))
        if final is not None:
            loss_history = [float(final)]

    if not loss_history:
        print("[POLICY-UPDATE] ⚠️ 无 loss 数据可视化")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = list(range(1, len(loss_history) + 1))
    ax.plot(steps, loss_history, "b-", alpha=0.7, linewidth=0.8)
    # 平滑曲线 (窗口=min(50, len//5))
    if len(loss_history) > 10:
        import numpy as np
        window = min(50, max(5, len(loss_history) // 10))
        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        s_steps = list(range(window, window + len(smoothed)))
        ax.plot(s_steps, smoothed, "r-", linewidth=1.5, label=f"Smoothed (w={window})")
        ax.legend(fontsize=8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Update Loss Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = viz_dir / "loss_curve.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[POLICY-UPDATE] 📊 {fig_path.name}")


if __name__ == "__main__":
    main()
