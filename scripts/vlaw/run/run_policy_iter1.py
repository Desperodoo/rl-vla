#!/usr/bin/env python3
"""VLAW P5.1 — Iter-1 策略更新 (Filtered BC on D_real+ ∪ D_demo).

使用 env_success 过滤的成功轨迹进行策略微调:
  - D_real: 3 个 real rollout 目录中的成功轨迹 (45+87 = 132 samples)
  - D_demo: 25 条演示轨迹 (43 samples, 100% success)
  - 总计: ~175 success samples
  - 无合成数据 (iter-1 暂无 Ctrl-World imagination)

GPU: 8 (CUDA_VISIBLE_DEVICES=8, gpu_id=0)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from rlft.vlaw.policy.policy_updater import PolicyUpdaterConfig, VLAWPolicyUpdater

# ---------------------------------------------------------------------------
# 数据目录
# ---------------------------------------------------------------------------
REAL_SUCCESS_DIRS = [
    "data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1",
    "data/vlaw/rollouts/iter1_lift_inc20/LiftPegUpright-v1",
    "data/vlaw/rollouts/iter1/LiftPegUpright-v1",
]

DEMO_DIRS = [
    "data/vlaw/demos/LiftPegUpright-v1",
]

# iter-1 暂无合成数据
SYN_DIRS: list[str] = []

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
cfg = PolicyUpdaterConfig(
    checkpoint_path="checkpoints/il/best_eval_success_once.pt",
    output_dir="checkpoints/vlaw/policy/iter1",
    num_steps=2000,
    batch_size=16,
    learning_rate=1e-5,
    warmup_steps=100,
    gpu_id=0,  # CUDA_VISIBLE_DEVICES=8 → 逻辑 GPU 0
    use_wandb=True,
    wandb_run_name="vlaw_policy_iter1",
    use_visual_obs=True,
    state_dim=25,
    visual_feature_dim=256,
    obs_horizon=2,
    action_horizon=8,
    unet_down_dims=(64, 128, 256),
    unet_step_embed_dim=64,
    unet_n_groups=8,
    dry_run=False,
    iter_id=1,
)

# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[run_policy_iter1] 配置: num_steps={cfg.num_steps}, bs={cfg.batch_size}, "
          f"lr={cfg.learning_rate}, warmup={cfg.warmup_steps}")
    print(f"[run_policy_iter1] Real dirs: {REAL_SUCCESS_DIRS}")
    print(f"[run_policy_iter1] Demo dirs: {DEMO_DIRS}")
    print(f"[run_policy_iter1] Syn dirs: {SYN_DIRS}")

    updater = VLAWPolicyUpdater(cfg)
    metrics = updater.update(
        real_success_dirs=REAL_SUCCESS_DIRS,
        syn_success_dirs=SYN_DIRS,
        demo_dirs=DEMO_DIRS,
    )
    print(f"\n[run_policy_iter1] DONE: {metrics}")
