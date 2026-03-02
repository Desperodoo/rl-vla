#!/usr/bin/env python3
"""B3: Policy training with D_real+ data (env_success filtering)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")

from rlft.vlaw.policy.policy_updater import PolicyUpdaterConfig, VLAWPolicyUpdater

cfg = PolicyUpdaterConfig(
    checkpoint_path="checkpoints/il/best_eval_success_once.pt",
    output_dir="checkpoints/vlaw/policy/iter1_trackb",
    num_steps=2000,
    batch_size=16,
    learning_rate=1e-5,
    warmup_steps=100,
    gpu_id=0,
    use_wandb=False,
    dry_run=False,
    iter_id=1,
    action_horizon=8,
)
updater = VLAWPolicyUpdater(cfg)
metrics = updater.update(
    real_success_dirs=["data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1"],
    syn_success_dirs=[],
)
print(f"\n[B3] DONE: {metrics}")
