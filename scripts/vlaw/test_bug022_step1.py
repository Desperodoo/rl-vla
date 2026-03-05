#!/usr/bin/env python3
"""BUG-022 Ghost Episode 修复验证 - Step 1: 小规模测试 (num_envs=4, num_episodes=20)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from rlft.vlaw.data.collector import CollectorConfig, VLAWDataCollector
import numpy as np

cfg = CollectorConfig(
    env_id='LiftPegUpright-v1',
    num_envs=4,
    num_episodes=20,
    max_episode_steps=200,
    checkpoint_path='checkpoints/il/best_eval_success_once.pt',
    frame_skip=3,
    min_traj_length=10,
    gpu_id=0,
    output_dir='/tmp/debug_bug022',
    source_tag='debug',
    task_instruction='LiftPegUpright',
    verbose=True,
)
collector = VLAWDataCollector(cfg)
trajs = collector.collect_rollouts()
print()
print('=== BUG-022 验证 (Step 1: 小规模) ===')
all_T = [t['actions'].shape[0] for t in trajs]
print(f'轨迹数: {len(trajs)}')
print(f'T 分布: min={min(all_T)}, max={max(all_T)}, mean={np.mean(all_T):.1f}')
print(f'Success rate (at_end): {sum(bool(t["env_success"][-1]) for t in trajs)/len(trajs):.1%}')

# 验证 rgb_base vs rgb_render diff
diffs = [np.abs(t['rgb_base'].astype(float) - t['rgb_render'].astype(float)).mean() for t in trajs[:5]]
print(f'rgb_base vs rgb_render diff (前5条): {[f"{d:.1f}" for d in diffs]}')

# 检查是否有 T<10 轨迹（应被过滤）
short = [t for t in trajs if t['actions'].shape[0] < 10]
print(f'T<10 轨迹: {len(short)} 条 (应为 0)')

print()
print('=== STEP1_DONE ===')
