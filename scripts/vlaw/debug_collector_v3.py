"""Debug script: test collector with small num_envs + debug prints."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import numpy as np  
import torch
from rlft.vlaw.data.collector import (
    CollectorConfig, VLAWDataCollector, _np, _get_render_frame,
    extract_raw_frames, extract_agent_state, build_obs_features,
    Trajectory, RandomPolicy, ShortCutFlowPolicy,
)

cfg = CollectorConfig(
    env_id="LiftPegUpright-v1",
    num_envs=4,      # Very small for fast debug
    num_episodes=20,  # 20 episodes = ~5 rounds of 4 envs
    max_episode_steps=200,
    checkpoint_path="checkpoints/il/best_eval_success_once.pt",
    frame_skip=3,
    min_traj_length=10,
    gpu_id=0,         # Already restricted via CUDA_VISIBLE_DEVICES
    output_dir="/tmp/debug_collect",
    source_tag="debug",
    task_instruction="debug",
    verbose=True,
)

collector = VLAWDataCollector(cfg)
env = collector._make_env()
policy, visual_encoder = collector._load_policy(env)
N = cfg.num_envs

print(f"\n[DEBUG] Env created, N={N}, policy loaded")

obs, _ = env.reset(seed=42)
print(f"[DEBUG] Reset done, obs keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")

# Quick test: step 10 times
for step_i in range(10):
    # Extract obs
    rgb_base = _np(obs["rgb"]).astype(np.uint8)
    agent_state = _np(obs["state"]).astype(np.float32)
    
    print(f"[DEBUG] Step {step_i}: rgb_base.shape={rgb_base.shape}, agent_state.shape={agent_state.shape}")
    
    # Step with random actions
    actions = np.random.randn(N, 7).astype(np.float32) * 0.1
    actions_t = torch.from_numpy(actions).to(collector.device)
    
    obs, reward, terminated, truncated, info = env.step(actions_t)
    done = _np(terminated).astype(bool) | _np(truncated).astype(bool)
    
    print(f"[DEBUG] Step {step_i} done: {done}, terminated: {_np(terminated)}, truncated: {_np(truncated)}")
    
    if np.any(done):
        print(f"[DEBUG] Done envs: {np.where(done)[0]}")

print("\n[DEBUG] Basic stepping works. Now testing render...")

# Test render
t0 = time.time()
rgb_base = _np(obs["rgb"]).astype(np.uint8)
rgb_render = _get_render_frame(env, N, cfg.camera_height, cfg.camera_width, rgb_base)
t1 = time.time()
print(f"[DEBUG] Render: rgb_render.shape={rgb_render.shape}, took {t1-t0:.3f}s")
print(f"[DEBUG] rgb_base mean: {rgb_base.mean():.1f}, rgb_render mean: {rgb_render.mean():.1f}")

print("\n[DEBUG] Now testing full collector with 20 episodes...")

# Full collection test
t0 = time.time()
env.close()
trajs = collector.collect_rollouts()
elapsed = time.time() - t0

print(f"\n[DEBUG] === RESULT ===")
print(f"[DEBUG] Collected {len(trajs)} trajectories in {elapsed:.1f}s")
for i, t in enumerate(trajs[:10]):
    T = t["actions"].shape[0]
    suc = t["env_success"][-1]
    print(f"[DEBUG]   traj {i}: T={T}, success_at_end={suc}")

all_T = [t["actions"].shape[0] for t in trajs]
print(f"[DEBUG] T stats: min={min(all_T)}, max={max(all_T)}, mean={np.mean(all_T):.1f}")
print(f"[DEBUG] Success rate: {sum(t['env_success'][-1] for t in trajs)/len(trajs):.1%}")
