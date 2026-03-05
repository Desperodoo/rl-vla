"""Diagnostic: Test partial reset and elapsed_steps behavior in ManiSkill3 GPU vec env.

Key questions:
1. Does env.reset(options={"env_idx": [i]}) work for partial reset?
2. After partial reset, do truncated/terminated/elapsed_steps reset properly?
3. Does the env also reset elapsed_steps when using ManiSkillVectorEnv?
4. Can we use env.base_env._elapsed_steps to track and manually manage?
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
import numpy as np

print("=" * 70)
print("TEST A: Partial reset via env.reset(options={'env_idx': ...})")
print("=" * 70)

env = gym.make(
    "LiftPegUpright-v1",
    num_envs=4,
    sim_backend="physx_cuda",
    obs_mode="rgbd",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=5,
)
obs, _ = env.reset(seed=42)

# Step to truncation
for step in range(5):
    action = torch.randn(4, 7, device="cuda") * 0.01
    obs, reward, terminated, truncated, info = env.step(action)

t = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
print(f"After 5 steps: truncated={t}")

# Try partial reset of envs 0 and 2
print("\nAttempting partial reset: env_idx=[0, 2]")
try:
    obs_reset, info_reset = env.reset(options={"env_idx": torch.tensor([0, 2], device="cuda")})
    print(f"  Partial reset succeeded! obs type: {type(obs_reset)}")
    if isinstance(obs_reset, dict):
        for k, v in obs_reset.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={v.shape}")
            elif isinstance(v, dict):
                print(f"    {k}: dict with {list(v.keys())}")
except Exception as e:
    print(f"  Partial reset FAILED: {e}")
    
    # Try alternative: env.reset(seed=None, options={"env_idx": ...})
    print("\nTrying alternative partial reset format...")
    try:
        obs_reset, info_reset = env.reset(seed=None, options={"env_idx": [0, 2]})
        print(f"  Alternative succeeded!")
    except Exception as e2:
        print(f"  Alternative also FAILED: {e2}")

# Now step once more and check flags
print("\nStepping once after partial reset:")
action = torch.randn(4, 7, device="cuda") * 0.01
obs, reward, terminated, truncated, info = env.step(action)
t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
es = info.get("elapsed_steps", None)
if es is not None and isinstance(es, torch.Tensor):
    es = es.cpu().numpy()
print(f"  terminated={t}")
print(f"  truncated={tr}")
print(f"  elapsed_steps={es}")

# Step a few more to see if reset envs stay clean
for s in range(3):
    action = torch.randn(4, 7, device="cuda") * 0.01
    obs, reward, terminated, truncated, info = env.step(action)
    t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
    tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
    es = info.get("elapsed_steps", None)
    if es is not None and isinstance(es, torch.Tensor):
        es = es.cpu().numpy()
    print(f"  Step +{s+2}: term={t} trunc={tr} elapsed={es}")

env.close()


print("\n" + "=" * 70)
print("TEST B: Check _elapsed_steps attribute")
print("=" * 70)

env2 = gym.make(
    "LiftPegUpright-v1",
    num_envs=4,
    sim_backend="physx_cuda",
    obs_mode="rgbd",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=5,
)
obs, _ = env2.reset(seed=42)

# Check internal attributes
print("Checking internal attributes...")
base = env2
attrs_to_check = ["_elapsed_steps", "elapsed_steps", "base_env", "_env"]
for attr in attrs_to_check:
    if hasattr(base, attr):
        val = getattr(base, attr)
        if isinstance(val, (int, float, np.ndarray, torch.Tensor)):
            print(f"  env.{attr} = {val}")
        else:
            print(f"  env.{attr} = {type(val)}")

# Walk up wrapper chain
print("\nWrapper chain:")
e = env2
depth = 0
while e is not None:
    print(f"  {'  ' * depth}{type(e).__name__}")
    if hasattr(e, '_elapsed_steps'):
        es = getattr(e, '_elapsed_steps')
        if isinstance(es, torch.Tensor):
            es = es.cpu().numpy()
        print(f"  {'  ' * depth}  _elapsed_steps = {es}")
    if hasattr(e, 'env'):
        e = e.env
    elif hasattr(e, '_env'):
        e = e._env
    elif hasattr(e, 'base_env'):
        e = e.base_env
    else:
        break
    depth += 1
    if depth > 10:
        break

# Step to truncation and check internal state
for step in range(5):
    action = torch.randn(4, 7, device="cuda") * 0.01
    obs, reward, terminated, truncated, info = env2.step(action)

print(f"\nAfter 5 steps (at truncation):")
e = env2
while e is not None:
    if hasattr(e, '_elapsed_steps'):
        es = getattr(e, '_elapsed_steps')
        if isinstance(es, torch.Tensor):
            es = es.cpu().numpy()
        print(f"  {type(e).__name__}._elapsed_steps = {es}")
    if hasattr(e, 'env'):
        e = e.env
    elif hasattr(e, '_env'):
        e = e._env
    elif hasattr(e, 'base_env'):
        e = e.base_env
    else:
        break

env2.close()


print("\n" + "=" * 70)
print("TEST C: Full reset clears flags?")
print("=" * 70)

env3 = gym.make(
    "LiftPegUpright-v1",
    num_envs=4,
    sim_backend="physx_cuda",
    obs_mode="rgbd",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=5,
)
obs, _ = env3.reset(seed=42)

# Step to truncation
for step in range(5):
    action = torch.randn(4, 7, device="cuda") * 0.01
    obs, reward, terminated, truncated, info = env3.step(action)

t = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
print(f"Before full reset: truncated={t}")

# Full reset
obs, _ = env3.reset()
print("Full reset done.")

# Step and check
for step in range(7):
    action = torch.randn(4, 7, device="cuda") * 0.01
    obs, reward, terminated, truncated, info = env3.step(action)
    t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
    tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
    es = info.get("elapsed_steps", None)
    if es is not None and isinstance(es, torch.Tensor):
        es = es.cpu().numpy()
    print(f"  Step {step}: term={t} trunc={tr} elapsed={es}")

env3.close()


print("\n" + "=" * 70)
print("TEST D: Test early termination (success) behavior")
print("=" * 70)

# Use PickCube as it's easier to trigger success
env4 = gym.make(
    "LiftPegUpright-v1",
    num_envs=4,
    sim_backend="physx_cuda",
    obs_mode="rgbd",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=200,
)
obs, _ = env4.reset(seed=42)

# Check: does ManiSkill set terminated on success? Or only truncated at max_steps?
# Step a bunch and print any terminated=True events
print("Stepping with random actions, watching for terminated=True...")
for step in range(200):
    action = torch.randn(4, 7, device="cuda") * 0.3
    obs, reward, terminated, truncated, info = env4.step(action)
    t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
    tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
    if t.any() or tr.any():
        es = info.get("elapsed_steps", None)
        if es is not None and isinstance(es, torch.Tensor):
            es = es.cpu().numpy()
        s = info.get("success", None)
        if s is not None and isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
        print(f"  Step {step}: term={t} trunc={tr} success={s} elapsed={es}")
        if step > 210:  # Don't print forever
            break

env4.close()

print("\n" + "=" * 70)
print("ALL DIAGNOSTICS COMPLETE")
print("=" * 70)
