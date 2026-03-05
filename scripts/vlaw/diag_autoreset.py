"""Minimal diagnostic: ManiSkill3 GPU vec env auto-reset + done/truncated behavior.

Goal: Determine exactly what happens to terminated/truncated flags after auto-reset.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch
import numpy as np

print("=" * 70)
print("STEP 1: Raw ManiSkill env (no wrapper)")
print("=" * 70)

env = gym.make(
    "LiftPegUpright-v1",
    num_envs=4,
    sim_backend="physx_cuda",
    obs_mode="rgbd",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=200,
)
obs, _ = env.reset(seed=42)
print(f"Reset done. obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

first_done_step = None
for step in range(250):
    action = torch.randn(4, 7, device="cuda") * 0.3
    obs, reward, terminated, truncated, info = env.step(action)
    t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
    tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
    done = t | tr

    if done.any():
        done_envs = np.where(done)[0]
        print(f"\nStep {step}: FIRST DONE EVENT")
        print(f"  done_envs={done_envs}")
        print(f"  terminated={t}")
        print(f"  truncated={tr}")
        if "success" in info:
            s = info["success"]
            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
            print(f"  info['success']={s}")
        if "elapsed_steps" in info:
            es = info["elapsed_steps"]
            if isinstance(es, torch.Tensor):
                es = es.cpu().numpy()
            print(f"  info['elapsed_steps']={es}")
        first_done_step = step

        # Now step 10 more times and watch flags closely
        print(f"\n--- Observing next 10 steps after done ---")
        for extra in range(10):
            action2 = torch.randn(4, 7, device="cuda") * 0.3
            obs2, rew2, term2, trunc2, info2 = env.step(action2)
            t2 = term2.cpu().numpy() if isinstance(term2, torch.Tensor) else np.asarray(term2)
            tr2 = trunc2.cpu().numpy() if isinstance(trunc2, torch.Tensor) else np.asarray(trunc2)
            d2 = t2 | tr2
            line = f"  Step {step+1+extra}: term={t2} trunc={tr2} done={d2}"
            if "success" in info2:
                s2 = info2["success"]
                if isinstance(s2, torch.Tensor):
                    s2 = s2.cpu().numpy()
                line += f" success={s2}"
            if "elapsed_steps" in info2:
                es2 = info2["elapsed_steps"]
                if isinstance(es2, torch.Tensor):
                    es2 = es2.cpu().numpy()
                line += f" elapsed={es2}"
            print(line)
            obs = obs2
        break

env.close()


print("\n" + "=" * 70)
print("STEP 2: With FlattenRGBDObservationWrapper")
print("=" * 70)

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

env2 = gym.make(
    "LiftPegUpright-v1",
    num_envs=4,
    sim_backend="physx_cuda",
    obs_mode="rgbd",
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    max_episode_steps=200,
)
env2 = FlattenRGBDObservationWrapper(env2, rgb=True, depth=False, state=True)
obs, _ = env2.reset(seed=42)
print(f"Reset done. obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

for step in range(250):
    action = torch.randn(4, 7, device="cuda") * 0.3
    obs, reward, terminated, truncated, info = env2.step(action)
    t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
    tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
    done = t | tr

    if done.any():
        done_envs = np.where(done)[0]
        print(f"\nStep {step}: FIRST DONE EVENT (wrapped)")
        print(f"  done_envs={done_envs}")
        print(f"  terminated={t}")
        print(f"  truncated={tr}")

        print(f"\n--- Observing next 10 steps after done (wrapped) ---")
        for extra in range(10):
            action2 = torch.randn(4, 7, device="cuda") * 0.3
            obs2, rew2, term2, trunc2, info2 = env2.step(action2)
            t2 = term2.cpu().numpy() if isinstance(term2, torch.Tensor) else np.asarray(term2)
            tr2 = trunc2.cpu().numpy() if isinstance(trunc2, torch.Tensor) else np.asarray(trunc2)
            d2 = t2 | tr2
            line = f"  Step {step+1+extra}: term={t2} trunc={tr2} done={d2}"
            if "success" in info2:
                s2 = info2["success"]
                if isinstance(s2, torch.Tensor):
                    s2 = s2.cpu().numpy()
                line += f" success={s2}"
            if "elapsed_steps" in info2:
                es2 = info2["elapsed_steps"]
                if isinstance(es2, torch.Tensor):
                    es2 = es2.cpu().numpy()
                line += f" elapsed={es2}"
            print(line)
            obs = obs2
        break

env2.close()


print("\n" + "=" * 70)
print("STEP 3: Force early truncation to test auto-reset precisely")
print("=" * 70)

# Use max_episode_steps=5 to trigger truncation quickly
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
print(f"max_episode_steps=5, observing all steps:")

for step in range(20):
    action = torch.randn(4, 7, device="cuda") * 0.01  # small actions
    obs, reward, terminated, truncated, info = env3.step(action)
    t = terminated.cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
    tr = truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
    done = t | tr
    line = f"  Step {step:3d}: term={t} trunc={tr} done={done}"
    if "elapsed_steps" in info:
        es = info["elapsed_steps"]
        if isinstance(es, torch.Tensor):
            es = es.cpu().numpy()
        line += f" elapsed={es}"
    print(line)

env3.close()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
