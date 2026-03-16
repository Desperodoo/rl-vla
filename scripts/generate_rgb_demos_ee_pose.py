"""Generate RGB observation demos for pd_ee_pose from existing env_states.

Reads trajectory.none.pd_ee_pose.physx_cpu.h5 (actions + env_states only),
replays env_states in an obs_mode='rgb' env to capture RGB observations,
and writes trajectory.rgb.pd_ee_pose.physx_cpu.h5 with full obs structure.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n rlft_ms3 python scripts/generate_rgb_demos_ee_pose.py
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm


def h5_to_state_dict(env_states_group: h5py.Group, step_idx: int) -> dict:
    """Extract a single-step state dict from HDF5 env_states group."""
    state: dict = {}
    for cat in ("actors", "articulations"):
        state[cat] = {}
        for name in env_states_group[cat]:
            arr = env_states_group[cat][name][step_idx]
            state[cat][name] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return state


def capture_obs(env) -> dict:
    """Capture current observations from the environment.

    Returns dict with:
        agent/qpos: (9,) float32
        agent/qvel: (9,) float32
        extra/tcp_pose: (7,) float32
        sensor_data/base_camera/rgb: (128, 128, 3) uint8
    """
    obs = env.unwrapped.get_obs()
    # obs structure: {agent: {qpos, qvel}, extra: {tcp_pose}, sensor_data: {base_camera: {rgb}}}
    result = {}
    result["agent_qpos"] = obs["agent"]["qpos"].view(-1).cpu().numpy().astype(np.float32)
    result["agent_qvel"] = obs["agent"]["qvel"].view(-1).cpu().numpy().astype(np.float32)
    result["extra_tcp_pose"] = obs["extra"]["tcp_pose"].view(-1).cpu().numpy().astype(np.float32)
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    result["base_camera_rgb"] = rgb.squeeze(0).astype(np.uint8)  # (128, 128, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate RGB demos for pd_ee_pose")
    parser.add_argument(
        "--src",
        default=os.path.expanduser(
            "~/.maniskill/demos/LiftPegUpright-v1/rl/"
            "trajectory.none.pd_ee_pose.physx_cpu.h5"
        ),
    )
    parser.add_argument("--env_id", default="LiftPegUpright-v1")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument(
        "--output",
        default=os.path.expanduser(
            "~/.maniskill/demos/LiftPegUpright-v1/rl/"
            "trajectory.rgb.pd_ee_pose.physx_cpu.h5"
        ),
    )
    args = parser.parse_args()

    src_h5 = h5py.File(args.src, "r")
    traj_keys = sorted(
        [k for k in src_h5.keys() if k.startswith("traj_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if args.count is not None:
        traj_keys = traj_keys[: args.count]
    print(f"Source: {args.src}  ({len(traj_keys)} trajectories)")

    env = gym.make(
        args.env_id,
        obs_mode="rgb",
        control_mode="pd_ee_pose",
        sim_backend="cpu",
        num_envs=1,
    )
    env.reset()

    # Remove stale output file if it exists (could be an empty 800B leftover)
    if os.path.exists(args.output):
        os.remove(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_h5 = h5py.File(args.output, "w")

    total_frames = 0
    success_count = 0

    for traj_key in tqdm(traj_keys, desc="Generating RGB demos"):
        traj = src_h5[traj_key]
        if "env_states" not in traj:
            print(f"  SKIP {traj_key}: no env_states")
            continue

        env_states = traj["env_states"]
        actions = traj["actions"][:]  # (T, 7) pd_ee_pose
        # Unwrap euler_rx (dim 3) from [-pi, pi] to [0, 2*pi] to eliminate
        # bimodal discontinuity at ±pi that breaks Flow Matching training.
        # ManiSkill's pd_ee_pose controller handles 2*pi-periodic angles correctly.
        mask = actions[:, 3] < 0
        actions[mask, 3] += 2 * np.pi
        T = actions.shape[0]

        # Collect T+1 observations (state 0..T)
        qpos_list = []
        qvel_list = []
        tcp_pose_list = []
        rgb_list = []

        for t in range(T + 1):
            state_dict = h5_to_state_dict(env_states, t)
            env.unwrapped.set_state_dict(state_dict)
            obs = capture_obs(env)
            qpos_list.append(obs["agent_qpos"])
            qvel_list.append(obs["agent_qvel"])
            tcp_pose_list.append(obs["extra_tcp_pose"])
            rgb_list.append(obs["base_camera_rgb"])

        # Write output
        grp = out_h5.create_group(traj_key)
        grp.create_dataset("actions", data=actions)

        # obs group
        obs_grp = grp.create_group("obs")
        agent_grp = obs_grp.create_group("agent")
        agent_grp.create_dataset("qpos", data=np.stack(qpos_list))
        agent_grp.create_dataset("qvel", data=np.stack(qvel_list))
        extra_grp = obs_grp.create_group("extra")
        extra_grp.create_dataset("tcp_pose", data=np.stack(tcp_pose_list))
        sensor_data_grp = obs_grp.create_group("sensor_data")
        cam_grp = sensor_data_grp.create_group("base_camera")
        cam_grp.create_dataset("rgb", data=np.stack(rgb_list), dtype=np.uint8)

        # Copy env_states
        src_h5.copy(traj["env_states"], grp, "env_states")

        # Copy step-level signals
        for key in ("terminated", "truncated", "success", "rewards"):
            if key in traj:
                grp.create_dataset(key, data=traj[key][:])

        if "success" in traj and np.any(traj["success"][:]):
            success_count += 1
        total_frames += T

    out_h5.close()
    src_h5.close()
    env.close()

    n = len(traj_keys)
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nOutput: {args.output}")
    print(f"Trajectories: {n},  Frames: {total_frames},  Size: {file_size:.1f} MB")
    print(f"Success: {success_count}/{n} ({100*success_count/max(1, n):.1f}%)")


if __name__ == "__main__":
    main()
