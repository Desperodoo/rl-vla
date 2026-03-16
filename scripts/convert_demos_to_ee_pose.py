"""Convert ManiSkill demos to pd_ee_pose format using env_states.

For each demo trajectory, loads env_states at each timestep, sets the env state,
and reads the base-frame EE pose.  Action for step t = EE pose at state t+1
(the target the arm reaches toward during that step).

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/convert_demos_to_ee_pose.py --count 669
"""
from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
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


def ee_pose_from_env(env) -> np.ndarray:
    """Read current EE pose in base frame: [x, y, z, euler_rx, ry, rz, gripper_action]."""
    ctrl = env.unwrapped.agent.controller.controllers["arm"]
    p = ctrl.ee_pose_at_base.p.view(-1).cpu().numpy()
    q_wxyz = ctrl.ee_pose_at_base.q.view(-1).cpu().numpy()
    euler = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_euler("xyz")

    qpos = env.unwrapped.agent.robot.get_qpos().view(-1).cpu().numpy()
    gripper_raw = qpos[7]  # panda_finger_joint1 ∈ [0, 0.04]
    gripper_action = (gripper_raw / 0.04) * 2.0 - 1.0  # → [-1, 1]

    return np.concatenate([p, euler, [gripper_action]]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert demos to pd_ee_pose")
    parser.add_argument(
        "--src",
        default=os.path.expanduser(
            "~/.maniskill/demos/LiftPegUpright-v1/rl/"
            "trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
        ),
    )
    parser.add_argument("--env_id", default="LiftPegUpright-v1")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument(
        "--output",
        default=os.path.expanduser(
            "~/.maniskill/demos/LiftPegUpright-v1/rl/"
            "trajectory.none.pd_ee_pose.physx_cpu.h5"
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
        obs_mode="state",
        control_mode="pd_ee_pose",
        sim_backend="cpu",
        num_envs=1,
    )
    env.reset()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_h5 = h5py.File(args.output, "w")

    success_count = 0
    total_frames = 0
    all_actions = []

    for traj_key in tqdm(traj_keys, desc="Converting"):
        traj = src_h5[traj_key]
        if "env_states" not in traj:
            print(f"  SKIP {traj_key}: no env_states")
            continue

        env_states = traj["env_states"]
        # T = number of actions = number data points in actors minus 1
        sample_actor = list(env_states["actors"].keys())[0]
        T_plus_1 = env_states["actors"][sample_actor].shape[0]
        T = T_plus_1 - 1

        # For each step t, action = EE pose at state t+1
        actions = np.zeros((T, 7), dtype=np.float32)
        for t in range(T):
            state_dict = h5_to_state_dict(env_states, t + 1)
            env.unwrapped.set_state_dict(state_dict)
            actions[t] = ee_pose_from_env(env)

        all_actions.append(actions)

        grp = out_h5.create_group(traj_key)
        grp.create_dataset("actions", data=actions)
        # Copy env_states as-is (preserving the group structure)
        src_h5.copy(traj["env_states"], grp, "env_states")
        for key in ("terminated", "truncated", "success", "rewards"):
            if key in traj:
                grp.create_dataset(key, data=traj[key][:])

        if "success" in traj and np.any(traj["success"][:]):
            success_count += 1
        total_frames += T

    out_h5.close()
    src_h5.close()
    env.close()

    # Stats
    all_act = np.concatenate(all_actions, axis=0)
    act_min = all_act.min(axis=0)
    act_max = all_act.max(axis=0)
    act_mean = all_act.mean(axis=0)
    act_std = all_act.std(axis=0)

    n = len(traj_keys)
    print(f"\nOutput: {args.output}")
    print(f"Trajectories: {n},  Frames: {total_frames}")
    print(f"Success: {success_count}/{n} ({100*success_count/max(1,n):.1f}%)")
    print(f"\nAction statistics (pd_ee_pose, base frame):")
    labels = ["tcp_x", "tcp_y", "tcp_z", "euler_rx", "euler_ry", "euler_rz", "grip"]
    for i, lbl in enumerate(labels):
        print(f"  {lbl:10s}: [{act_min[i]:+8.4f}, {act_max[i]:+8.4f}]  "
              f"mean={act_mean[i]:+8.4f}  std={act_std[i]:.4f}")

    # JSON metadata
    json_path = args.output.replace(".h5", ".json")
    meta = {
        "env_id": args.env_id,
        "env_kwargs": {"obs_mode": "none", "control_mode": "pd_ee_pose"},
        "total_trajectories": n,
        "total_frames": total_frames,
        "success_count": success_count,
        "action_min": act_min.tolist(),
        "action_max": act_max.tolist(),
        "action_mean": act_mean.tolist(),
        "action_std": act_std.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {json_path}")


if __name__ == "__main__":
    main()
