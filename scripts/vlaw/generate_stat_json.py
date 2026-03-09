"""Generate stat.json (EE pose normalization) from v3 rollout data.

Computes state_01, state_99 from **absolute EE (TCP) pose** extracted from
the ``state`` field in HDF5 rollouts.  The 7-D "action conditioning" vector
fed to Ctrl-World is ``[tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz,
gripper_normalized]``, matching the DROID convention where the WM is
conditioned on the per-frame end-effector state rather than action deltas.

ManiSkill HDF5 state layout (25-D):
    [0:9]   = qpos   (7 arm joints + 2 gripper fingers)
    [9:18]  = qvel   (velocities)
    [18:25] = tcp_pose (x, y, z, qw, qx, qy, qz)

Gripper: qpos[7] ∈ [0, 0.04] → normalized to [0, 1] by dividing by 0.04.

Usage:
    python scripts/vlaw/generate_stat_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Panda gripper max finger opening (one finger, in metres).
PANDA_FINGER_MAX = 0.04


def _state_to_ee_pose_7d(state: np.ndarray) -> np.ndarray:
    """Convert 25-D ManiSkill state → 7-D EE conditioning vector.

    Args:
        state: (N, 25) — raw state from HDF5.

    Returns:
        (N, 7) float32 — [tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz,
                           gripper_norm].
    """
    tcp_pos = state[:, 18:21].astype(np.float64)      # (N, 3) xyz
    tcp_quat_wxyz = state[:, 21:25].astype(np.float64) # (N, 4) qw,qx,qy,qz
    # scipy expects xyzw ordering
    tcp_quat_xyzw = tcp_quat_wxyz[:, [1, 2, 3, 0]]
    euler = Rot.from_quat(tcp_quat_xyzw).as_euler("xyz")  # (N, 3)
    gripper_norm = (state[:, 7] / PANDA_FINGER_MAX).clip(0.0, 1.0)  # (N,)
    return np.column_stack([tcp_pos, euler, gripper_norm[:, None]]).astype(np.float32)


def collect_data_from_h5(
    h5_path: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Extract actions, states, and EE poses from all trajectories."""
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_ee_poses: list[np.ndarray] = []
    with h5py.File(str(h5_path), "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("traj_"):
                continue
            grp = f[key]
            if "actions" in grp:
                all_actions.append(grp["actions"][:])
            if "state" in grp:
                st = grp["state"][:].astype(np.float32)
                all_states.append(st)
                all_ee_poses.append(_state_to_ee_pose_7d(st))
    return all_actions, all_states, all_ee_poses


def main() -> None:
    # Input directories: mixed + high_suc (training data only, not eval)
    dirs = [
        PROJECT_ROOT / "data/vlaw/rollouts/mixed/LiftPegUpright-v1",
        PROJECT_ROOT / "data/vlaw/rollouts/high_suc/LiftPegUpright-v1",
    ]
    output_path = PROJECT_ROOT / "data/vlaw/meta_info/maniskill/stat.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_ee_poses: list[np.ndarray] = []

    for d in dirs:
        h5_files = sorted(d.glob("*.h5"))
        for h5_path in h5_files:
            print(f"Reading {h5_path.name}...")
            actions, states, ee_poses = collect_data_from_h5(h5_path)
            all_actions.extend(actions)
            all_states.extend(states)
            all_ee_poses.extend(ee_poses)

    if not all_actions:
        print("ERROR: No action data found!")
        sys.exit(1)

    actions = np.concatenate(all_actions, axis=0)  # (N, 7) delta pose
    ee_poses = np.concatenate(all_ee_poses, axis=0)  # (N, 7) EE pose
    print(f"Total frames: {ee_poses.shape[0]}, EE pose dim={ee_poses.shape[1]}")

    action_mean = actions.mean(axis=0).tolist()
    action_std = actions.std(axis=0).tolist()

    # state_01 / state_99 are now EE-pose percentiles (matches DROID semantics)
    p01 = np.percentile(ee_poses, 1, axis=0).tolist()
    p99 = np.percentile(ee_poses, 99, axis=0).tolist()

    stat: dict = {
        "action_mean": action_mean,
        "action_std": action_std,
        "num_samples": int(ee_poses.shape[0]),
        "action_dim": 7,
        "state_01": p01,
        "state_99": p99,
        "ee_pose_labels": [
            "tcp_x", "tcp_y", "tcp_z",
            "euler_rx", "euler_ry", "euler_rz",
            "gripper_norm",
        ],
        "note": (
            "state_01/state_99 are p1/p99 of absolute EE pose "
            "[tcp_xyz + euler_xyz + gripper_norm], matching DROID's "
            "cartesian_position + gripper_position convention."
        ),
    }

    with open(str(output_path), "w") as f:
        json.dump(stat, f, indent=2)

    labels = stat["ee_pose_labels"]
    print(f"\nstat.json saved to: {output_path}")
    print(f"  EE pose percentiles (state_01 / state_99):")
    for i, lbl in enumerate(labels):
        print(f"    {lbl:12s}: p01={p01[i]:+.6f}  p99={p99[i]:+.6f}  range={p99[i]-p01[i]:.6f}")
    print(f"  action_mean (delta, info only): {[f'{x:.4f}' for x in action_mean]}")
    print(f"  action_std  (delta, info only): {[f'{x:.4f}' for x in action_std]}")
    print(f"  num_samples: {stat['num_samples']}")


if __name__ == "__main__":
    main()
