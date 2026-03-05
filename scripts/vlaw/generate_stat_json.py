"""Generate stat.json (action normalization) from v3 rollout data.

Computes action_mean, action_std, state_01, state_99 from mixed + high_suc data.
Output format matches Ctrl-World Dataset_ManiSkill expectations.

Usage:
    python scripts/vlaw/generate_stat_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def collect_data_from_h5(h5_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract actions and states from all trajectories in an HDF5 file."""
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    with h5py.File(str(h5_path), "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("traj_"):
                continue
            grp = f[key]
            if "actions" in grp:
                all_actions.append(grp["actions"][:])
            if "state" in grp:
                all_states.append(grp["state"][:])
    return all_actions, all_states


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

    for d in dirs:
        h5_files = sorted(d.glob("*.h5"))
        for h5_path in h5_files:
            print(f"Reading {h5_path.name}...")
            actions, states = collect_data_from_h5(h5_path)
            all_actions.extend(actions)
            all_states.extend(states)

    if not all_actions:
        print("ERROR: No action data found!")
        sys.exit(1)

    actions = np.concatenate(all_actions, axis=0)  # (N, action_dim)
    print(f"Total action samples: {actions.shape[0]}, dim={actions.shape[1]}")

    action_mean = actions.mean(axis=0).tolist()
    action_std = actions.std(axis=0).tolist()

    stat: dict = {
        "action_mean": action_mean,
        "action_std": action_std,
        "num_samples": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
    }

    # Add state percentiles if available
    if all_states:
        # state_01 and state_99 use only first action_dim columns of state
        # (matching the old stat.json format — state is typically [qpos, qvel])
        states = np.concatenate(all_states, axis=0)  # (N, state_dim)
        print(f"Total state samples: {states.shape[0]}, dim={states.shape[1]}")

        # Use first action_dim columns for state percentiles (joint positions)
        action_dim = actions.shape[1]
        state_for_pct = states[:, :action_dim] if states.shape[1] >= action_dim else states
        stat["state_01"] = np.percentile(state_for_pct, 1, axis=0).tolist()
        stat["state_99"] = np.percentile(state_for_pct, 99, axis=0).tolist()

    with open(str(output_path), "w") as f:
        json.dump(stat, f, indent=2)

    print(f"\nstat.json saved to: {output_path}")
    print(f"  action_mean: {[f'{x:.4f}' for x in action_mean]}")
    print(f"  action_std:  {[f'{x:.4f}' for x in action_std]}")
    print(f"  num_samples: {stat['num_samples']}")
    if "state_01" in stat:
        print(f"  state_01:    {[f'{x:.4f}' for x in stat['state_01']]}")
        print(f"  state_99:    {[f'{x:.4f}' for x in stat['state_99']]}")


if __name__ == "__main__":
    main()
