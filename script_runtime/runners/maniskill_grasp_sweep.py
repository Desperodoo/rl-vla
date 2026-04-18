"""Sweep ManiSkill grasp parameters and render a simple heatmap report."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from script_runtime.validation.maniskill_rollout import run_maniskill_validation
from script_runtime.session import load_runtime_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep oracle grasp parameters in ManiSkill.")
    parser.add_argument(
        "--config",
        default="script_runtime/configs/tasks/pick_cube_maniskill.yaml",
        help="Base validation config.",
    )
    parser.add_argument(
        "--heights",
        default="0.015,0.025,0.035,0.045,0.055",
        help="Comma-separated grasp height offsets (meters).",
    )
    parser.add_argument(
        "--gripper-steps",
        default="4,8,12",
        help="Comma-separated gripper hold substeps.",
    )
    parser.add_argument(
        "--out-dir",
        default="script_runtime/artifacts/maniskill_grasp_sweep",
        help="Where to store JSON and heatmap.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Optional override for validation.num_episodes. Use 1 for quick smoke sweeps.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    base = load_runtime_config(args.config)
    if args.num_episodes > 0:
        base.setdefault("validation", {})["num_episodes"] = args.num_episodes
    heights = [float(x) for x in args.heights.split(",") if x.strip()]
    gripper_steps = [int(x) for x in args.gripper_steps.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    matrix: List[List[int]] = []
    for steps in gripper_steps:
        row: List[int] = []
        for height in heights:
            config = deepcopy(base)
            config.setdefault("maniskill", {})["grasp_height_offset"] = height
            config.setdefault("maniskill", {})["gripper_substeps"] = steps
            summary = run_maniskill_validation_with_overrides(config)
            successes = int(summary["sim_successes"])
            row.append(successes)
            results.append(
                {
                    "gripper_substeps": steps,
                    "grasp_height_offset": height,
                    "runtime_successes": int(summary["runtime_successes"]),
                    "sim_successes": successes,
                    "num_episodes": int(summary["num_episodes"]),
                }
            )
        matrix.append(row)

    json_path = out_dir / "sweep_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    heatmap_path = out_dir / "sweep_heatmap.png"
    _render_heatmap(matrix, heights, gripper_steps, heatmap_path)
    print(f"results_json={json_path}")
    print(f"heatmap_png={heatmap_path}")
    return 0


def run_maniskill_validation_with_overrides(config: Dict[str, object]) -> Dict[str, object]:
    from script_runtime.adapters import ManiSkillBridge
    from script_runtime.session import build_pick_place_session

    validation_cfg = config.get("validation", {}) if isinstance(config, dict) else {}
    num_episodes = int(validation_cfg.get("num_episodes", 3))
    base_seed = validation_cfg.get("seed")
    runtime_successes = 0
    sim_successes = 0

    sim_cfg = config.get("maniskill", {}) if isinstance(config, dict) else {}
    for episode_index in range(num_episodes):
        bridge = ManiSkillBridge(
            env_id=str(sim_cfg.get("env_id", "PickCube-v1")),
            obs_mode=str(sim_cfg.get("obs_mode", "state")),
            control_mode=str(sim_cfg.get("control_mode", "pd_ee_pose")),
            render_mode=sim_cfg.get("render_mode"),
            sim_backend=str(sim_cfg.get("sim_backend", "physx_cpu")),
            num_envs=int(sim_cfg.get("num_envs", 1)),
            max_episode_steps=sim_cfg.get("max_episode_steps"),
            cached_reset_options=dict(sim_cfg.get("reset_options", {})),
            reset_seed=(None if base_seed is None else int(base_seed) + episode_index),
            grasp_height_offset=float(sim_cfg.get("grasp_height_offset", 0.0)),
            pregrasp_height_offset=float(sim_cfg.get("pregrasp_height_offset", 0.08)),
            place_height_offset=float(sim_cfg.get("place_height_offset", 0.06)),
            pose_substeps=int(sim_cfg.get("pose_substeps", 60)),
            gripper_substeps=int(sim_cfg.get("gripper_substeps", 15)),
            settle_substeps=int(sim_cfg.get("settle_substeps", 25)),
            position_tolerance=float(sim_cfg.get("position_tolerance", 0.01)),
            guarded_close_descend_delta=float(sim_cfg.get("guarded_close_descend_delta", 0.02)),
            guarded_close_descent_steps=int(sim_cfg.get("guarded_close_descent_steps", 6)),
            guarded_close_hold_steps=int(sim_cfg.get("guarded_close_hold_steps", 10)),
            grasp_phase_descend_segments=int(sim_cfg.get("grasp_phase_descend_segments", 5)),
            grasp_phase_close_hold_steps=int(sim_cfg.get("grasp_phase_close_hold_steps", 12)),
            grasp_phase_lift_delta=float(sim_cfg.get("grasp_phase_lift_delta", 0.03)),
            grasp_phase_lift_steps=int(sim_cfg.get("grasp_phase_lift_steps", 6)),
            lock_orientation=bool(sim_cfg.get("lock_orientation", True)),
            delta_translation_limit=float(sim_cfg.get("delta_translation_limit", 0.02)),
            delta_rotation_limit=float(sim_cfg.get("delta_rotation_limit", 0.05)),
            env_kwargs=dict(sim_cfg.get("env_kwargs", {})),
        )
        session = build_pick_place_session(config, sdk_bridge=bridge)
        session.adapters["maniskill"] = bridge
        try:
            result = session.run()
            status = bridge.get_status()
        finally:
            session.shutdown()
        if result.status.value == "SUCCESS":
            runtime_successes += 1
        if bool(status.get("success", False)):
            sim_successes += 1

    return {
        "num_episodes": num_episodes,
        "runtime_successes": runtime_successes,
        "sim_successes": sim_successes,
    }


def _render_heatmap(matrix: List[List[int]], heights: List[float], steps: List[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(heights)))
    ax.set_xticklabels([f"{h:.3f}" for h in heights])
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([str(s) for s in steps])
    ax.set_xlabel("grasp_height_offset (m)")
    ax.set_ylabel("gripper_substeps")
    ax.set_title("ManiSkill Grasp Sweep: Sim Success Count")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
