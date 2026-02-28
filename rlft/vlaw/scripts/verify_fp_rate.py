#!/usr/bin/env python3
"""
FP 率验证脚本 — 用 LoRA fine-tuned 模型验证 D_real iter1 数据的 False Positive 率

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward --no-capture-output \
        python rlft/vlaw/scripts/verify_fp_rate.py \
        --model_path checkpoints/vlaw/reward_model/qwen_vl/ \
        --lora_path checkpoints/vlaw/reward_model/lora_iter1/final/ \
        --rollout_dir data/vlaw/rollouts/iter1/ \
        --output_dir data/vlaw/labeled/iter1_lora/ \
        --max_trajectories 20

FP 定义: vlm_reward=1.0 but success_at_end=False
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


TASK_INSTRUCTIONS: dict[str, str] = {
    "LiftPegUpright-v1": (
        "Look at the FINAL frame of this robot manipulation sequence. "
        "The robot's goal is to lift the peg and insert it upright into the holder. "
        "Based on the last frame, has the robot fully completed this task with the peg standing upright in the holder? "
        "Answer only 'yes' or 'no'."
    ),
    "PickCube-v1": (
        "Look at the FINAL frame of this robot manipulation sequence. "
        "The robot's goal is to pick up the red cube from the table. "
        "Based on the last frame, is the robot currently holding the cube above the table? "
        "Answer only 'yes' or 'no'."
    ),
    "StackCube-v1": (
        "Look at the FINAL frame of this robot manipulation sequence. "
        "The robot's goal is to stack the red cube on top of the green cube. "
        "Based on the last frame, is the red cube resting stably on top of the green cube? "
        "Answer only 'yes' or 'no'."
    ),
}


def get_instruction(task_id: str, meta_grp=None) -> str:
    if meta_grp is not None and "instruction" in meta_grp:
        val = meta_grp["instruction"]
        if hasattr(val, "asstr"):
            return str(val.asstr()[()])
        return str(val[()])
    return TASK_INSTRUCTIONS.get(task_id, f"Did the robot successfully complete the {task_id} task?")


def load_traj_frames(traj_grp: h5py.Group) -> np.ndarray:
    if "rgb_base" in traj_grp:
        frames = traj_grp["rgb_base"][:]
    elif "rgb_render" in traj_grp:
        frames = traj_grp["rgb_render"][:]
    else:
        raise KeyError(f"找不到 rgb_base / rgb_render: {list(traj_grp.keys())}")
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8) if frames.max() <= 1.0 \
                 else frames.astype(np.uint8)
    return frames


def sample_frames(frames: np.ndarray, max_frames: int = 16) -> np.ndarray:
    T = len(frames)
    if T <= max_frames:
        return frames
    n_head = max_frames - 3
    if n_head > 0 and T > 3:
        head_idxs = np.linspace(0, T - 4, n_head, dtype=int)
        tail_idxs = np.array([T - 3, T - 2, T - 1])
        idxs = np.concatenate([head_idxs, tail_idxs])
    else:
        idxs = np.array([T - 3, T - 2, T - 1])
    return frames[idxs]


def verify_task(
    task_id: str,
    rollout_dir: Path,
    reward_model,
    max_trajectories: int = 20,
    max_frames: int = 16,
) -> dict:
    task_dir = rollout_dir / task_id
    h5_files = sorted(task_dir.glob("*.h5"))

    if not h5_files:
        print(f"  [WARN] {task_id}: 找不到 HDF5 文件 in {task_dir}")
        return {"task_id": task_id, "n": 0, "vlm_pos": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    results: list[dict] = []

    for h5_path in h5_files:
        if len(results) >= max_trajectories:
            break
        print(f"  处理: {h5_path.name}")

        with h5py.File(h5_path, "r") as f:
            meta_grp = f.get("meta", None)
            instruction = get_instruction(task_id, meta_grp)
            traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
            print(f"    共 {len(traj_keys)} 条轨迹，取前 {min(max_trajectories - len(results), len(traj_keys))} 条")

            for traj_key in traj_keys:
                if len(results) >= max_trajectories:
                    break
                grp = f[traj_key]
                try:
                    frames = load_traj_frames(grp)
                    sampled = sample_frames(frames, max_frames)
                    score = reward_model.score_trajectory(sampled, instruction)

                    env_succ_at_end = 0.0
                    env_succ_once = 0.0
                    if "env_success" in grp:
                        arr = grp["env_success"][:].astype(bool)
                        env_succ_at_end = float(bool(arr[-1]))
                        env_succ_once = float(np.any(arr))

                    vlm_reward = float(score.get("reward", 0.0))
                    p_yes = float(score.get("p_yes", 0.0))
                    results.append({
                        "traj_key": traj_key,
                        "vlm_reward": vlm_reward,
                        "p_yes": p_yes,
                        "env_success_at_end": env_succ_at_end,
                        "env_success_once": env_succ_once,
                    })
                    is_fp = vlm_reward >= 1.0 and env_succ_at_end < 0.5
                    is_tp = vlm_reward >= 1.0 and env_succ_at_end >= 0.5
                    print(
                        f"    {traj_key}: vlm={vlm_reward:.3f} p_yes={p_yes:.4f} "
                        f"env_at_end={env_succ_at_end:.0f} "
                        f"{'[FP!]' if is_fp else '[TP]' if is_tp else '[TN/FN]'}"
                    )
                except Exception as e:
                    print(f"    [WARN] {traj_key}: {e}", file=sys.stderr)

    n = len(results)
    vlm_pos = sum(1 for r in results if r["vlm_reward"] >= 1.0)
    tp = sum(1 for r in results if r["vlm_reward"] >= 1.0 and r["env_success_at_end"] >= 0.5)
    fp = sum(1 for r in results if r["vlm_reward"] >= 1.0 and r["env_success_at_end"] < 0.5)
    fn = sum(1 for r in results if r["vlm_reward"] < 1.0 and r["env_success_at_end"] >= 0.5)
    tn = sum(1 for r in results if r["vlm_reward"] < 1.0 and r["env_success_at_end"] < 0.5)
    env_pos = sum(1 for r in results if r["env_success_at_end"] >= 0.5)
    fp_rate = fp / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    p_yes_values = [r["p_yes"] for r in results]

    print(f"\n  [{task_id}] n={n}, vlm_pos={vlm_pos}, env_pos={env_pos}")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"    FP 率={fp_rate:.1%}, Precision={precision:.2f}")
    print(f"    p_yes: min={min(p_yes_values):.4f}, max={max(p_yes_values):.4f}, mean={np.mean(p_yes_values):.4f}")

    return {
        "task_id": task_id,
        "n": n,
        "vlm_pos": vlm_pos,
        "env_pos": env_pos,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_rate": fp_rate,
        "precision": precision,
        "p_yes_min": float(min(p_yes_values)) if p_yes_values else 0.0,
        "p_yes_max": float(max(p_yes_values)) if p_yes_values else 0.0,
        "p_yes_mean": float(np.mean(p_yes_values)) if p_yes_values else 0.0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA VLM FP 率验证")
    parser.add_argument("--model_path", type=str, default="checkpoints/vlaw/reward_model/qwen_vl/")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--rollout_dir", type=str, default="data/vlaw/rollouts/iter1/")
    parser.add_argument("--output_dir", type=str, default="data/vlaw/labeled/iter1_lora/")
    parser.add_argument("--tasks", type=str, default="LiftPegUpright-v1",
                        help="任务列表（默认 Lift-only；PickCube/StackCube deferred）")
    parser.add_argument("--max_trajectories", type=int, default=20, help="每任务最多处理轨迹数")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    workspace = Path(__file__).parents[3]
    model_path = workspace / args.model_path
    rollout_dir = workspace / args.rollout_dir
    output_dir = workspace / args.output_dir
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    lora_abs = str(workspace / args.lora_path) if args.lora_path else None

    print("=" * 60)
    print("[FP验证] LoRA VLM 奖励模型 FP 率验证")
    print(f"  model_path      : {model_path}")
    print(f"  lora_path       : {lora_abs or '(none - 基础模型)'}")
    print(f"  rollout_dir     : {rollout_dir}")
    print(f"  tasks           : {tasks}")
    print(f"  max_trajectories: {args.max_trajectories}")
    print(f"  threshold       : {args.threshold}")
    print("=" * 60)

    # 路径检查
    print("\n[步骤1] 路径检查")
    ok = True
    for p, name in [(model_path, "VLM base"), (rollout_dir, "rollout_dir")]:
        exists = p.exists()
        print(f"  {'✅' if exists else '❌'} {name}: {p}")
        if not exists:
            ok = False
    if lora_abs:
        lp = Path(lora_abs)
        exists = lp.exists()
        print(f"  {'✅' if exists else '❌'} LoRA adapter: {lp}")
        if not exists:
            ok = False
    if not ok:
        print("[ERROR] 路径检查失败，退出")
        sys.exit(1)

    # 加载模型
    print(f"\n[步骤2] 加载 VLM 模型")
    sys.path.insert(0, str(workspace))
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel  # type: ignore

    t0 = time.time()
    cfg = VLAWRewardConfig(
        model_path=str(model_path),
        threshold=args.threshold,
        device="cuda",
    )
    reward_model = VLAWRewardModel(cfg)
    reward_model.load_model(lora_path=lora_abs)
    print(f"  模型已加载 ({time.time() - t0:.1f}s)")

    # 逐任务推理
    print(f"\n[步骤3] 推理 (每任务前 {args.max_trajectories} 条)")
    all_stats: list[dict] = []

    for task_id in tasks:
        print(f"\n--- {task_id} ---")
        t0 = time.time()
        stats = verify_task(
            task_id=task_id,
            rollout_dir=rollout_dir,
            reward_model=reward_model,
            max_trajectories=args.max_trajectories,
            max_frames=args.max_frames,
        )
        stats["elapsed"] = time.time() - t0
        all_stats.append(stats)

        # 保存该任务结果
        task_out = output_dir / task_id
        task_out.mkdir(parents=True, exist_ok=True)
        out_path = task_out / f"{task_id}_lora_rewards.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats["results"], f, ensure_ascii=False, indent=2)
        print(f"  [SAVE] {out_path}")

    # 汇总报告
    print("\n" + "=" * 60)
    print("[步骤4] FP 率汇总报告")
    print("=" * 60)

    total_n = sum(s["n"] for s in all_stats)
    total_fp = sum(s["fp"] for s in all_stats)
    total_fp_rate = total_fp / total_n if total_n > 0 else 0.0

    pass_gate = True
    for s in all_stats:
        fp_ok = s["fp_rate"] < 0.20
        if not fp_ok:
            pass_gate = False
        status = "✅ OK" if fp_ok else "❌ FAIL"
        print(
            f"  {s['task_id']}: n={s['n']}, FP={s['fp']}/{s['n']} "
            f"FP率={s['fp_rate']:.1%} {status} "
            f"| TP={s['tp']} FN={s['fn']} TN={s['tn']} "
            f"| p_yes_max={s['p_yes_max']:.4f}"
        )

    print(f"\n  总计: n={total_n}, FP={total_fp}/{total_n}, 全局FP率={total_fp_rate:.1%}")
    gate_status = "✅ PASS (<20%)" if pass_gate else "❌ FAIL (≥20%)"
    print(f"  质量门控: {gate_status}")
    print("=" * 60)

    # 保存汇总
    summary = {
        "lora_path": lora_abs,
        "max_trajectories": args.max_trajectories,
        "threshold": args.threshold,
        "total_n": total_n,
        "total_fp": total_fp,
        "total_fp_rate": total_fp_rate,
        "gate_pass": pass_gate,
        "per_task": {s["task_id"]: {k: v for k, v in s.items() if k != "results"} for s in all_stats},
    }
    summary_path = output_dir / "fp_verification_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  [SAVE] 汇总: {summary_path}")

    sys.exit(0 if pass_gate else 1)


if __name__ == "__main__":
    main()
