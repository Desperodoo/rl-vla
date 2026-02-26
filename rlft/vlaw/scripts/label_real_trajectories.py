#!/usr/bin/env python3
"""
P3.2 VLM 零样本奖励标注脚本 — 处理 D_real HDF5 格式

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python \
        rlft/vlaw/scripts/label_real_trajectories.py \
        --rollout_dir data/vlaw/rollouts/iter1 \
        --output_dir data/vlaw/labeled/iter1 \
        --iter_id 1

HDF5 结构 (我们的格式):
    iter1/{task}/{task}_real_{timestamp}.h5
      ├── meta/
      │     ├── task_id
      │     └── instruction
      └── traj_XXXX/
            ├── rgb_base       (T, 192, 192, 3)  uint8
            ├── rgb_render     (T, 192, 192, 3)  uint8
            ├── state          (T, 25)            float32
            ├── obs_agent      (T, 25)            float32
            ├── actions        (T, 7)             float32
            └── env_success    (T,)               float32

输出:
    output_dir/{task}_vlm_rewards.json  (每条轨迹的奖励分值)
    output_dir/{task}_vlm_rewards.h5    (同上，写回 HDF5 供 P5 使用)
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
import torch


# ── TASK 指令映射 ─────────────────────────────────────────────────────────────

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


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def get_instruction(task_id: str, meta_grp=None) -> str:
    """获取任务指令, 优先从 HDF5 meta 读取, 其次查映射表."""
    if meta_grp is not None and "instruction" in meta_grp:
        val = meta_grp["instruction"]
        if hasattr(val, "asstr"):
            return str(val.asstr()[()])
        return str(val[()])
    return TASK_INSTRUCTIONS.get(
        task_id,
        f"Did the robot successfully complete the {task_id} task?"
    )


def load_traj_frames(traj_grp: h5py.Group) -> np.ndarray:
    """从 traj 组加载 RGB 帧, 返回 (T, H, W, 3) uint8."""
    if "rgb_base" in traj_grp:
        frames = traj_grp["rgb_base"][:]            # (T, H, W, 3)
    elif "rgb_render" in traj_grp:
        frames = traj_grp["rgb_render"][:]
    else:
        raise KeyError(f"traj 组中找不到 rgb_base / rgb_render: {list(traj_grp.keys())}")
    
    # 确保 uint8
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8) if frames.max() <= 1.0 \
                 else frames.astype(np.uint8)
    return frames


def process_hdf5_file(
    h5_path: Path,
    reward_model,
    task_id: str,
    max_frames: int = 16,
    verbose: bool = True,
) -> list[dict]:
    """处理单个 HDF5 文件中所有轨迹，返回标注结果列表."""
    results: list[dict] = []

    with h5py.File(h5_path, "r") as f:
        # 读取指令
        meta_grp = f.get("meta", None)
        instruction = get_instruction(task_id, meta_grp)
        if verbose:
            print(f"  [INFO] 指令: {instruction[:80]}...")

        # 遍历所有 traj_XXXX
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        if verbose:
            print(f"  [INFO] 共 {len(traj_keys)} 条轨迹")

        for traj_key in traj_keys:
            grp = f[traj_key]
            try:
                frames = load_traj_frames(grp)         # (T, H, W, 3)

                # 采样策略：必须包含最后3帧，其余均匀填充到 max_frames
                T = len(frames)
                if T <= max_frames:
                    sampled = frames
                else:
                    # 最后3帧 + 从前面均匀取 max_frames-3 帧
                    n_head = max_frames - 3
                    if n_head > 0 and T > 3:
                        head_idxs = np.linspace(0, T - 4, n_head, dtype=int)
                        tail_idxs = np.array([T - 3, T - 2, T - 1])
                        idxs = np.concatenate([head_idxs, tail_idxs])
                    else:
                        idxs = np.array([T - 3, T - 2, T - 1])
                    sampled = frames[idxs]

                # VLM 评分
                score_result = reward_model.score_trajectory(sampled, instruction)

                # 读取 env_success (作为 ground-truth 备用，同时记录两种语义)
                env_success_once: float = 0.0
                env_success_at_end: float = 0.0
                if "env_success" in grp:
                    env_succ_arr = grp["env_success"][:].astype(bool)
                    env_success_once   = float(np.any(env_succ_arr))
                    env_success_at_end = float(bool(env_succ_arr[-1]))

                entry = {
                    "traj_key": traj_key,
                    "task_id": task_id,
                    "T": int(T),
                    "vlm_reward": float(score_result.get("reward", 0.0)),
                    "vlm_yes_prob": float(score_result.get("p_yes", 0.0)),
                    "vlm_success": bool(score_result.get("reward", 0) > 0),
                    "env_success_once": env_success_once,
                    "env_success_at_end": env_success_at_end,
                    # 向后兼容：主 env_success 用 at_end（VLAW 论文语义）
                    "env_success": env_success_at_end,
                    "instruction": instruction,
                }
                results.append(entry)

                if verbose:
                    print(
                        f"    {traj_key}: T={T} → "
                        f"vlm_reward={entry['vlm_reward']:.4f} "
                        f"vlm_yes={entry['vlm_yes_prob']:.4f} "
                        f"env_succ_at_end={env_success_at_end:.0f} "
                        f"env_succ_once={env_success_once:.0f}"
                    )

            except Exception as e:
                print(f"    [WARN] {traj_key} 处理失败: {e}", file=sys.stderr)
                results.append({
                    "traj_key": traj_key,
                    "task_id": task_id,
                    "T": 0,
                    "vlm_reward": 0.0,
                    "vlm_yes_prob": 0.0,
                    "vlm_success": False,
                    "env_success_once": 0.0,
                    "env_success_at_end": 0.0,
                    "env_success": 0.0,
                    "error": str(e),
                })

    return results


def write_results(results: list[dict], output_dir: Path, task_id: str) -> None:
    """将标注结果写入 JSON 和 HDF5。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{task_id}_vlm_rewards.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] JSON: {json_path}")

    # HDF5 — 写 vlm_reward 和 vlm_yes_prob
    h5_path = output_dir / f"{task_id}_vlm_rewards.h5"
    with h5py.File(h5_path, "w") as f:
        for item in results:
            grp = f.require_group(item["traj_key"])
            grp.attrs["vlm_reward"] = item["vlm_reward"]
            grp.attrs["vlm_yes_prob"] = item["vlm_yes_prob"]
            grp.attrs["vlm_success"] = int(item["vlm_success"])
            grp.attrs["env_success"] = item["env_success"]
            grp.attrs["env_success_once"] = item.get("env_success_once", item["env_success"])
            grp.attrs["env_success_at_end"] = item.get("env_success_at_end", item["env_success"])
            grp.attrs["task_id"] = item["task_id"]
    print(f"  [SAVE] HDF5: {h5_path}")

    # 统计摘要
    vlm_rewards = [r["vlm_reward"] for r in results if "error" not in r]
    vlm_succs = [r["vlm_success"] for r in results if "error" not in r]
    env_succs_at_end = [r.get("env_success_at_end", r["env_success"]) for r in results if "error" not in r]
    env_succs_once   = [r.get("env_success_once",   r["env_success"]) for r in results if "error" not in r]
    errors = sum(1 for r in results if "error" in r)
    print(
        f"  [STAT] {task_id}: "
        f"n={len(vlm_rewards)} "
        f"vlm_success_rate={sum(vlm_succs)/max(len(vlm_succs),1):.1%} "
        f"env_success_at_end_rate={sum(env_succs_at_end)/max(len(env_succs_at_end),1):.1%} "
        f"env_success_once_rate={sum(env_succs_once)/max(len(env_succs_once),1):.1%} "
        f"mean_reward={np.mean(vlm_rewards):.4f} "
        f"errors={errors}"
    )


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 零样本奖励标注 D_real")
    parser.add_argument("--rollout_dir", type=str, default="data/vlaw/rollouts/iter1",
                        help="D_real rollout 目录 (包含 {task}/ 子目录)")
    parser.add_argument("--output_dir", type=str, default="data/vlaw/labeled/iter1",
                        help="标注结果输出目录")
    parser.add_argument("--iter_id", type=int, default=1, help="迭代轮次")
    parser.add_argument("--tasks", type=str, default="LiftPegUpright-v1,PickCube-v1,StackCube-v1",
                        help="任务列表, 逗号分隔")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/vlaw/reward_model/qwen_vl",
                        help="VLM 模型路径")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter 路径 (None = 不加载 LoRA, 使用基础模型)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="成功/失败判定阈值 (α)")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="每条轨迹最多采样帧数")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅检查数据结构，不加载 VLM")
    args = parser.parse_args()

    # 路径对齐
    workspace = Path(__file__).parents[3]   # /home/wjz/rl-vla
    rollout_dir = workspace / args.rollout_dir
    output_dir = workspace / args.output_dir
    model_path = workspace / args.model_path
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"[P3.2] VLM 奖励标注开始 — iter_id={args.iter_id}")
    print(f"  rollout_dir : {rollout_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"  tasks       : {tasks}")
    print(f"  model_path  : {model_path}")
    print(f"  threshold   : {args.threshold}")
    print(f"  max_frames  : {args.max_frames}")

    # ── Dry-run: 只验证 HDF5 结构 ───────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] 仅验证数据结构...")
        for task_id in tasks:
            task_dir = rollout_dir / task_id
            h5_files = sorted(task_dir.glob("*.h5"))
            if not h5_files:
                print(f"  [WARN] {task_id}: 找不到 HDF5 文件")
                continue
            h5_path = h5_files[0]
            with h5py.File(h5_path, "r") as f:
                traj_keys = [k for k in f.keys() if k.startswith("traj_")]
                print(f"  {task_id}: {len(traj_keys)} trajs in {h5_path.name}")
                if traj_keys:
                    grp = f[traj_keys[0]]
                    for k in grp.keys():
                        print(f"    {k}: {grp[k].shape} {grp[k].dtype}")
        print("[DRY RUN] 完成")
        return

    # ── 加载 VLM 模型 ──────────────────────────────────────────────────────
    sys.path.insert(0, str(workspace))
    from rlft.vlaw.reward_model import VLAWRewardConfig, VLAWRewardModel

    print(f"\n[P3.2] 加载 VLM 模型: {model_path}")
    t0 = time.time()
    cfg = VLAWRewardConfig(
        model_path=str(model_path),
        threshold=args.threshold,
        device="cuda",
    )
    reward_model = VLAWRewardModel(cfg)
    lora_abs = str(workspace / args.lora_path) if args.lora_path else None
    if lora_abs:
        print(f"  LoRA adapter : {lora_abs}")
    reward_model.load_model(lora_path=lora_abs)
    print(f"  模型加载完成 ({time.time()-t0:.1f}s)")

    # ── 逐任务标注 ─────────────────────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}

    for task_id in tasks:
        task_dir = rollout_dir / task_id
        h5_files = sorted(task_dir.glob("*.h5"))

        if not h5_files:
            print(f"\n[WARN] {task_id}: 找不到 HDF5 文件，跳过")
            continue

        print(f"\n[P3.2] 处理 {task_id} ({len(h5_files)} 个 HDF5 文件)...")
        t0 = time.time()
        task_results: list[dict] = []

        for h5_path in h5_files:
            print(f"  处理: {h5_path.name}")
            results = process_hdf5_file(
                h5_path,
                reward_model,
                task_id,
                max_frames=args.max_frames,
            )
            task_results.extend(results)

        print(f"  耗时: {time.time()-t0:.1f}s")

        # 写入结果
        write_results(task_results, output_dir / task_id, task_id)
        all_results[task_id] = task_results

    # ── 全局摘要 ──────────────────────────────────────────────────────────
    print("\n[P3.2] === 全局摘要 ===")
    total_trajs = sum(len(v) for v in all_results.values())
    total_vlm_succ = sum(
        sum(1 for r in v if r.get("vlm_success", False))
        for v in all_results.values()
    )
    print(f"  总轨迹数      : {total_trajs}")
    print(f"  VLM 成功数    : {total_vlm_succ} ({total_vlm_succ/max(total_trajs,1):.1%})")

    # 写全局汇总 JSON
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "iter_id": args.iter_id,
        "tasks": tasks,
        "total_trajs": total_trajs,
        "total_vlm_success": total_vlm_succ,
        "per_task": {
            task_id: {
                "n": len(v),
                "vlm_success_rate": sum(1 for r in v if r.get("vlm_success")) / max(len(v), 1),
                "env_success_at_end_rate": sum(r.get("env_success_at_end", r["env_success"]) for r in v) / max(len(v), 1),
                "env_success_once_rate":   sum(r.get("env_success_once",   r["env_success"]) for r in v) / max(len(v), 1),
                # 向后兼容
                "env_success_rate": sum(r["env_success"] for r in v) / max(len(v), 1),
                "mean_vlm_reward": float(np.mean([r["vlm_reward"] for r in v]))
                    if v else 0.0,
            }
            for task_id, v in all_results.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  [SAVE] 摘要: {summary_path}")
    print("[P3.2] 标注完成 ✅")


if __name__ == "__main__":
    main()
