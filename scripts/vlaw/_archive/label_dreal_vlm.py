#!/usr/bin/env python3
"""标注 D_real 数据: 用微调后 16帧 VLM 对所有 D_real 轨迹打分, 生成 D_real+.

输入:
    data/vlaw/rollouts/iter1/LiftPegUpright-v1/
    data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1/
    data/vlaw/rollouts/iter1_lift_inc20/LiftPegUpright-v1/

输出:
    data/vlaw/labeled/iter1_16frame_lora/vlm_labels.h5
    data/vlaw/labeled/iter1_16frame_lora/vlm_label_report.json

Usage:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python scripts/label_dreal_vlm.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASK_ID = "LiftPegUpright-v1"
INSTRUCTION = "Lift the peg and insert it upright into the holder."

MODEL_PATH = "checkpoints/vlaw/reward_model/qwen_vl"
LORA_PATH = "checkpoints/vlaw/reward_model/lora_iter1_16frame"

DATA_DIRS = [
    ("iter1", Path("data/vlaw/rollouts/iter1/LiftPegUpright-v1")),
    ("iter1_highsuc", Path("data/vlaw/rollouts/iter1_highsuc/LiftPegUpright-v1")),
    ("iter1_lift_inc20", Path("data/vlaw/rollouts/iter1_lift_inc20/LiftPegUpright-v1")),
]

OUTPUT_DIR = Path("data/vlaw/labeled/iter1_16frame_lora")
NUM_FRAMES = 16
THRESHOLD = 0.8


def collect_trajectories() -> list[dict]:
    """从所有 D_real HDF5 文件中读取轨迹元信息 (不加载 RGB 到内存)."""
    trajs: list[dict] = []
    for source_label, data_dir in DATA_DIRS:
        if not data_dir.exists():
            print(f"[WARN] 目录不存在, 跳过: {data_dir}")
            continue
        h5_files = sorted(data_dir.glob("*.h5"))
        for h5_path in h5_files:
            with h5py.File(str(h5_path), "r") as f:
                keys = sorted(k for k in f.keys() if k.startswith("traj_"))
                for key in keys:
                    grp = f[key]
                    if "rgb_base" not in grp:
                        continue
                    rgb_shape = grp["rgb_base"].shape  # (T, H, W, 3)
                    env_success_end = bool(grp["env_success"][-1]) if "env_success" in grp else None
                    trajs.append({
                        "h5_path": str(h5_path),
                        "traj_key": key,
                        "source": source_label,
                        "num_frames_total": rgb_shape[0],
                        "resolution": f"{rgb_shape[1]}x{rgb_shape[2]}",
                        "env_success_at_end": env_success_end,
                    })
    return trajs


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Step 1: 收集轨迹列表 ===
    print(f"[Label] Step 1: 收集 D_real 轨迹列表 ...")
    trajs = collect_trajectories()
    print(f"[Label] ✅ 共 {len(trajs)} 条轨迹")
    for src_label, _ in DATA_DIRS:
        n = sum(1 for t in trajs if t["source"] == src_label)
        print(f"  {src_label}: {n} 条")

    # === Step 2: 加载 VLM ===
    print(f"\n[Label] Step 2: 加载 VLM (Qwen3-VL + LoRA) ...")
    from PIL import Image
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel

    cfg = VLAWRewardConfig(
        model_path=MODEL_PATH,
        num_frames=NUM_FRAMES,
        threshold=THRESHOLD,
        use_video_format=True,
        video_fps=2.0,
    )
    model = VLAWRewardModel(cfg)
    model.load_model(lora_path=LORA_PATH)
    print(f"[Label] ✅ VLM 加载完成")

    # === Step 3: 逐条标注 ===
    print(f"\n[Label] Step 3: 开始标注 {len(trajs)} 条轨迹 ...")
    results: list[dict] = []
    for i, traj_info in enumerate(trajs):
        # 读取 RGB 帧
        with h5py.File(traj_info["h5_path"], "r") as f:
            rgb = f[traj_info["traj_key"]]["rgb_base"][:]  # (T, H, W, 3) uint8

        T = rgb.shape[0]
        # 均匀采样 (如果帧数 < 16, 则用全部帧)
        n_sample = min(NUM_FRAMES, T)
        indices = np.linspace(0, T - 1, n_sample, dtype=int)
        frames = [Image.fromarray(rgb[idx]) for idx in indices]

        # VLM 评分
        score = model.score_trajectory(frames, INSTRUCTION)

        result = {
            "idx": i,
            "source": traj_info["source"],
            "h5_file": os.path.basename(traj_info["h5_path"]),
            "h5_path": traj_info["h5_path"],
            "traj_key": traj_info["traj_key"],
            "num_frames_total": T,
            "num_frames_sampled": n_sample,
            "resolution": traj_info["resolution"],
            "p_yes": score["p_yes"],
            "vlm_reward": score["reward"],
            "threshold": score["threshold"],
            "env_success_at_end": traj_info["env_success_at_end"],
        }
        results.append(result)

        tag = "✅" if score["reward"] == 1 else "❌"
        env_tag = "S" if traj_info["env_success_at_end"] else "F"
        print(
            f"  [{i+1:03d}/{len(trajs)}] {tag} p_yes={score['p_yes']:.4f} "
            f"vlm={score['reward']} env={env_tag} "
            f"T={T} src={traj_info['source']} ({traj_info['traj_key']})"
        )

    # === Step 4: 保存结果 ===
    print(f"\n[Label] Step 4: 保存标注结果 ...")

    # 统计
    p_values = np.array([r["p_yes"] for r in results])
    vlm_rewards = np.array([r["vlm_reward"] for r in results])
    env_success = np.array([r["env_success_at_end"] for r in results], dtype=bool)
    n_pos = int(vlm_rewards.sum())
    n_neg = len(results) - n_pos
    n_env_pos = int(env_success.sum())

    # 一致性分析
    tp = int(((vlm_rewards == 1) & env_success).sum())
    tn = int(((vlm_rewards == 0) & ~env_success).sum())
    fp = int(((vlm_rewards == 1) & ~env_success).sum())
    fn = int(((vlm_rewards == 0) & env_success).sum())

    accuracy = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 保存 HDF5
    label_path = OUTPUT_DIR / "vlm_labels.h5"
    with h5py.File(str(label_path), "w") as out_f:
        for r in results:
            key = f"traj_{r['idx']:04d}"
            grp = out_f.create_group(key)
            grp.attrs["p_yes"] = r["p_yes"]
            grp.attrs["vlm_reward"] = r["vlm_reward"]
            grp.attrs["threshold"] = r["threshold"]
            grp.attrs["source"] = r["source"]
            grp.attrs["h5_file"] = r["h5_file"]
            grp.attrs["h5_path"] = r["h5_path"]
            grp.attrs["traj_key"] = r["traj_key"]
            grp.attrs["num_frames_total"] = r["num_frames_total"]
            grp.attrs["num_frames_sampled"] = r["num_frames_sampled"]
            grp.attrs["resolution"] = r["resolution"]
            grp.attrs["env_success_at_end"] = r["env_success_at_end"]

        out_f.attrs["task_id"] = TASK_ID
        out_f.attrs["num_trajectories"] = len(results)
        out_f.attrs["num_positive"] = n_pos
        out_f.attrs["num_negative"] = n_neg
        out_f.attrs["positive_rate"] = n_pos / len(results) if results else 0.0
        out_f.attrs["p_yes_mean"] = float(p_values.mean())
        out_f.attrs["p_yes_std"] = float(p_values.std())
        out_f.attrs["p_yes_min"] = float(p_values.min())
        out_f.attrs["p_yes_max"] = float(p_values.max())
        out_f.attrs["model_path"] = MODEL_PATH
        out_f.attrs["lora_path"] = LORA_PATH
        out_f.attrs["accuracy"] = accuracy
        out_f.attrs["precision"] = precision
        out_f.attrs["recall"] = recall
        out_f.attrs["f1"] = f1
        out_f.attrs["fp_rate"] = fp_rate

    # 保存 JSON 报告
    # 按来源分组统计
    source_stats: dict[str, dict] = {}
    for src_label, _ in DATA_DIRS:
        src_results = [r for r in results if r["source"] == src_label]
        if not src_results:
            continue
        src_p = [r["p_yes"] for r in src_results]
        src_vlm = [r["vlm_reward"] for r in src_results]
        src_env = [r["env_success_at_end"] for r in src_results]
        source_stats[src_label] = {
            "count": len(src_results),
            "vlm_positive": sum(src_vlm),
            "env_positive": sum(src_env),
            "p_yes_mean": round(float(np.mean(src_p)), 4),
            "p_yes_std": round(float(np.std(src_p)), 4),
            "p_yes_min": round(float(np.min(src_p)), 4),
            "p_yes_max": round(float(np.max(src_p)), 4),
        }

    report = {
        "task_id": TASK_ID,
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH,
        "threshold": THRESHOLD,
        "num_frames": NUM_FRAMES,
        "total_trajectories": len(results),
        "vlm_positive (reward=1)": n_pos,
        "vlm_negative (reward=0)": n_neg,
        "positive_rate": round(n_pos / len(results), 4) if results else 0.0,
        "env_success_positive": n_env_pos,
        "p_yes_stats": {
            "mean": round(float(p_values.mean()), 4),
            "std": round(float(p_values.std()), 4),
            "min": round(float(p_values.min()), 4),
            "max": round(float(p_values.max()), 4),
            "median": round(float(np.median(p_values)), 4),
        },
        "confusion_matrix": {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fp_rate": round(fp_rate, 4),
        },
        "source_stats": source_stats,
        "per_trajectory": results,
    }
    report_path = OUTPUT_DIR / "vlm_label_report.json"
    with open(report_path, "w") as jf:
        json.dump(report, jf, indent=2)

    elapsed = time.time() - t0

    # === 打印报告 ===
    print(f"\n{'='*70}")
    print(f"[Label] ✅ D_real 标注完成!")
    print(f"  总轨迹数: {len(results)}")
    print(f"  vlm_reward=1: {n_pos}/{len(results)} ({n_pos/len(results)*100:.1f}%)")
    print(f"  vlm_reward=0: {n_neg}/{len(results)} ({n_neg/len(results)*100:.1f}%)")
    print(f"  env_success=True: {n_env_pos}/{len(results)} ({n_env_pos/len(results)*100:.1f}%)")
    print(f"\n  p_yes 分布:")
    print(f"    mean={p_values.mean():.4f}, std={p_values.std():.4f}")
    print(f"    min={p_values.min():.4f}, max={p_values.max():.4f}, median={np.median(p_values):.4f}")
    print(f"\n  与 env_success 一致性 (Confusion Matrix):")
    print(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"    Accuracy={accuracy:.4f}, Precision={precision:.4f}")
    print(f"    Recall={recall:.4f}, F1={f1:.4f}")
    print(f"    FP Rate={fp_rate:.4f}")
    print(f"\n  按数据来源:")
    for src, st in source_stats.items():
        print(f"    {src}: {st['count']} 条, vlm+={st['vlm_positive']}, env+={st['env_positive']}, p_yes_mean={st['p_yes_mean']:.4f}")
    print(f"\n  输出文件:")
    print(f"    HDF5: {label_path}")
    print(f"    JSON: {report_path}")
    print(f"  耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
