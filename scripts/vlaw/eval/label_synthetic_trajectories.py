#!/usr/bin/env python3
"""T-VLM-LABEL-001 — 用 LoRA VLM 标注合成轨迹

将 Imagination 引擎生成的 latent-only 合成轨迹解码为 RGB，
然后使用 fine-tuned VLM 打分，生成 D_syn+ 用于策略训练。

流程:
    1. 加载 SVD VAE (仅解码器，~1.5 GB)
    2. 加载 Qwen3-VL + LoRA adapter (~10 GB)
    3. 对每条轨迹:
       - 均匀采样 num_frames 帧索引
       - 解码 VAE latent → RGB (384×192 → 裁剪为 192×192 base camera)
       - VLM 评分 → p_yes, reward (α=0.8 阈值)
    4. 保存 JSON + HDF5 结果

用法:
    CUDA_VISIBLE_DEVICES=6 python scripts/vlaw/eval/label_synthetic_trajectories.py \
        --h5_path data/vlaw/synthetic/iter1_wm_real/synthetic_iter1_final_1772339809.h5 \
        --output_dir data/vlaw/labeled/synthetic_iter1_wm_real \
        --lora_path checkpoints/vlaw/reward_model/lora_iter1_16frame \
        --num_frames 16 --threshold 0.8

GPUs:
    单卡即可: VAE (~1.5GB) + VLM (~10GB) ~ 11.5GB < 24GB (4090)
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

TASK_INSTRUCTIONS: dict[str, str] = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


# ── VAE 加载 & 解码 ──────────────────────────────────────────────────────────

def load_vae(vae_path: str, device: str = "cuda") -> torch.nn.Module:
    """加载 SVD AutoencoderKLTemporalDecoder (仅用其解码器)."""
    from diffusers.models import AutoencoderKLTemporalDecoder

    print(f"[VAE] 加载: {vae_path}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path, torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[VAE] 就绪 | device={device} | VRAM={mem_gb:.1f} GB")
    return vae


@torch.inference_mode()
def decode_latent_frames(
    vae: torch.nn.Module,
    latents: np.ndarray,
    frame_indices: np.ndarray,
    device: str = "cuda",
    decode_chunk_size: int = 4,
) -> np.ndarray:
    """解码指定帧的 latent 为 RGB.

    Args:
        vae: VAE 解码器
        latents: (T, 4, 48, 24) float16 全轨迹 latent
        frame_indices: 采样的帧索引 (num_frames,)
        device: GPU 设备
        decode_chunk_size: 分块解码大小

    Returns:
        frames: (num_frames, 192, 192, 3) uint8 — base camera RGB
    """
    selected = torch.from_numpy(latents[frame_indices]).to(device).to(torch.float16)
    # selected: (N, 4, 48, 24) — 2-cam 纵向拼接的 latent

    scaling_factor = vae.config.scaling_factor  # 0.18215
    decoded_list = []
    for i in range(0, selected.shape[0], decode_chunk_size):
        chunk = selected[i : i + decode_chunk_size] / scaling_factor
        # VAE decode expects (B, C, H, W), output (B, C, H, W)
        out = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        decoded_list.append(out)

    decoded = torch.cat(decoded_list, dim=0)  # (N, 3, 384, 192)
    # 转为 uint8
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
    decoded = decoded.float().cpu().numpy()  # (N, 3, 384, 192)
    decoded = decoded.transpose(0, 2, 3, 1).astype(np.uint8)  # (N, 384, 192, 3)

    # 裁剪 base camera (上半部分: 0:192)
    base_frames = decoded[:, :192, :, :]  # (N, 192, 192, 3)
    return base_frames


# ── 主流程 ────────────────────────────────────────────────────────────────────

def label_synthetic_hdf5(
    h5_path: str,
    vae: torch.nn.Module,
    reward_model,
    instruction: str,
    num_frames: int = 16,
    device: str = "cuda",
    verbose: bool = True,
) -> list[dict]:
    """标注一个合成 HDF5 文件中的所有轨迹."""
    results: list[dict] = []
    t0 = time.time()

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        n_total = len(traj_keys)
        if verbose:
            print(f"[LABEL] 总轨迹数: {n_total} | 指令: {instruction[:80]}")

        for idx, traj_key in enumerate(traj_keys):
            grp = f[traj_key]
            latents = grp["latent"][:]  # (T, 4, 48, 24) float16

            T = latents.shape[0]
            n = min(num_frames, T)
            frame_idx = np.linspace(0, T - 1, n, dtype=int)

            try:
                # 1) VAE 解码
                rgb_frames = decode_latent_frames(
                    vae, latents, frame_idx, device=device
                )  # (n, 192, 192, 3)

                # 2) VLM 评分
                score = reward_model.score_trajectory(rgb_frames, instruction)

                entry = {
                    "traj_key": traj_key,
                    "T": int(T),
                    "n_frames_sampled": int(n),
                    "vlm_reward": float(score["reward"]),
                    "vlm_yes_prob": float(score["p_yes"]),
                    "vlm_success": bool(score["reward"] > 0),
                    "threshold": float(score["threshold"]),
                    "instruction": instruction,
                    "source": "synthetic_iter1_wm_real",
                }
                results.append(entry)

                if verbose and ((idx + 1) % 10 == 0 or idx == 0):
                    elapsed = time.time() - t0
                    n_success = sum(1 for r in results if r["vlm_success"])
                    print(
                        f"  [{idx+1}/{n_total}] {traj_key}: "
                        f"p_yes={score['p_yes']:.4f} "
                        f"reward={score['reward']} | "
                        f"cumulative_success={n_success}/{len(results)} "
                        f"({elapsed:.0f}s)"
                    )
            except Exception as e:
                print(f"  [ERROR] {traj_key}: {e}", file=sys.stderr)
                results.append({
                    "traj_key": traj_key,
                    "T": int(T),
                    "vlm_reward": 0.0,
                    "vlm_yes_prob": 0.0,
                    "vlm_success": False,
                    "threshold": 0.8,
                    "error": str(e),
                })

    elapsed = time.time() - t0
    return results


def save_results(
    results: list[dict],
    output_dir: Path,
    prefix: str = "LiftPegUpright-v1",
) -> dict:
    """保存标注结果并返回统计摘要."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{prefix}_vlm_rewards.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON → {json_path}")

    # HDF5
    h5_path = output_dir / f"{prefix}_vlm_rewards.h5"
    with h5py.File(h5_path, "w") as f:
        for item in results:
            grp = f.require_group(item["traj_key"])
            grp.attrs["vlm_reward"] = item.get("vlm_reward", 0.0)
            grp.attrs["vlm_yes_prob"] = item.get("vlm_yes_prob", 0.0)
            grp.attrs["vlm_success"] = int(item.get("vlm_success", False))
            grp.attrs["threshold"] = item.get("threshold", 0.8)
    print(f"[SAVE] HDF5 → {h5_path}")

    # 统计
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    n_success = sum(1 for r in valid if r["vlm_success"])
    p_yes_arr = np.array([r["vlm_yes_prob"] for r in valid])
    n_errors = len(results) - n_valid

    summary = {
        "n_total": len(results),
        "n_valid": n_valid,
        "n_errors": n_errors,
        "n_vlm_success": n_success,
        "vlm_success_rate": n_success / max(n_valid, 1),
        "p_yes_mean": float(p_yes_arr.mean()) if n_valid > 0 else 0.0,
        "p_yes_std": float(p_yes_arr.std()) if n_valid > 0 else 0.0,
        "p_yes_median": float(np.median(p_yes_arr)) if n_valid > 0 else 0.0,
        "p_yes_max": float(p_yes_arr.max()) if n_valid > 0 else 0.0,
        "p_yes_min": float(p_yes_arr.min()) if n_valid > 0 else 0.0,
    }

    # 保存统计
    stat_path = output_dir / f"{prefix}_label_summary.json"
    with open(stat_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary → {stat_path}")

    print(f"\n{'='*60}")
    print(f"  标注完成: {n_valid}/{len(results)} 条有效")
    print(f"  VLM success: {n_success} ({summary['vlm_success_rate']:.1%})")
    print(f"  p_yes: mean={summary['p_yes_mean']:.4f} ± {summary['p_yes_std']:.4f}")
    print(f"  p_yes: median={summary['p_yes_median']:.4f}, "
          f"min={summary['p_yes_min']:.4f}, max={summary['p_yes_max']:.4f}")
    print(f"  Errors: {n_errors}")
    print(f"{'='*60}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T-VLM-LABEL-001: LoRA VLM 标注合成轨迹"
    )
    parser.add_argument(
        "--h5_path", type=str, required=True,
        help="合成轨迹 HDF5 文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/vlaw/labeled/synthetic_iter1_wm_real",
        help="输出目录"
    )
    parser.add_argument(
        "--task", type=str, default="LiftPegUpright-v1",
        help="任务名 (用于指令和文件名)"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="checkpoints/vlaw/reward_model/qwen_vl",
        help="Qwen3-VL base model 路径"
    )
    parser.add_argument(
        "--lora_path", type=str,
        default="checkpoints/vlaw/reward_model/lora_iter1_16frame",
        help="LoRA adapter 路径"
    )
    parser.add_argument(
        "--vae_path", type=str,
        default="checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae",
        help="SVD VAE 路径"
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 解析路径
    h5_path = Path(args.h5_path)
    if not h5_path.is_absolute():
        h5_path = ROOT / h5_path
    output_dir = ROOT / args.output_dir
    model_path = str(ROOT / args.model_path)
    lora_path = str(ROOT / args.lora_path)
    vae_path = str(ROOT / args.vae_path)

    instruction = TASK_INSTRUCTIONS.get(
        args.task, f"Complete the {args.task} task."
    )

    print(f"{'='*60}")
    print(f"  T-VLM-LABEL-001: 合成轨迹 VLM 标注")
    print(f"  HDF5     : {h5_path}")
    print(f"  Output   : {output_dir}")
    print(f"  Task     : {args.task}")
    print(f"  VLM Base : {model_path}")
    print(f"  LoRA     : {lora_path}")
    print(f"  VAE      : {vae_path}")
    print(f"  Frames   : {args.num_frames}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device   : {args.device}")
    print(f"{'='*60}")

    # Step 1: 加载 VAE
    t0 = time.time()
    vae = load_vae(vae_path, device=args.device)
    print(f"[TIME] VAE 加载: {time.time()-t0:.1f}s\n")

    # Step 2: 加载 VLM + LoRA
    t1 = time.time()
    from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig

    vlm_cfg = VLAWRewardConfig(
        model_path=model_path,
        threshold=args.threshold,
        device=args.device,
        num_frames=args.num_frames,
    )
    vlm = VLAWRewardModel(vlm_cfg)
    vlm.load_model(lora_path=lora_path)
    print(f"[TIME] VLM 加载: {time.time()-t1:.1f}s")

    # VRAM 状态
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"[GPU] 当前 VRAM 使用: {mem:.1f} GB\n")

    # Step 3: 标注
    t2 = time.time()
    results = label_synthetic_hdf5(
        str(h5_path), vae, vlm, instruction,
        num_frames=args.num_frames,
        device=args.device,
    )
    label_time = time.time() - t2
    print(f"\n[TIME] 标注耗时: {label_time:.0f}s ({label_time/max(len(results),1):.1f}s/traj)")

    # Step 4: 保存
    summary = save_results(results, output_dir, prefix=args.task)

    # 清理
    vlm.unload_model()
    del vae
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    print(f"\n[DONE] 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
