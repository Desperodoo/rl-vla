#!/usr/bin/env python3
"""VLM 轨迹标注 — 统一稳定入口脚本.

统一处理 D_real (HDF5 RGB) 和 D_syn (HDF5 latent → VAE 解码) 两种格式。
合并自: label_real_trajectories.py / label_synthetic_trajectories.py / label_dreal_vlm.py
已包含:
  - BUG-011 修复: yes/no token 大小写变体 + process_vision_info
  - ADR-019: 默认 video 模式 (use_video_format=True)
  - ADR-028: 最佳 VLM 配置 (r=16, 300步, α=0.5~0.8)

用法:
    # 标注真实轨迹 (D_real)
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python \\
        rlft/vlaw/scripts/label_trajectories.py \\
        --mode real \\
        --data_dir data/vlaw/rollouts/mixed/LiftPegUpright-v1 \\
        --output_dir data/vlaw/labeled/iter1 \\
        --lora_path checkpoints/vlaw/reward_model/ablation_v3/steps_300

    # 标注合成轨迹 (D_syn, 需 VAE 解码)
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python \\
        rlft/vlaw/scripts/label_trajectories.py \\
        --mode synthetic \\
        --data_dir data/vlaw/synthetic/iter1_v3 \\
        --output_dir data/vlaw/labeled/synthetic_iter1_v3 \\
        --lora_path checkpoints/vlaw/reward_model/ablation_v3/steps_300

    # Dry-run
    conda run -n vlaw_reward python rlft/vlaw/scripts/label_trajectories.py --dry_run
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

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))

TASK_INSTRUCTIONS: dict[str, str] = {
    "LiftPegUpright-v1": "Lift the peg and insert it upright into the holder.",
    "PickCube-v1": "Pick up the cube and place it on the target location.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
}


# ── 帧加载 ──────────────────────────────────────────────────────────────────

def load_frames_real(traj_grp: h5py.Group) -> np.ndarray:
    """从 HDF5 traj 组加载 RGB 帧 (D_real 格式)."""
    for key in ("rgb_base", "rgb_render"):
        if key in traj_grp:
            frames = traj_grp[key][:]
            if frames.dtype != np.uint8:
                frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8) if frames.max() <= 1.0 else frames.astype(np.uint8)
            return frames
    raise KeyError(f"traj 组中找不到 rgb_base/rgb_render: {list(traj_grp.keys())}")


@torch.inference_mode()
def decode_latent_frames(vae, latents: np.ndarray, frame_indices: np.ndarray,
                         device: str = "cuda", chunk_size: int = 4) -> np.ndarray:
    """解码 VAE latent 为 RGB (D_syn 格式)."""
    selected = torch.from_numpy(latents[frame_indices]).to(device).to(torch.float16)
    decoded_list = []
    for i in range(0, selected.shape[0], chunk_size):
        chunk = selected[i:i+chunk_size] / vae.config.scaling_factor
        out = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        decoded_list.append(out)
    decoded = torch.cat(decoded_list, dim=0)
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
    decoded = decoded.float().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    return decoded[:, :192, :, :]  # base camera (上半部分)


def sample_frames(frames_or_T: int | np.ndarray, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """均匀采样帧索引."""
    T = frames_or_T if isinstance(frames_or_T, int) else frames_or_T.shape[0]
    if T <= max_frames:
        return np.arange(T), None
    return np.linspace(0, T - 1, max_frames, dtype=int), None


# ── 标注核心 ────────────────────────────────────────────────────────────────

def label_real_hdf5(h5_path: Path, reward_model, task_id: str,
                    max_frames: int = 16) -> list[dict]:
    """标注 D_real HDF5 中所有轨迹."""
    results = []
    instruction = TASK_INSTRUCTIONS.get(task_id, f"Complete the {task_id} task.")
    with h5py.File(h5_path, "r") as f:
        meta_grp = f.get("meta", None)
        if meta_grp and "instruction" in meta_grp:
            instruction = str(meta_grp["instruction"][()])
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for traj_key in traj_keys:
            grp = f[traj_key]
            try:
                frames = load_frames_real(grp)
                T = len(frames)
                idxs, _ = sample_frames(T, max_frames)
                sampled = frames[idxs]
                score = reward_model.score_trajectory(sampled, instruction)
                env_succ = grp["env_success"][:].astype(bool) if "env_success" in grp else np.zeros(T, dtype=bool)
                results.append({
                    "traj_key": traj_key, "task_id": task_id, "T": T,
                    "vlm_reward": float(score.get("reward", 0.0)),
                    "vlm_yes_prob": float(score.get("p_yes", 0.0)),
                    "vlm_success": bool(score.get("reward", 0) > 0),
                    "env_success_once": float(np.any(env_succ)),
                    "env_success_at_end": float(bool(env_succ[-1])) if len(env_succ) > 0 else 0.0,
                })
            except Exception as e:
                results.append({"traj_key": traj_key, "task_id": task_id, "T": 0,
                                "vlm_reward": 0.0, "vlm_yes_prob": 0.0,
                                "vlm_success": False, "error": str(e)})
    return results


def label_synthetic_hdf5(h5_path: Path, vae, reward_model, task_id: str,
                         max_frames: int = 16, device: str = "cuda") -> list[dict]:
    """标注 D_syn HDF5 中所有轨迹 (需 VAE 解码)."""
    results = []
    instruction = TASK_INSTRUCTIONS.get(task_id, f"Complete the {task_id} task.")
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for idx, traj_key in enumerate(traj_keys):
            grp = f[traj_key]
            latents = grp["latent"][:]  # (T, 4, 48, 24)
            T = latents.shape[0]
            n = min(max_frames, T)
            frame_idx = np.linspace(0, T - 1, n, dtype=int)
            try:
                rgb_frames = decode_latent_frames(vae, latents, frame_idx, device=device)
                score = reward_model.score_trajectory(rgb_frames, instruction)
                results.append({
                    "traj_key": traj_key, "task_id": task_id, "T": T,
                    "vlm_reward": float(score["reward"]),
                    "vlm_yes_prob": float(score["p_yes"]),
                    "vlm_success": bool(score["reward"] > 0),
                    "threshold": float(score["threshold"]),
                    "source": "synthetic",
                })
            except Exception as e:
                results.append({"traj_key": traj_key, "task_id": task_id, "T": T,
                                "vlm_reward": 0.0, "vlm_yes_prob": 0.0,
                                "vlm_success": False, "error": str(e)})
            if (idx + 1) % 20 == 0:
                n_ok = sum(1 for r in results if r.get("vlm_success"))
                print(f"  [{idx+1}/{len(traj_keys)}] success={n_ok}/{len(results)}")
    return results


# ── 结果保存 ────────────────────────────────────────────────────────────────

def save_results(results: list[dict], output_dir: Path, task_id: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    with open(output_dir / f"{task_id}_vlm_rewards.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # HDF5
    with h5py.File(output_dir / f"{task_id}_vlm_rewards.h5", "w") as f:
        for item in results:
            grp = f.require_group(item["traj_key"])
            for k in ("vlm_reward", "vlm_yes_prob", "vlm_success", "env_success_once", "env_success_at_end"):
                if k in item:
                    grp.attrs[k] = float(item[k]) if k != "vlm_success" else int(item[k])
    # 统计
    valid = [r for r in results if "error" not in r]
    n_success = sum(1 for r in valid if r.get("vlm_success"))
    pyes = [r["vlm_yes_prob"] for r in valid]
    summary = {
        "n_total": len(results), "n_valid": len(valid), "n_errors": len(results) - len(valid),
        "n_vlm_success": n_success,
        "vlm_success_rate": n_success / max(len(valid), 1),
        "p_yes_mean": float(np.mean(pyes)) if pyes else 0.0,
        "p_yes_max": float(np.max(pyes)) if pyes else 0.0,
    }
    with open(output_dir / f"{task_id}_label_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [LABEL] {task_id}: {len(valid)} valid, {n_success} success ({summary['vlm_success_rate']:.1%}), "
          f"p_yes mean={summary['p_yes_mean']:.4f} max={summary['p_yes_max']:.4f}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VLAW VLM 轨迹标注 (稳定版)")
    parser.add_argument("--mode", choices=["real", "synthetic"], default="real",
                        help="标注模式: real (HDF5 RGB) 或 synthetic (HDF5 latent + VAE)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录 (含 .h5 文件)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=str, default="LiftPegUpright-v1")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/vlaw/reward_model/qwen_vl")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter 路径 (推荐 ablation_v3/steps_300)")
    parser.add_argument("--vae_path", type=str,
                        default="checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae",
                        help="VAE 路径 (仅 synthetic 模式)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="成功判定阈值 α (ADR-028: 0.5=平衡, 0.8=保守)")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="生成成功轨迹帧 strip + p_yes 分布图")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = WORKSPACE / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = WORKSPACE / output_dir

    print(f"[LABEL] 模式={args.mode} task={args.task_id} threshold={args.threshold}")
    print(f"  data_dir  : {data_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  lora_path : {args.lora_path or '(none, base model)'}")

    if args.dry_run:
        h5_files = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("**/*.h5"))
        print(f"[DRY RUN] 找到 {len(h5_files)} 个 HDF5 文件")
        for hf in h5_files[:3]:
            with h5py.File(hf, "r") as f:
                n = len([k for k in f.keys() if k.startswith("traj_")])
                print(f"  {hf.name}: {n} trajs")
        print("[DRY RUN] ✅ 完成")
        return

    # 加载 VLM
    from rlft.vlaw.reward.reward_model import VLAWRewardConfig, VLAWRewardModel
    model_path = str(WORKSPACE / args.model_path) if not Path(args.model_path).is_absolute() else args.model_path
    vlm_cfg = VLAWRewardConfig(model_path=model_path, threshold=args.threshold,
                               device=args.device, num_frames=args.max_frames)
    vlm = VLAWRewardModel(vlm_cfg)
    lora_abs = str(WORKSPACE / args.lora_path) if args.lora_path and not Path(args.lora_path).is_absolute() else args.lora_path
    vlm.load_model(lora_path=lora_abs)

    # 加载 VAE (仅 synthetic)
    vae = None
    if args.mode == "synthetic":
        from diffusers.models import AutoencoderKLTemporalDecoder
        vae_path = str(WORKSPACE / args.vae_path) if not Path(args.vae_path).is_absolute() else args.vae_path
        vae = AutoencoderKLTemporalDecoder.from_pretrained(vae_path, torch_dtype=torch.float16).to(args.device).eval()

    # 标注
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        h5_files = sorted(data_dir.glob("**/*.h5"))
    all_results: list[dict] = []
    t0 = time.time()

    for h5f in h5_files:
        print(f"\n[LABEL] 处理: {h5f.name}")
        if args.mode == "real":
            results = label_real_hdf5(h5f, vlm, args.task_id, args.max_frames)
        else:
            results = label_synthetic_hdf5(h5f, vae, vlm, args.task_id, args.max_frames, args.device)
        all_results.extend(results)

    summary = save_results(all_results, output_dir / args.task_id, args.task_id)
    # 总汇
    total_summary = {**summary, "mode": args.mode, "threshold": args.threshold,
                     "lora_path": args.lora_path, "elapsed_s": round(time.time() - t0, 1)}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(total_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[LABEL] ✅ 标注完成: {len(all_results)} 条, {summary['n_vlm_success']} success ({time.time()-t0:.0f}s)")

    if args.visualize:
        _visualize_labels(all_results, output_dir, args.task_id, data_dir, args.mode)

    vlm.unload_model()
    if vae:
        del vae
    torch.cuda.empty_cache()


# ── 可视化 ──────────────────────────────────────────────────────────────────

def _visualize_labels(results: list[dict], output_dir: Path, task_id: str,
                      data_dir: Path, mode: str) -> None:
    """生成成功轨迹帧 strip + p_yes 分布柱状图.

    保存到 {output_dir}/viz/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("[LABEL] ⚠️ 无有效结果可视化")
        return

    # ── 1. p_yes 分布直方图 ──
    p_yes_vals = [r["vlm_yes_prob"] for r in valid]
    success_vals = [r["vlm_yes_prob"] for r in valid if r.get("vlm_success")]
    fail_vals = [r["vlm_yes_prob"] for r in valid if not r.get("vlm_success")]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    if fail_vals:
        ax.hist(fail_vals, bins=bins, alpha=0.6, label="VLM fail", color="salmon")
    if success_vals:
        ax.hist(success_vals, bins=bins, alpha=0.6, label="VLM success", color="steelblue")
    ax.set_xlabel("p(yes)")
    ax.set_ylabel("Count")
    ax.set_title(f"{task_id} — VLM p_yes Distribution ({len(valid)} trajs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = viz_dir / "p_yes_distribution.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LABEL] 📊 {fig_path.name}")

    # ── 2. 成功轨迹采样帧 strip (仅 real 模式, 最多 5 条) ──
    if mode != "real":
        return
    success_results = [r for r in valid if r.get("vlm_success")][:5]
    if not success_results:
        return

    h5_files = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("**/*.h5"))
    h5_map: dict[str, str] = {}
    for hf in h5_files:
        try:
            with h5py.File(hf, "r") as f:
                for k in f.keys():
                    if k.startswith("traj_"):
                        h5_map[k] = str(hf)
        except Exception:
            pass

    for r in success_results:
        traj_key = r["traj_key"]
        h5_path = h5_map.get(traj_key)
        if not h5_path:
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                frames = load_frames_real(f[traj_key])
            T = frames.shape[0]
            idxs = np.linspace(0, T - 1, min(6, T), dtype=int)
            strip = np.concatenate([frames[j] for j in idxs], axis=1)
            img = PILImage.fromarray(strip)
            save_path = viz_dir / f"success_{traj_key}.png"
            img.save(str(save_path))
            print(f"[LABEL] 🖼️ {save_path.name} (p_yes={r['vlm_yes_prob']:.3f})")
        except Exception as e:
            print(f"[LABEL] ⚠️ {traj_key}: {e}")


if __name__ == "__main__":
    main()
