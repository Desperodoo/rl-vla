#!/usr/bin/env python3
"""T-TIMING-QUICKTEST Phase 2 — VLM 标注 num_interact=4 合成轨迹

对 50 条合成轨迹 (num_interact=4, 20帧/条) 做 VLM 标注：
  1. VAE 解码 latent → RGB
  2. 均匀采样 16 帧
  3. Fine-tuned VLM 评分 → p_yes
  4. 统计 D_syn+ @ α=0.4 和 α=0.8

用法:
    CUDA_VISIBLE_DEVICES=6 conda run -n vlaw_reward python scripts/vlaw/eval/label_timing_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── 配置 ──────────────────────────────────────────────────────────────────────
H5_PATH = ROOT / "data/vlaw/synthetic/iter1_timing_test/synthetic_iter1_final_1772601046.h5"
OUTPUT_DIR = ROOT / "data/vlaw/labeled/synthetic_iter1_timing_test"
MODEL_PATH = str(ROOT / "checkpoints/vlaw/reward_model/qwen_vl")
LORA_PATH = str(ROOT / "checkpoints/vlaw/reward_model/lora_iter1_16frame")
VAE_PATH = str(ROOT / "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae")
NUM_FRAMES = 16
THRESHOLDS = [0.4, 0.8]
DEVICE = "cuda:0"
TASK = "LiftPegUpright-v1"
INSTRUCTION = "Lift the peg and insert it upright into the holder."


# ── VAE 工具 ──────────────────────────────────────────────────────────────────

def load_vae(vae_path: str, device: str = "cuda") -> torch.nn.Module:
    from diffusers.models import AutoencoderKLTemporalDecoder
    print(f"[VAE] 加载: {vae_path}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path, torch_dtype=torch.float16,
    ).to(device)
    vae.eval()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[VAE] 就绪 | VRAM={mem_gb:.1f} GB")
    return vae


@torch.inference_mode()
def decode_latent_frames(
    vae: torch.nn.Module,
    latents: np.ndarray,
    frame_indices: np.ndarray,
    device: str = "cuda",
    decode_chunk_size: int = 4,
) -> np.ndarray:
    """解码指定帧的 latent → RGB (192×192 base camera)."""
    selected = torch.from_numpy(latents[frame_indices]).to(device).to(torch.float16)
    scaling_factor = vae.config.scaling_factor
    decoded_list = []
    for i in range(0, selected.shape[0], decode_chunk_size):
        chunk = selected[i : i + decode_chunk_size] / scaling_factor
        out = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        decoded_list.append(out)
    decoded = torch.cat(decoded_list, dim=0)  # (N, 3, 384, 192)
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1) * 255
    decoded = decoded.float().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    return decoded[:, :192, :, :]  # base camera: 上半部 (N, 192, 192, 3)


# ── 标注主循环 ────────────────────────────────────────────────────────────────

def label_all(
    h5_path: Path,
    vae: torch.nn.Module,
    reward_model: object,
    instruction: str,
    num_frames: int,
    device: str,
) -> list[dict]:
    results: list[dict] = []
    t0 = time.time()

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        n_total = len(traj_keys)
        print(f"[LABEL] 轨迹数={n_total} | 帧采样={num_frames} | 指令: {instruction}")

        for idx, tkey in enumerate(traj_keys):
            latents = f[tkey]["latent"][:]  # (T, 4, 48, 24)
            T = latents.shape[0]
            n = min(num_frames, T)
            fidx = np.linspace(0, T - 1, n, dtype=int)

            try:
                rgb = decode_latent_frames(vae, latents, fidx, device=device)
                score = reward_model.score_trajectory(rgb, instruction)
                entry = {
                    "traj_key": tkey,
                    "T": int(T),
                    "n_frames": int(n),
                    "p_yes": float(score["p_yes"]),
                    "instruction": instruction,
                }
                results.append(entry)

                if (idx + 1) % 10 == 0 or idx == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [{idx+1}/{n_total}] {tkey}: p_yes={score['p_yes']:.4f} "
                        f"({elapsed:.0f}s)"
                    )
            except Exception as e:
                print(f"  [ERROR] {tkey}: {e}", file=sys.stderr)
                results.append({"traj_key": tkey, "T": int(T), "p_yes": 0.0, "error": str(e)})

    return results


# ── 统计 & 保存 ───────────────────────────────────────────────────────────────

def compute_and_save(results: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # save raw json
    json_path = output_dir / "vlm_scores.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # save h5
    h5_out = output_dir / "vlm_scores.h5"
    with h5py.File(h5_out, "w") as f:
        for r in results:
            grp = f.require_group(r["traj_key"])
            grp.attrs["p_yes"] = r["p_yes"]
            grp.attrs["error"] = r.get("error", "")

    valid = [r for r in results if "error" not in r]
    p_arr = np.array([r["p_yes"] for r in valid])
    n_valid = len(valid)

    summary: dict = {
        "task": TASK,
        "source": "iter1_timing_test (num_interact=4)",
        "n_total": len(results),
        "n_valid": n_valid,
        "n_errors": len(results) - n_valid,
        "num_frames_sampled": NUM_FRAMES,
        "lora": LORA_PATH,
        "p_yes_mean": float(p_arr.mean()) if n_valid else 0.0,
        "p_yes_std": float(p_arr.std()) if n_valid else 0.0,
        "p_yes_median": float(np.median(p_arr)) if n_valid else 0.0,
        "p_yes_max": float(p_arr.max()) if n_valid else 0.0,
        "p_yes_min": float(p_arr.min()) if n_valid else 0.0,
    }

    # 双阈值统计
    for alpha in THRESHOLDS:
        cnt = int((p_arr > alpha).sum()) if n_valid else 0
        rate = cnt / max(n_valid, 1)
        summary[f"dsyn_plus_alpha_{alpha}"] = cnt
        summary[f"dsyn_plus_rate_alpha_{alpha}"] = rate

    # 对照组 002b 数据
    summary["baseline_002b"] = {
        "n_total": 200,
        "dsyn_plus_alpha_0.4": 7,
        "dsyn_plus_rate_alpha_0.4": 0.035,
        "dsyn_plus_alpha_0.8": 0,
        "dsyn_plus_rate_alpha_0.8": 0.0,
        "p_yes_max": 0.531,
        "note": "002b sliding window, num_interact=1, 100 frames",
    }

    stat_path = output_dir / "label_summary.json"
    with open(stat_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # pretty print
    print(f"\n{'='*65}")
    print(f"  T-TIMING-QUICKTEST Phase 2 — VLM 标注结果")
    print(f"{'='*65}")
    print(f"  数据源     : iter1_timing_test (num_interact=4, 50 traj, 20帧)")
    print(f"  有效/总计  : {n_valid}/{len(results)}")
    print(f"  p_yes 分布 : mean={summary['p_yes_mean']:.4f} ± {summary['p_yes_std']:.4f}")
    print(f"               median={summary['p_yes_median']:.4f}, "
          f"min={summary['p_yes_min']:.4f}, max={summary['p_yes_max']:.4f}")
    for alpha in THRESHOLDS:
        cnt = summary[f"dsyn_plus_alpha_{alpha}"]
        rate = summary[f"dsyn_plus_rate_alpha_{alpha}"]
        print(f"  D_syn+(α={alpha}) : {cnt}/{n_valid} ({rate:.1%})")
    print(f"  ────────── 对照组 002b (sliding, 200条) ──────────")
    print(f"  D_syn+(α=0.4) : 7/200 (3.5%) | p_yes_max=0.531")
    print(f"{'='*65}")

    # per-trajectory breakdown (top 10)
    if n_valid > 0:
        sorted_v = sorted(valid, key=lambda x: x["p_yes"], reverse=True)
        print(f"\n  Top-10 p_yes:")
        for i, r in enumerate(sorted_v[:10]):
            marker = ""
            if r["p_yes"] > 0.8:
                marker = " ★★"
            elif r["p_yes"] > 0.4:
                marker = " ★"
            print(f"    {i+1}. {r['traj_key']}: p_yes={r['p_yes']:.4f}{marker}")

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    print(f"{'='*65}")
    print(f"  T-TIMING-QUICKTEST Phase 2: VLM 标注")
    print(f"  H5     : {H5_PATH}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  VLM    : {MODEL_PATH}")
    print(f"  LoRA   : {LORA_PATH}")
    print(f"  VAE    : {VAE_PATH}")
    print(f"  Frames : {NUM_FRAMES}")
    print(f"  Thresh : {THRESHOLDS}")
    print(f"  Device : {DEVICE}")
    print(f"{'='*65}\n")

    # 1. VAE
    t0 = time.time()
    vae = load_vae(VAE_PATH, device=DEVICE)
    print(f"[TIME] VAE: {time.time()-t0:.1f}s\n")

    # 2. VLM + LoRA
    t1 = time.time()
    from rlft.vlaw.reward.reward_model import VLAWRewardModel, VLAWRewardConfig
    vlm_cfg = VLAWRewardConfig(
        model_path=MODEL_PATH,
        threshold=0.4,  # default; we'll compare both later from p_yes
        device=DEVICE,
        num_frames=NUM_FRAMES,
    )
    vlm = VLAWRewardModel(vlm_cfg)
    vlm.load_model(lora_path=LORA_PATH)
    print(f"[TIME] VLM: {time.time()-t1:.1f}s")
    if torch.cuda.is_available():
        print(f"[GPU] VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB\n")

    # 3. 标注
    t2 = time.time()
    results = label_all(H5_PATH, vae, vlm, INSTRUCTION, NUM_FRAMES, DEVICE)
    label_time = time.time() - t2
    print(f"\n[TIME] 标注: {label_time:.0f}s ({label_time/max(len(results),1):.1f}s/traj)")

    # 4. 统计 & 保存
    summary = compute_and_save(results, OUTPUT_DIR)

    # 5. 清理
    vlm.unload_model()
    del vae
    torch.cuda.empty_cache()

    total = time.time() - t_start
    print(f"\n[DONE] 总耗时: {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
