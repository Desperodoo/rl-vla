"""Iter-1 Imagination 合成数据全面评估脚本 (P7 — Eval Agent).

评估内容:
  1. 逐轮解码质量分析 (Per-round VAE decode → no-reference quality)
  2. Latent 统计分析 (Synthetic vs Real)
  3. Action 分析 (分布对比 / 平滑度 / 异常检测)
  4. State 轨迹分析 (成功 vs 失败 / 各维度走势)
  5. VLM 标注分解 (p_yes 分布 / 相关性)

使用方式:
  # 不含 VAE 解码 (快速, rlft_ms3 env)
  conda run -n rlft_ms3 python rlft/vlaw/scripts/eval_imagination.py

  # 含 VAE 解码 (需 ctrl_world env + GPU)
  CUDA_VISIBLE_DEVICES=4 conda run -n ctrl_world python rlft/vlaw/scripts/eval_imagination.py --visualize

  # 只运行部分分析
  conda run -n rlft_ms3 python rlft/vlaw/scripts/eval_imagination.py --sections latent,action,state,vlm
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]  # rl-vla root


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class EvalImaginationConfig:
    """Imagination 合成数据评估配置."""

    # ── 数据路径 ──
    synthetic_h5: str = "data/vlaw/synthetic/iter1/synthetic_iter1_merged.h5"
    real_h5: str = "data/vlaw/encoded/train/LiftPegUpright-v1/LiftPegUpright-v1_real_1772643507.h5"
    vlm_rewards_json: str = "data/vlaw/labeled/iter1_syn/LiftPegUpright-v1/LiftPegUpright-v1_vlm_rewards.json"

    # ── VAE 路径 ──
    vae_path: str = "checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/vae"

    # ── 输出 ──
    output_dir: str = "results/vlaw/imagination_eval"

    # ── 行为 ──
    visualize: bool = False  # 需要 VAE + GPU
    sections: str = "all"  # "all" or comma-sep: "latent,action,state,vlm,decode"
    device: str = "cuda:0"
    vis_traj_count: int = 5  # 可视化展示轨迹数
    decode_chunk: int = 4  # VAE decode batch size

    # ── 合成数据参数 ──
    frames_per_round: int = 5
    total_rounds: int = 12
    total_frames: int = 60

    # ── Real data 采样上限 (避免 OOM) ──
    real_sample_max: int = 200

    def __post_init__(self) -> None:
        for attr in ("synthetic_h5", "real_h5", "vlm_rewards_json", "vae_path", "output_dir"):
            val = getattr(self, attr)
            if not Path(val).is_absolute():
                setattr(self, attr, str(WORKSPACE / val))


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _parse_sections(s: str) -> set[str]:
    if s == "all":
        return {"decode", "latent", "action", "state", "vlm"}
    return set(s.split(","))


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL-IMG] 📊 Saved {path.name}")


def _load_synthetic(h5_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load all synthetic trajectories as dict[traj_key -> {latent, actions, state}]."""
    data: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        for tk in traj_keys:
            grp = f[tk]
            entry: dict[str, np.ndarray] = {}
            for dk in ("latent", "actions", "state"):
                if dk in grp:
                    entry[dk] = grp[dk][:]
            data[tk] = entry
    return data


def _load_real_latents(h5_path: str, max_trajs: int = 200) -> np.ndarray:
    """Load real latent_concat from encoded H5 — returns (N*T, 4, 48, 24)."""
    all_lat: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))[:max_trajs]
        for tk in traj_keys:
            if "latent_concat" in f[tk]:
                all_lat.append(f[tk]["latent_concat"][:].astype(np.float32))
    return np.concatenate(all_lat, axis=0) if all_lat else np.empty((0, 4, 48, 24))


def _load_real_actions(h5_path: str, max_trajs: int = 200) -> np.ndarray:
    """Load real actions from encoded H5 — returns (N*T, action_dim)."""
    all_act: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))[:max_trajs]
        for tk in traj_keys:
            if "actions" in f[tk]:
                all_act.append(f[tk]["actions"][:])
    return np.concatenate(all_act, axis=0) if all_act else np.empty((0, 7))


def _load_vlm_rewards(json_path: str) -> list[dict[str, Any]]:
    with open(json_path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# Section 1: Per-round VAE Decode Quality
# ──────────────────────────────────────────────────────────────
def eval_decode_quality(
    syn_data: dict[str, dict[str, np.ndarray]],
    vlm_results: list[dict[str, Any]],
    cfg: EvalImaginationConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """VAE 解码每一轮关键帧, 计算 no-reference 图像质量指标."""
    import torch
    from PIL import Image as PILImage

    print("[EVAL-IMG] === Section 1: Per-round VAE Decode Quality ===")

    # Load VAE
    from diffusers.models import AutoencoderKLTemporalDecoder

    vae_path = cfg.vae_path
    if not Path(vae_path).exists():
        # fallback
        vae_path = str(Path(vae_path).parent)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path, torch_dtype=torch.float16
    ).to(cfg.device).eval()

    decode_dir = out_dir / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-round quality across all trajs ──
    rounds = cfg.total_rounds
    fpr = cfg.frames_per_round
    per_round_brightness: list[list[float]] = [[] for _ in range(rounds)]
    per_round_color_var: list[list[float]] = [[] for _ in range(rounds)]
    per_round_sharpness: list[list[float]] = [[] for _ in range(rounds)]

    traj_keys = sorted(syn_data.keys())

    # Build vlm lookup
    vlm_lookup: dict[str, dict[str, Any]] = {}
    for r in vlm_results:
        vlm_lookup[r["traj_key"]] = r

    # ── Select representative trajs for strip viz ──
    success_keys = [k for k in traj_keys if vlm_lookup.get(k, {}).get("vlm_success", False)]
    fail_keys = [k for k in traj_keys if not vlm_lookup.get(k, {}).get("vlm_success", True)]
    # Pick 3 success + 2 fail (or whatever available)
    n_suc = min(3, len(success_keys))
    n_fail = min(2, len(fail_keys))
    vis_keys = success_keys[:n_suc] + fail_keys[:n_fail]
    vis_keys = vis_keys[: cfg.vis_traj_count]

    print(f"[EVAL-IMG] Processing {len(traj_keys)} trajs, {rounds} rounds each")

    for ti, tk in enumerate(traj_keys):
        latent = syn_data[tk]["latent"].astype(np.float32)  # (60, 4, 48, 24)
        T = latent.shape[0]
        actual_rounds = T // fpr

        for r_idx in range(min(actual_rounds, rounds)):
            # Take first frame of each round
            frame_idx = r_idx * fpr
            frame_lat = latent[frame_idx: frame_idx + 1]  # (1, 4, 48, 24)

            with torch.no_grad():
                lat_t = torch.from_numpy(frame_lat).to(cfg.device, torch.float16)
                lat_t = lat_t / vae.config.scaling_factor
                decoded = vae.decode(lat_t, num_frames=1).sample
                rgb = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).float()
                rgb_np = rgb.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
                # Take base camera (upper half: :192 rows)
                rgb_np = rgb_np[:, :192, :, :]  # (1, 192, W, 3)

            img = rgb_np[0]  # (192, W, 3)

            # No-reference metrics
            gray = np.mean(img, axis=2)
            brightness = float(np.mean(gray))
            color_var = float(np.var(img.astype(np.float32), axis=(0, 1)).mean())
            # Sharpness via Laplacian variance
            from scipy.ndimage import laplace
            sharpness = float(np.var(laplace(gray)))

            per_round_brightness[r_idx].append(brightness)
            per_round_color_var[r_idx].append(color_var)
            per_round_sharpness[r_idx].append(sharpness)

        # ── Strip viz for representative trajs ──
        if tk in vis_keys:
            strip_frames = []
            for r_idx in range(min(actual_rounds, rounds)):
                frame_idx = r_idx * fpr
                frame_lat = latent[frame_idx: frame_idx + 1]
                with torch.no_grad():
                    lat_t = torch.from_numpy(frame_lat).to(cfg.device, torch.float16)
                    lat_t = lat_t / vae.config.scaling_factor
                    decoded = vae.decode(lat_t, num_frames=1).sample
                    rgb = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).float()
                    rgb_np = rgb.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
                    rgb_np = rgb_np[:, :192, :, :]
                strip_frames.append(rgb_np[0])

            strip = np.concatenate(strip_frames, axis=1)
            label = "suc" if vlm_lookup.get(tk, {}).get("vlm_success", False) else "fail"
            p_yes = vlm_lookup.get(tk, {}).get("vlm_yes_prob", -1)
            save_path = decode_dir / f"{tk}_{label}_p{p_yes:.2f}_strip.png"
            PILImage.fromarray(strip).save(str(save_path))
            print(f"[EVAL-IMG]   Strip: {save_path.name}")

        if (ti + 1) % 50 == 0:
            print(f"[EVAL-IMG]   Processed {ti + 1}/{len(traj_keys)} trajs")

    # ── Aggregate per-round stats ──
    round_stats: dict[str, list[dict[str, float]]] = {
        "brightness": [],
        "color_var": [],
        "sharpness": [],
    }
    for r_idx in range(rounds):
        for metric, data_list in zip(
            ["brightness", "color_var", "sharpness"],
            [per_round_brightness, per_round_color_var, per_round_sharpness],
        ):
            vals = data_list[r_idx]
            if vals:
                round_stats[metric].append({
                    "round": r_idx,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                })

    # ── Plot per-round quality decay ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, stats) in zip(axes, round_stats.items()):
        rounds_x = [s["round"] for s in stats]
        means = [s["mean"] for s in stats]
        stds = [s["std"] for s in stats]
        ax.errorbar(rounds_x, means, yerr=stds, marker="o", capsize=3, linewidth=1.5)
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.set_title(f"Per-round {metric}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-round VAE Decode Quality Decay (No-reference)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, decode_dir / "per_round_quality_decay.png")

    # Cleanup
    del vae
    torch.cuda.empty_cache()

    result = {"round_stats": round_stats, "vis_trajs": vis_keys}
    with open(decode_dir / "decode_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[EVAL-IMG] Section 1 done. Stats saved to {decode_dir / 'decode_stats.json'}")
    return result


# ──────────────────────────────────────────────────────────────
# Section 2: Latent Statistics (Synthetic vs Real)
# ──────────────────────────────────────────────────────────────
def eval_latent_stats(
    syn_data: dict[str, dict[str, np.ndarray]],
    cfg: EvalImaginationConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """Compare latent distributions: synthetic vs real."""
    print("[EVAL-IMG] === Section 2: Latent Statistics (Synthetic vs Real) ===")

    lat_dir = out_dir / "latent"
    lat_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real latents ──
    real_path = cfg.real_h5
    print(f"[EVAL-IMG] Loading real latents from {Path(real_path).name}...")
    real_latents = _load_real_latents(real_path, cfg.real_sample_max)
    print(f"[EVAL-IMG]   Real latents shape: {real_latents.shape}")

    # ── Collect synthetic latents ──
    syn_all_lat: list[np.ndarray] = []
    syn_per_round: dict[int, list[np.ndarray]] = {r: [] for r in range(cfg.total_rounds)}
    fpr = cfg.frames_per_round

    for tk, entry in syn_data.items():
        lat = entry["latent"].astype(np.float32)  # (60, 4, 48, 24)
        syn_all_lat.append(lat)
        T = lat.shape[0]
        for r_idx in range(min(T // fpr, cfg.total_rounds)):
            chunk = lat[r_idx * fpr: (r_idx + 1) * fpr]
            syn_per_round[r_idx].append(chunk)

    syn_all = np.concatenate(syn_all_lat, axis=0)  # (N*60, 4, 48, 24)
    print(f"[EVAL-IMG]   Synthetic latents shape: {syn_all.shape}")

    # ── Per-channel mean/std ──
    # Flatten spatial dims
    def channel_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (4,) mean, (4,) std over all frames/spatial."""
        # arr: (N, 4, H, W) -> per channel
        n = arr.shape[0]
        reshaped = arr.reshape(n, 4, -1)  # (N, 4, H*W)
        per_frame_mean = reshaped.mean(axis=2)  # (N, 4)
        per_frame_std = reshaped.std(axis=2)  # (N, 4)
        return per_frame_mean.mean(axis=0), per_frame_std.mean(axis=0)

    syn_ch_mean, syn_ch_std = channel_stats(syn_all)
    real_ch_mean, real_ch_std = channel_stats(real_latents)

    ch_stats = {
        "synthetic": {"mean": syn_ch_mean.tolist(), "std": syn_ch_std.tolist()},
        "real": {"mean": real_ch_mean.tolist(), "std": real_ch_std.tolist()},
        "delta_mean": (syn_ch_mean - real_ch_mean).tolist(),
        "delta_std": (syn_ch_std - real_ch_std).tolist(),
    }
    print(f"[EVAL-IMG]   Per-channel mean - syn: {syn_ch_mean}, real: {real_ch_mean}")
    print(f"[EVAL-IMG]   Per-channel std  - syn: {syn_ch_std}, real: {real_ch_std}")

    # ── Per-channel distribution plots ──
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for c in range(4):
        syn_flat = syn_all[:, c, :, :].flatten()
        real_flat = real_latents[:, c, :, :].flatten()

        # Subsample for histogram (avoid memory blow-up)
        rng = np.random.default_rng(42)
        n_sample = min(500_000, len(syn_flat), len(real_flat))
        syn_sample = rng.choice(syn_flat, n_sample, replace=False)
        real_sample = rng.choice(real_flat, n_sample, replace=False)

        axes[0, c].hist(real_sample, bins=100, alpha=0.6, label="Real", density=True, color="tab:blue")
        axes[0, c].hist(syn_sample, bins=100, alpha=0.6, label="Synthetic", density=True, color="tab:orange")
        axes[0, c].set_title(f"Channel {c} value dist")
        axes[0, c].legend(fontsize=8)
        axes[0, c].set_xlim(-5, 5)

        # Per-frame mean across spatial
        syn_frame_mean = syn_all[:, c, :, :].reshape(syn_all.shape[0], -1).mean(axis=1)
        real_frame_mean = real_latents[:, c, :, :].reshape(real_latents.shape[0], -1).mean(axis=1)
        axes[1, c].hist(real_frame_mean, bins=60, alpha=0.6, label="Real", density=True, color="tab:blue")
        axes[1, c].hist(syn_frame_mean, bins=60, alpha=0.6, label="Synthetic", density=True, color="tab:orange")
        axes[1, c].set_title(f"Ch {c} frame-mean dist")
        axes[1, c].legend(fontsize=8)

    fig.suptitle("Latent Distribution: Synthetic vs Real (per channel)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, lat_dir / "latent_channel_distribution.png")

    # ── L2 norm per round (synthetic) vs real baseline ──
    # Real L2 norm baseline
    real_l2 = np.linalg.norm(real_latents.reshape(real_latents.shape[0], -1), axis=1)
    real_l2_mean = float(np.mean(real_l2))
    real_l2_std = float(np.std(real_l2))

    round_l2_stats: list[dict[str, float]] = []
    for r_idx in range(cfg.total_rounds):
        chunks = syn_per_round[r_idx]
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        l2 = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
        round_l2_stats.append({
            "round": r_idx,
            "l2_mean": float(np.mean(l2)),
            "l2_std": float(np.std(l2)),
        })

    fig, ax = plt.subplots(figsize=(8, 4))
    r_x = [s["round"] for s in round_l2_stats]
    l2_means = [s["l2_mean"] for s in round_l2_stats]
    l2_stds = [s["l2_std"] for s in round_l2_stats]
    ax.errorbar(r_x, l2_means, yerr=l2_stds, marker="o", capsize=3, label="Synthetic (per round)", linewidth=1.5)
    ax.axhline(real_l2_mean, color="tab:blue", linestyle="--", label=f"Real mean ({real_l2_mean:.1f})")
    ax.fill_between(
        [0, cfg.total_rounds - 1],
        real_l2_mean - real_l2_std, real_l2_mean + real_l2_std,
        alpha=0.15, color="tab:blue", label="Real ±1σ",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Latent L2 Norm: Synthetic per-round vs Real")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, lat_dir / "latent_l2_drift.png")

    # ── Drift: per-round channel mean shift ──
    round_ch_mean_shift: list[dict[str, Any]] = []
    for r_idx in range(cfg.total_rounds):
        chunks = syn_per_round[r_idx]
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        syn_rm, _ = channel_stats(arr)
        shift = (syn_rm - real_ch_mean).tolist()
        round_ch_mean_shift.append({"round": r_idx, "shift": shift})

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in range(4):
        shifts = [s["shift"][c] for s in round_ch_mean_shift]
        rounds_x = [s["round"] for s in round_ch_mean_shift]
        ax.plot(rounds_x, shifts, marker="o", label=f"Ch {c}", linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean shift (syn - real)")
    ax.set_title("Latent Channel Mean Drift per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, lat_dir / "latent_channel_drift.png")

    result = {
        "channel_stats": ch_stats,
        "round_l2": round_l2_stats,
        "real_l2_mean": real_l2_mean,
        "real_l2_std": real_l2_std,
        "round_channel_drift": round_ch_mean_shift,
    }
    with open(lat_dir / "latent_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[EVAL-IMG] Section 2 done. Stats saved to {lat_dir / 'latent_stats.json'}")
    return result


# ──────────────────────────────────────────────────────────────
# Section 3: Action Analysis
# ──────────────────────────────────────────────────────────────
def eval_actions(
    syn_data: dict[str, dict[str, np.ndarray]],
    cfg: EvalImaginationConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """Action distribution, smoothness, anomaly detection."""
    print("[EVAL-IMG] === Section 3: Action Analysis ===")

    act_dir = out_dir / "action"
    act_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all synthetic actions ──
    syn_actions_list: list[np.ndarray] = []
    per_traj_smoothness: dict[str, float] = {}
    anomalies: dict[str, list[str]] = {}

    for tk, entry in syn_data.items():
        act = entry["actions"]  # (T, 7)
        syn_actions_list.append(act)

        # Smoothness: L2 norm of consecutive action differences
        diffs = np.linalg.norm(np.diff(act, axis=0), axis=1)  # (T-1,)
        per_traj_smoothness[tk] = float(np.mean(diffs))

        # Anomaly detection
        issues: list[str] = []
        if np.any(np.isnan(act)):
            issues.append("NaN")
        if np.any(np.isinf(act)):
            issues.append("Inf")
        if np.any(np.abs(act) > 10.0):
            issues.append(f"OutOfRange(max={np.abs(act).max():.2f})")
        # Check for constant actions (all same)
        if np.std(act, axis=0).max() < 1e-6:
            issues.append("Constant")
        # Check for frozen segments (>10 consecutive identical frames)
        for d in range(act.shape[1]):
            changes = np.diff(act[:, d])
            frozen = np.sum(np.abs(changes) < 1e-8)
            if frozen > act.shape[0] * 0.8:
                issues.append(f"Frozen_dim{d}")
                break
        if issues:
            anomalies[tk] = issues

    syn_actions_all = np.concatenate(syn_actions_list, axis=0)  # (N*T, 7)
    print(f"[EVAL-IMG]   Synthetic actions shape: {syn_actions_all.shape}")

    # ── Load real actions ──
    real_actions = _load_real_actions(cfg.real_h5, cfg.real_sample_max)
    print(f"[EVAL-IMG]   Real actions shape: {real_actions.shape}")

    # ── Action distribution histograms ──
    dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes_flat = axes.flatten()
    for d in range(min(7, syn_actions_all.shape[1])):
        ax = axes_flat[d]
        ax.hist(real_actions[:, d], bins=80, alpha=0.6, label="Real", density=True, color="tab:blue")
        ax.hist(syn_actions_all[:, d], bins=80, alpha=0.6, label="Synthetic", density=True, color="tab:orange")
        ax.set_title(f"Dim {d} ({dim_names[d]})")
        ax.legend(fontsize=7)
    # Hide last subplot if 7 dims
    if syn_actions_all.shape[1] < 8:
        axes_flat[7].axis("off")
    fig.suptitle("Action Distribution: Synthetic vs Real", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, act_dir / "action_distribution.png")

    # ── Per-dim stats comparison ──
    dim_stats = {}
    for d in range(syn_actions_all.shape[1]):
        dim_stats[dim_names[d]] = {
            "syn_mean": float(np.mean(syn_actions_all[:, d])),
            "syn_std": float(np.std(syn_actions_all[:, d])),
            "real_mean": float(np.mean(real_actions[:, d])),
            "real_std": float(np.std(real_actions[:, d])),
        }

    # ── Smoothness plot ──
    smoothness_vals = list(per_traj_smoothness.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(smoothness_vals, bins=30, color="tab:green", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Mean L2 action diff")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Action Smoothness Distribution")
    axes[0].axvline(np.mean(smoothness_vals), color="red", linestyle="--",
                    label=f"Mean={np.mean(smoothness_vals):.4f}")
    axes[0].legend()

    # Smoothness across time (average across all trajs)
    all_diffs_per_step: list[list[float]] = [[] for _ in range(cfg.total_frames - 1)]
    for entry in syn_data.values():
        act = entry["actions"]
        diffs = np.linalg.norm(np.diff(act, axis=0), axis=1)
        for t, v in enumerate(diffs):
            if t < len(all_diffs_per_step):
                all_diffs_per_step[t].append(float(v))

    step_smooth_mean = [np.mean(s) if s else 0 for s in all_diffs_per_step]
    step_smooth_std = [np.std(s) if s else 0 for s in all_diffs_per_step]
    axes[1].fill_between(
        range(len(step_smooth_mean)),
        np.array(step_smooth_mean) - np.array(step_smooth_std),
        np.array(step_smooth_mean) + np.array(step_smooth_std),
        alpha=0.2, color="tab:green",
    )
    axes[1].plot(step_smooth_mean, color="tab:green", linewidth=1.2)
    # Mark round boundaries
    for r in range(1, cfg.total_rounds):
        axes[1].axvline(r * cfg.frames_per_round - 1, color="gray", linestyle=":", alpha=0.4)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Mean L2 action diff")
    axes[1].set_title("Action Smoothness over Time")

    fig.tight_layout()
    _save_fig(fig, act_dir / "action_smoothness.png")

    # ── Anomaly report ──
    n_anomalous = len(anomalies)
    print(f"[EVAL-IMG]   Anomalous trajs: {n_anomalous}/{len(syn_data)}")
    for tk, issues in list(anomalies.items())[:5]:
        print(f"[EVAL-IMG]     {tk}: {issues}")

    result = {
        "dim_stats": dim_stats,
        "smoothness_mean": float(np.mean(smoothness_vals)),
        "smoothness_std": float(np.std(smoothness_vals)),
        "n_anomalous": n_anomalous,
        "anomalies": anomalies,
        "per_traj_smoothness": per_traj_smoothness,
    }
    with open(act_dir / "action_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[EVAL-IMG] Section 3 done. Stats saved to {act_dir / 'action_stats.json'}")
    return result


# ──────────────────────────────────────────────────────────────
# Section 4: State Trajectory Analysis
# ──────────────────────────────────────────────────────────────
def eval_states(
    syn_data: dict[str, dict[str, np.ndarray]],
    vlm_results: list[dict[str, Any]],
    cfg: EvalImaginationConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """State trajectory analysis: success vs failure, key dims."""
    print("[EVAL-IMG] === Section 4: State Trajectory Analysis ===")

    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    vlm_lookup = {r["traj_key"]: r for r in vlm_results}
    traj_keys = sorted(syn_data.keys())

    success_states: list[np.ndarray] = []
    fail_states: list[np.ndarray] = []

    for tk in traj_keys:
        state = syn_data[tk].get("state")
        if state is None:
            continue
        if vlm_lookup.get(tk, {}).get("vlm_success", False):
            success_states.append(state)
        else:
            fail_states.append(state)

    print(f"[EVAL-IMG]   Success trajs: {len(success_states)}, Fail trajs: {len(fail_states)}")

    # ── Key state dimensions ──
    # State dim 25: typical ManiSkill state includes:
    #   0-2: end-effector xyz, 3-6: ee quat, 7: gripper, 8+: object states
    # We'll plot the first few dims and label them generically
    state_dim = success_states[0].shape[1] if success_states else 25
    dim_labels = {
        0: "ee_x", 1: "ee_y", 2: "ee_z",
        3: "ee_qx", 4: "ee_qy", 5: "ee_qz", 6: "ee_qw",
        7: "gripper",
    }
    # Show EE xyz + gripper + a few object dims
    plot_dims = [0, 1, 2, 7, 8, 9, 10]
    plot_dims = [d for d in plot_dims if d < state_dim]

    n_dims = len(plot_dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i, d in enumerate(plot_dims):
        ax = axes[i]
        # Plot success trajectories (light green)
        for st in success_states[:30]:  # limit to avoid clutter
            ax.plot(st[:, d], color="tab:green", alpha=0.1, linewidth=0.8)
        # Plot failure trajectories (light red)
        for st in fail_states[:30]:
            ax.plot(st[:, d], color="tab:red", alpha=0.1, linewidth=0.8)

        # Mean curves
        if success_states:
            min_T = min(s.shape[0] for s in success_states)
            suc_stack = np.stack([s[:min_T, d] for s in success_states])
            ax.plot(np.mean(suc_stack, axis=0), color="tab:green", linewidth=2,
                    label=f"Success mean (n={len(success_states)})")
        if fail_states:
            min_T = min(s.shape[0] for s in fail_states)
            fail_stack = np.stack([s[:min_T, d] for s in fail_states])
            ax.plot(np.mean(fail_stack, axis=0), color="tab:red", linewidth=2,
                    label=f"Fail mean (n={len(fail_states)})")

        label = dim_labels.get(d, f"dim_{d}")
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
        # Mark round boundaries
        for r in range(1, cfg.total_rounds):
            ax.axvline(r * cfg.frames_per_round, color="gray", linestyle=":", alpha=0.3)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("State Trajectories: Success (green) vs Fail (red)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, state_dir / "state_trajectories.png")

    # ── Terminal state comparison ──
    suc_terminal = np.array([s[-1] for s in success_states]) if success_states else np.empty((0, state_dim))
    fail_terminal = np.array([s[-1] for s in fail_states]) if fail_states else np.empty((0, state_dim))

    terminal_stats: dict[str, Any] = {}
    for d in plot_dims:
        label = dim_labels.get(d, f"dim_{d}")
        terminal_stats[label] = {}
        if suc_terminal.shape[0] > 0:
            terminal_stats[label]["suc_mean"] = float(np.mean(suc_terminal[:, d]))
            terminal_stats[label]["suc_std"] = float(np.std(suc_terminal[:, d]))
        if fail_terminal.shape[0] > 0:
            terminal_stats[label]["fail_mean"] = float(np.mean(fail_terminal[:, d]))
            terminal_stats[label]["fail_std"] = float(np.std(fail_terminal[:, d]))

    # ── State range / anomaly check ──
    all_states_cat = np.concatenate(
        [entry["state"] for entry in syn_data.values() if "state" in entry],
        axis=0,
    )
    state_range = {
        "min": all_states_cat.min(axis=0).tolist(),
        "max": all_states_cat.max(axis=0).tolist(),
        "mean": all_states_cat.mean(axis=0).tolist(),
        "std": all_states_cat.std(axis=0).tolist(),
    }
    has_nan = bool(np.any(np.isnan(all_states_cat)))
    has_inf = bool(np.any(np.isinf(all_states_cat)))
    print(f"[EVAL-IMG]   State range check: NaN={has_nan}, Inf={has_inf}")

    result = {
        "n_success": len(success_states),
        "n_fail": len(fail_states),
        "terminal_stats": terminal_stats,
        "state_range": state_range,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }
    with open(state_dir / "state_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[EVAL-IMG] Section 4 done. Stats saved to {state_dir / 'state_stats.json'}")
    return result


# ──────────────────────────────────────────────────────────────
# Section 5: VLM Label Analysis
# ──────────────────────────────────────────────────────────────
def eval_vlm_labels(
    syn_data: dict[str, dict[str, np.ndarray]],
    vlm_results: list[dict[str, Any]],
    action_stats: dict[str, Any] | None,
    cfg: EvalImaginationConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """VLM label breakdown: p_yes distribution, correlations with data features."""
    print("[EVAL-IMG] === Section 5: VLM Label Analysis ===")

    vlm_dir = out_dir / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)

    traj_keys = [r["traj_key"] for r in vlm_results]
    p_yes_arr = np.array([r["vlm_yes_prob"] for r in vlm_results])
    vlm_success = np.array([r["vlm_success"] for r in vlm_results])

    n_total = len(vlm_results)
    n_success = int(vlm_success.sum())
    print(f"[EVAL-IMG]   VLM results: {n_total} total, {n_success} success ({n_success / n_total * 100:.1f}%)")

    # ── p_yes histogram ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(p_yes_arr, bins=40, color="tab:purple", alpha=0.7, edgecolor="black")
    axes[0].axvline(0.5, color="red", linestyle="--", label="α=0.5")
    axes[0].axvline(0.8, color="orange", linestyle="--", label="α=0.8")
    axes[0].set_xlabel("p_yes")
    axes[0].set_ylabel("Count")
    axes[0].set_title("VLM p_yes Distribution")
    axes[0].legend()

    # Cumulative distribution
    p_sorted = np.sort(p_yes_arr)
    axes[1].plot(p_sorted, np.linspace(0, 1, len(p_sorted)), color="tab:purple", linewidth=2)
    axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="α=0.5")
    axes[1].axvline(0.8, color="orange", linestyle="--", alpha=0.5, label="α=0.8")
    axes[1].set_xlabel("p_yes threshold")
    axes[1].set_ylabel("Fraction below threshold")
    axes[1].set_title("p_yes CDF")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("VLM Label Distribution", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, vlm_dir / "p_yes_distribution.png")

    # ── Threshold sensitivity ──
    thresholds = np.arange(0.1, 1.0, 0.05)
    pass_rates = [float((p_yes_arr >= t).mean()) for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, pass_rates, marker="o", markersize=4, linewidth=1.5, color="tab:purple")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="α=0.5")
    ax.axvline(0.8, color="orange", linestyle="--", alpha=0.5, label="α=0.8")
    ax.set_xlabel("Threshold α")
    ax.set_ylabel("Pass rate (fraction with p_yes ≥ α)")
    ax.set_title("VLM Threshold Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, vlm_dir / "threshold_sensitivity.png")

    # ── Correlation: p_yes vs action smoothness ──
    correlations: dict[str, float] = {}
    if action_stats and "per_traj_smoothness" in action_stats:
        per_traj_smooth = action_stats["per_traj_smoothness"]
        p_yes_for_corr: list[float] = []
        smooth_for_corr: list[float] = []
        for r in vlm_results:
            tk = r["traj_key"]
            if tk in per_traj_smooth:
                p_yes_for_corr.append(r["vlm_yes_prob"])
                smooth_for_corr.append(per_traj_smooth[tk])

        if len(p_yes_for_corr) > 10:
            corr = float(np.corrcoef(p_yes_for_corr, smooth_for_corr)[0, 1])
            correlations["p_yes_vs_smoothness"] = corr
            print(f"[EVAL-IMG]   Corr(p_yes, smoothness) = {corr:.3f}")

            fig, ax = plt.subplots(figsize=(6, 5))
            colors = ["tab:green" if r["vlm_success"] else "tab:red" for r in vlm_results
                      if r["traj_key"] in per_traj_smooth]
            ax.scatter(smooth_for_corr, p_yes_for_corr, c=colors, alpha=0.5, s=20)
            ax.set_xlabel("Action Smoothness (mean L2 diff)")
            ax.set_ylabel("p_yes")
            ax.set_title(f"p_yes vs Action Smoothness (r={corr:.3f})")
            ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
            ax.grid(True, alpha=0.2)
            _save_fig(fig, vlm_dir / "corr_pyes_smoothness.png")

    # ── Correlation: p_yes vs latent L2 norm (last round) ──
    p_yes_lat: list[float] = []
    lat_l2_last: list[float] = []
    for r in vlm_results:
        tk = r["traj_key"]
        if tk in syn_data and "latent" in syn_data[tk]:
            lat = syn_data[tk]["latent"].astype(np.float32)
            # Last round latent L2
            last_round_start = max(0, lat.shape[0] - cfg.frames_per_round)
            last_lat = lat[last_round_start:]
            l2 = float(np.mean(np.linalg.norm(last_lat.reshape(last_lat.shape[0], -1), axis=1)))
            p_yes_lat.append(r["vlm_yes_prob"])
            lat_l2_last.append(l2)

    if len(p_yes_lat) > 10:
        corr_lat = float(np.corrcoef(p_yes_lat, lat_l2_last)[0, 1])
        correlations["p_yes_vs_last_round_l2"] = corr_lat
        print(f"[EVAL-IMG]   Corr(p_yes, last_round_L2) = {corr_lat:.3f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["tab:green" if v >= 0.5 else "tab:red" for v in p_yes_lat]
        ax.scatter(lat_l2_last, p_yes_lat, c=colors, alpha=0.5, s=20)
        ax.set_xlabel("Last Round Latent L2 Norm")
        ax.set_ylabel("p_yes")
        ax.set_title(f"p_yes vs Last Round Latent L2 (r={corr_lat:.3f})")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.2)
        _save_fig(fig, vlm_dir / "corr_pyes_latent_l2.png")

    # ── Correlation: p_yes vs terminal state Z ──
    p_yes_z: list[float] = []
    terminal_z: list[float] = []
    for r in vlm_results:
        tk = r["traj_key"]
        if tk in syn_data and "state" in syn_data[tk]:
            state = syn_data[tk]["state"]
            # dim 2 = ee_z (end-effector height)
            if state.shape[1] > 2:
                p_yes_z.append(r["vlm_yes_prob"])
                terminal_z.append(float(state[-1, 2]))

    if len(p_yes_z) > 10:
        corr_z = float(np.corrcoef(p_yes_z, terminal_z)[0, 1])
        correlations["p_yes_vs_terminal_ee_z"] = corr_z
        print(f"[EVAL-IMG]   Corr(p_yes, terminal_ee_z) = {corr_z:.3f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["tab:green" if v >= 0.5 else "tab:red" for v in p_yes_z]
        ax.scatter(terminal_z, p_yes_z, c=colors, alpha=0.5, s=20)
        ax.set_xlabel("Terminal EE Z (height)")
        ax.set_ylabel("p_yes")
        ax.set_title(f"p_yes vs Terminal EE Height (r={corr_z:.3f})")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.2)
        _save_fig(fig, vlm_dir / "corr_pyes_terminal_z.png")

    # ── p_yes stats by bins ──
    bins = [(0, 0.3, "low"), (0.3, 0.5, "medium-low"), (0.5, 0.7, "medium-high"), (0.7, 1.01, "high")]
    bin_stats: list[dict[str, Any]] = []
    for lo, hi, label in bins:
        mask = (p_yes_arr >= lo) & (p_yes_arr < hi)
        bin_stats.append({
            "range": f"[{lo:.1f}, {hi:.1f})",
            "label": label,
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
        })

    result = {
        "n_total": n_total,
        "n_success_05": n_success,
        "n_success_08": int((p_yes_arr >= 0.8).sum()),
        "p_yes_mean": float(p_yes_arr.mean()),
        "p_yes_std": float(p_yes_arr.std()),
        "p_yes_median": float(np.median(p_yes_arr)),
        "p_yes_min": float(p_yes_arr.min()),
        "p_yes_max": float(p_yes_arr.max()),
        "bin_stats": bin_stats,
        "correlations": correlations,
    }
    with open(vlm_dir / "vlm_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[EVAL-IMG] Section 5 done. Stats saved to {vlm_dir / 'vlm_stats.json'}")
    return result


# ──────────────────────────────────────────────────────────────
# Summary Report
# ──────────────────────────────────────────────────────────────
def generate_report(
    results: dict[str, Any],
    cfg: EvalImaginationConfig,
    out_dir: Path,
) -> None:
    """Generate a summary Markdown report."""
    print("[EVAL-IMG] === Generating Summary Report ===")

    lines = [
        "# Iter-1 Imagination 合成数据评估报告",
        "",
        f"- **合成数据**: {Path(cfg.synthetic_h5).name}",
        f"- **真实数据**: {Path(cfg.real_h5).name}",
        f"- **VLM 标注**: {Path(cfg.vlm_rewards_json).name}",
        f"- **评估时间**: {__import__('datetime').datetime.now().isoformat()}",
        "",
    ]

    # Section 2: Latent
    if "latent" in results:
        lat = results["latent"]
        ch = lat.get("channel_stats", {})
        lines += [
            "## 2. Latent 统计 (Synthetic vs Real)",
            "",
            "| Channel | Syn Mean | Syn Std | Real Mean | Real Std | Δ Mean |",
            "|---------|----------|---------|-----------|----------|--------|",
        ]
        if ch:
            for c in range(4):
                sm = ch["synthetic"]["mean"][c]
                ss = ch["synthetic"]["std"][c]
                rm = ch["real"]["mean"][c]
                rs = ch["real"]["std"][c]
                dm = ch["delta_mean"][c]
                lines.append(f"| {c} | {sm:.4f} | {ss:.4f} | {rm:.4f} | {rs:.4f} | {dm:+.4f} |")
        lines += [
            "",
            f"- Real L2 norm: {lat.get('real_l2_mean', 0):.2f} ± {lat.get('real_l2_std', 0):.2f}",
        ]
        rd = lat.get("round_l2", [])
        if rd:
            lines.append(f"- Syn L2 round 0: {rd[0]['l2_mean']:.2f}, round 11: {rd[-1]['l2_mean']:.2f}")
            drift = abs(rd[-1]["l2_mean"] - rd[0]["l2_mean"])
            lines.append(f"- L2 drift (R11-R0): {drift:.2f}")
        lines.append("")

    # Section 3: Actions
    if "action" in results:
        act = results["action"]
        lines += [
            "## 3. Action 分析",
            "",
            f"- Smoothness: {act.get('smoothness_mean', 0):.4f} ± {act.get('smoothness_std', 0):.4f}",
            f"- Anomalous trajs: {act.get('n_anomalous', 0)}/200",
        ]
        if act.get("anomalies"):
            for tk, issues in list(act["anomalies"].items())[:5]:
                lines.append(f"  - {tk}: {issues}")
        lines += [
            "",
            "| Dim | Syn Mean | Syn Std | Real Mean | Real Std |",
            "|-----|----------|---------|-----------|----------|",
        ]
        for dname, ds in act.get("dim_stats", {}).items():
            lines.append(
                f"| {dname} | {ds['syn_mean']:.4f} | {ds['syn_std']:.4f} "
                f"| {ds['real_mean']:.4f} | {ds['real_std']:.4f} |"
            )
        lines.append("")

    # Section 4: States
    if "state" in results:
        st = results["state"]
        lines += [
            "## 4. State 轨迹分析",
            "",
            f"- Success trajs: {st.get('n_success', 0)}",
            f"- Fail trajs: {st.get('n_fail', 0)}",
            f"- State NaN: {st.get('has_nan', False)}, Inf: {st.get('has_inf', False)}",
        ]
        ts = st.get("terminal_stats", {})
        if ts:
            lines += [
                "",
                "### Terminal state comparison",
                "| Dim | Suc Mean ± Std | Fail Mean ± Std |",
                "|-----|----------------|-----------------|",
            ]
            for dname, vals in ts.items():
                sm = vals.get("suc_mean", float("nan"))
                ss = vals.get("suc_std", float("nan"))
                fm = vals.get("fail_mean", float("nan"))
                fs = vals.get("fail_std", float("nan"))
                lines.append(f"| {dname} | {sm:.4f} ± {ss:.4f} | {fm:.4f} ± {fs:.4f} |")
        lines.append("")

    # Section 5: VLM
    if "vlm" in results:
        vlm = results["vlm"]
        lines += [
            "## 5. VLM 标注分解",
            "",
            f"- p_yes mean: {vlm.get('p_yes_mean', 0):.4f} ± {vlm.get('p_yes_std', 0):.4f}",
            f"- p_yes range: [{vlm.get('p_yes_min', 0):.4f}, {vlm.get('p_yes_max', 0):.4f}]",
            f"- p_yes median: {vlm.get('p_yes_median', 0):.4f}",
            f"- Success (α=0.5): {vlm.get('n_success_05', 0)}/200 ({vlm.get('n_success_05', 0) / 200 * 100:.1f}%)",
            f"- Success (α=0.8): {vlm.get('n_success_08', 0)}/200 ({vlm.get('n_success_08', 0) / 200 * 100:.1f}%)",
        ]
        bins = vlm.get("bin_stats", [])
        if bins:
            lines += [
                "",
                "| p_yes Range | Count | Fraction |",
                "|-------------|-------|----------|",
            ]
            for b in bins:
                lines.append(f"| {b['range']} {b['label']} | {b['count']} | {b['fraction']:.1%} |")

        corrs = vlm.get("correlations", {})
        if corrs:
            lines += ["", "### Correlations"]
            for k, v in corrs.items():
                lines.append(f"- {k}: r={v:.3f}")
        lines.append("")

    # Section 1: Decode (if available)
    if "decode" in results:
        dec = results["decode"]
        lines += [
            "## 1. 逐轮解码质量",
            "",
            f"- Visualized trajs: {dec.get('vis_trajs', [])}",
        ]
        rs = dec.get("round_stats", {})
        if rs:
            lines += [
                "",
                "| Round | Brightness | Color Var | Sharpness |",
                "|-------|-----------|-----------|-----------|",
            ]
            n_rounds = len(rs.get("brightness", []))
            for i in range(n_rounds):
                b = rs["brightness"][i]
                c = rs["color_var"][i]
                s = rs["sharpness"][i]
                lines.append(
                    f"| {b['round']} | {b['mean']:.1f}±{b['std']:.1f} "
                    f"| {c['mean']:.1f}±{c['std']:.1f} "
                    f"| {s['mean']:.1f}±{s['std']:.1f} |"
                )
        lines.append("")

    # Output files
    lines += [
        "## 输出文件",
        "",
        f"- 报告: `{out_dir / 'report.md'}`",
        f"- Latent: `{out_dir / 'latent/'}`",
        f"- Action: `{out_dir / 'action/'}`",
        f"- State: `{out_dir / 'state/'}`",
        f"- VLM: `{out_dir / 'vlm/'}`",
    ]
    if "decode" in results:
        lines.append(f"- Decode: `{out_dir / 'decode/'}`")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[EVAL-IMG] Report saved to {report_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """Run imagination evaluation pipeline."""
    # Parse args via tyro
    try:
        import tyro
        cfg = tyro.cli(EvalImaginationConfig)
    except ImportError:
        # Fallback: parse manually
        import argparse
        parser = argparse.ArgumentParser()
        for f_name, f_obj in EvalImaginationConfig.__dataclass_fields__.items():
            default = f_obj.default
            if isinstance(default, bool):
                parser.add_argument(f"--{f_name}", action="store_true" if not default else "store_false")
            else:
                parser.add_argument(f"--{f_name}", type=type(default), default=default)
        args = parser.parse_args()
        cfg = EvalImaginationConfig(**vars(args))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = _parse_sections(cfg.sections)
    print(f"[EVAL-IMG] Active sections: {sections}")
    print(f"[EVAL-IMG] Output dir: {out_dir}")
    print(f"[EVAL-IMG] Visualize (VAE decode): {cfg.visualize}")

    # ── Load data ──
    print("[EVAL-IMG] Loading synthetic data...")
    syn_data = _load_synthetic(cfg.synthetic_h5)
    print(f"[EVAL-IMG] Loaded {len(syn_data)} synthetic trajectories")

    vlm_results = _load_vlm_rewards(cfg.vlm_rewards_json)
    print(f"[EVAL-IMG] Loaded {len(vlm_results)} VLM results")

    results: dict[str, Any] = {}

    # ── Section 2: Latent stats (no GPU needed) ──
    if "latent" in sections:
        results["latent"] = eval_latent_stats(syn_data, cfg, out_dir)

    # ── Section 3: Actions (no GPU needed) ──
    if "action" in sections:
        results["action"] = eval_actions(syn_data, cfg, out_dir)

    # ── Section 4: States (no GPU needed) ──
    if "state" in sections:
        results["state"] = eval_states(syn_data, vlm_results, cfg, out_dir)

    # ── Section 5: VLM labels (no GPU needed) ──
    if "vlm" in sections:
        results["vlm"] = eval_vlm_labels(
            syn_data, vlm_results,
            results.get("action"),
            cfg, out_dir,
        )

    # ── Section 1: Decode quality (needs GPU + VAE) ──
    if "decode" in sections:
        if cfg.visualize:
            results["decode"] = eval_decode_quality(syn_data, vlm_results, cfg, out_dir)
        else:
            print("[EVAL-IMG] ⚠️  Skipping Section 1 (decode) — use --visualize to enable")

    # ── Summary report ──
    generate_report(results, cfg, out_dir)

    # Save full results JSON
    full_json = out_dir / "full_results.json"
    with open(full_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[EVAL-IMG] Full results saved to {full_json}")
    print("[EVAL-IMG] ✅ Evaluation complete!")


if __name__ == "__main__":
    main()
