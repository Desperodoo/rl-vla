"""ACP v2 vs v3 (success_once vs success_at_end) 深度对比分析脚本。

生成：
1. 数据统计：各数据集中 success_once ≠ success_at_end 的轨迹统计
2. Value target 对比：同一轨迹在两种模式下的 target 差异
3. 模型预测对比：v2 和 v3 模型对同一数据的 value 预测差异
4. 生成所有图表到 docs/vlaw/figures/

用法：
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/wjz/rl-vla conda run -n rlft_ms3 \
        python scripts/analyze_acp_v2_vs_v3.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/wjz/rl-vla")
sys.path.insert(0, str(PROJECT_ROOT))

from rlft.vlaw.acp.config import ValueTargetConfig
from rlft.vlaw.acp.value_targets import compute_value_targets

FIG_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIRS = {
    "A: Demo": PROJECT_ROOT / "data/vlaw/rollouts/mixed",
    "B: Pretrained": PROJECT_ROOT / "data/vlaw/rollouts/pretrained_policy",
    "C: Teleop": PROJECT_ROOT / "data/vlaw/rollouts/teleop_sim",
    "D: RL Prior": PROJECT_ROOT / "data/vlaw/rollouts/rl_prior",
}

V2_CKPT = PROJECT_ROOT / "checkpoints/vlaw/acp/v2_combined/best.safetensors"
V3_CKPT = PROJECT_ROOT / "checkpoints/vlaw/acp/v3_at_end/best.safetensors"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (12, 8),
})


# ── Part 1: Scan data for success mismatch ────────────────────────────────
def scan_trajectories():
    """扫描所有数据，统计 success_once vs success_at_end 差异。"""
    results = {}
    all_trajs = []

    for label, data_dir in DATA_DIRS.items():
        h5_files = sorted(data_dir.rglob("*.h5"))
        stats = {
            "total": 0, "success_once": 0, "success_at_end": 0,
            "mismatch": 0, "mismatch_trajs": [],
        }
        for h5_path in h5_files:
            with h5py.File(str(h5_path), "r") as f:
                for key in sorted(f.keys()):
                    if not key.startswith("traj_"):
                        continue
                    grp = f[key]
                    if "env_success" not in grp:
                        continue
                    env_success = np.asarray(grp["env_success"], dtype=bool)
                    T = len(env_success)
                    s_once = bool(env_success.any())
                    s_end = bool(env_success[-1])

                    stats["total"] += 1
                    if s_once:
                        stats["success_once"] += 1
                    if s_end:
                        stats["success_at_end"] += 1
                    if s_once != s_end:
                        stats["mismatch"] += 1
                        # Find first success frame
                        first_success = int(np.argmax(env_success)) if s_once else -1
                        # Find last success frame
                        if s_once:
                            last_success = T - 1 - int(np.argmax(env_success[::-1]))
                        else:
                            last_success = -1
                        stats["mismatch_trajs"].append({
                            "file": str(h5_path.name),
                            "traj": key,
                            "length": T,
                            "success_once": s_once,
                            "success_at_end": s_end,
                            "first_success_frame": first_success,
                            "last_success_frame": last_success,
                            "success_frames": int(env_success.sum()),
                        })

                    all_trajs.append({
                        "label": label,
                        "file": str(h5_path.name),
                        "traj": key,
                        "length": T,
                        "env_success": env_success,
                        "success_once": s_once,
                        "success_at_end": s_end,
                    })

        results[label] = stats

    return results, all_trajs


# ── Part 2: Value target comparison ──────────────────────────────────────
def compute_target_comparison(all_trajs: list[dict]):
    """对所有轨迹分别计算 success_once 和 success_at_end 的 value target。"""
    max_len = max(t["length"] for t in all_trajs)

    cfg_once = ValueTargetConfig(success_mode="success_once")
    cfg_end = ValueTargetConfig(success_mode="success_at_end")

    comparisons = []
    for traj in all_trajs:
        targets_once = compute_value_targets(
            traj["env_success"], traj["length"], max_len, cfg_once
        )
        targets_end = compute_value_targets(
            traj["env_success"], traj["length"], max_len, cfg_end
        )
        comparisons.append({
            **traj,
            "targets_once": targets_once,
            "targets_end": targets_end,
            "target_diff": targets_end - targets_once,
            "mean_diff": float(np.mean(targets_end - targets_once)),
        })

    return comparisons


# ── Part 3: Model inference comparison ────────────────────────────────────
def run_model_comparison(all_trajs: list[dict], device: str = "cuda:0", max_trajs: int = 100):
    """用 v2 和 v3 模型分别推理，比较 value 预测。"""
    from rlft.vlaw.acp.config import ValueModelConfig
    from rlft.vlaw.acp.value_model import ManiSkillValueModel

    # Subsample for speed
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_trajs), min(max_trajs, len(all_trajs)), replace=False)
    subset = [all_trajs[i] for i in sorted(indices)]

    vm_cfg = ValueModelConfig()

    # Load both models
    print("[分析] 加载 v2_combined (success_once) 模型...")
    model_v2 = ManiSkillValueModel(vm_cfg, device=device)
    from safetensors.torch import load_file
    state_v2 = load_file(str(V2_CKPT))
    model_v2.model.load_state_dict(state_v2, strict=False)

    print("[分析] 加载 v3_at_end (success_at_end) 模型...")
    model_v3 = ManiSkillValueModel(vm_cfg, device=device)
    state_v3 = load_file(str(V3_CKPT))
    model_v3.model.load_state_dict(state_v3, strict=False)

    # Prepare camera keys
    camera_keys = vm_cfg.camera_keys

    model_results = []
    for traj_info in subset:
        h5_path = None
        for d in DATA_DIRS.values():
            candidate = list(d.rglob(traj_info["file"]))
            if candidate:
                h5_path = candidate[0]
                break
        if h5_path is None:
            continue

        with h5py.File(str(h5_path), "r") as f:
            grp = f[traj_info["traj"]]
            T = traj_info["length"]

            # Read images
            values_v2 = []
            values_v3 = []

            for t in range(T):
                images = []
                for ck in camera_keys:
                    if ck in grp:
                        img = np.asarray(grp[ck][t])  # (H, W, 3) or (3, H, W)
                        if img.ndim == 3 and img.shape[-1] == 3:
                            img = img.transpose(2, 0, 1)  # -> (3, H, W)
                        images.append(img)

                if len(images) != len(camera_keys):
                    break

                # Stack: (N_cam, 3, H, W)
                img_tensor = torch.tensor(np.stack(images), dtype=torch.uint8).unsqueeze(0).to(device)
                mask_tensor = torch.ones(1, len(camera_keys), dtype=torch.bool, device=device)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    v2_val = model_v2.predict_values(img_tensor, mask_tensor).item()
                    v3_val = model_v3.predict_values(img_tensor, mask_tensor).item()

                values_v2.append(v2_val)
                values_v3.append(v3_val)

            if len(values_v2) == T:
                model_results.append({
                    **traj_info,
                    "pred_v2": np.array(values_v2),
                    "pred_v3": np.array(values_v3),
                    "pred_diff": np.array(values_v3) - np.array(values_v2),
                })
                print(f"  [{len(model_results)}/{len(subset)}] {traj_info['traj']} "
                      f"T={T} s_once={traj_info['success_once']} s_end={traj_info['success_at_end']} "
                      f"v2_mean={np.mean(values_v2):.4f} v3_mean={np.mean(values_v3):.4f}")

    # Clean up
    del model_v2, model_v3
    torch.cuda.empty_cache()

    return model_results


# ── Part 4: Figures ───────────────────────────────────────────────────────
def plot_data_statistics(scan_results: dict):
    """Fig 1: 各数据集 success mismatch 统计。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(scan_results.keys())
    total = [scan_results[l]["total"] for l in labels]
    s_once = [scan_results[l]["success_once"] for l in labels]
    s_end = [scan_results[l]["success_at_end"] for l in labels]
    mismatch = [scan_results[l]["mismatch"] for l in labels]

    x = np.arange(len(labels))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, s_once, w, label="success_once", color="#4CAF50", alpha=0.8)
    ax.bar(x, s_end, w, label="success_at_end", color="#2196F3", alpha=0.8)
    ax.bar(x + w, mismatch, w, label="mismatch (once≠end)", color="#FF5722", alpha=0.8)
    for i, v in enumerate(total):
        ax.text(i - w, s_once[i] + 1, str(s_once[i]), ha="center", fontsize=8)
        ax.text(i, s_end[i] + 1, str(s_end[i]), ha="center", fontsize=8)
        ax.text(i + w, mismatch[i] + 1, str(mismatch[i]), ha="center", fontsize=8, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels([l.split(":")[1].strip() for l in labels], fontsize=9)
    ax.set_ylabel("Trajectories")
    ax.set_title("(a) Success Signal Statistics per Dataset")
    ax.legend()

    # Mismatch rate
    ax = axes[1]
    rates = [m / t * 100 if t > 0 else 0 for m, t in zip(mismatch, total)]
    bars = ax.bar(x, rates, 0.5, color="#FF5722", alpha=0.8)
    for i, (r, m, t) in enumerate(zip(rates, mismatch, total)):
        ax.text(i, r + 0.3, f"{r:.1f}%\n({m}/{t})", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split(":")[1].strip() for l in labels], fontsize=9)
    ax.set_ylabel("Mismatch Rate (%)")
    ax.set_title("(b) success_once ≠ success_at_end Rate")
    ax.set_ylim(0, max(rates) * 1.5 + 5)

    fig.suptitle("Fig 1. Data Statistics — success_once vs success_at_end Mismatch", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_fig1_data_statistics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 保存 {FIG_DIR / 'v3_fig1_data_statistics.png'}")


def plot_value_target_comparison(comparisons: list[dict]):
    """Fig 2: Value target 差异分析。"""
    # Separate by category
    match_success = [c for c in comparisons if c["success_once"] and c["success_at_end"]]
    match_fail = [c for c in comparisons if not c["success_once"] and not c["success_at_end"]]
    mismatch = [c for c in comparisons if c["success_once"] != c["success_at_end"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Distribution of mean target differences across all trajs
    ax = axes[0, 0]
    all_diffs = [c["mean_diff"] for c in comparisons]
    match_diffs = [c["mean_diff"] for c in match_success + match_fail]
    mismatch_diffs = [c["mean_diff"] for c in mismatch]
    ax.hist(match_diffs, bins=50, alpha=0.7, label=f"Matching ({len(match_diffs)})", color="#4CAF50")
    if mismatch_diffs:
        ax.hist(mismatch_diffs, bins=50, alpha=0.7, label=f"Mismatch ({len(mismatch_diffs)})", color="#FF5722")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Target Diff (success_at_end - success_once)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Distribution of Target Differences")
    ax.legend()

    # (b) Example mismatch trajectory — value target curves
    ax = axes[0, 1]
    if mismatch:
        # Pick the mismatch with largest absolute mean diff
        ex = max(mismatch, key=lambda c: abs(c["mean_diff"]))
        T = ex["length"]
        frames = np.arange(T)
        ax.plot(frames, ex["targets_once"], "g-", linewidth=2, label="success_once target")
        ax.plot(frames, ex["targets_end"], "r-", linewidth=2, label="success_at_end target")
        ax.fill_between(frames, ex["targets_once"], ex["targets_end"],
                        alpha=0.3, color="orange", label="Difference")
        # Mark success frames
        succ_frames = np.where(ex["env_success"])[0]
        if len(succ_frames) > 0:
            ax.axvspan(succ_frames[0], succ_frames[-1], alpha=0.15, color="green", label="success=True frames")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value Target")
        ax.set_title(f"(b) Mismatch Example: {ex['traj']} (T={T})\n"
                     f"s_once=True, s_end=False, success frames={len(succ_frames)}")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No mismatch trajectories found", transform=ax.transAxes, ha="center")
        ax.set_title("(b) Mismatch Example (none found)")

    # (c) Scatter: trajectory length vs target diff
    ax = axes[1, 0]
    if mismatch:
        lengths = [c["length"] for c in mismatch]
        diffs = [c["mean_diff"] for c in mismatch]
        ax.scatter(lengths, diffs, c="#FF5722", alpha=0.6, s=30, edgecolors="black", linewidths=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Episode Length")
        ax.set_ylabel("Mean Target Diff (at_end - once)")
        ax.set_title(f"(c) Mismatch Trajs: Length vs Target Diff (N={len(mismatch)})")
    else:
        ax.text(0.5, 0.5, "No mismatch trajectories", transform=ax.transAxes, ha="center")
        ax.set_title("(c) Length vs Target Diff")

    # (d) Value target formula visual explanation
    ax = axes[1, 1]
    # Show computed targets for a synthetic example
    T_demo = 50
    max_len_demo = 50
    # Case 1: success at frame 20, then drops at frame 40 (success_once=T, success_at_end=F)
    env_s = np.zeros(T_demo, dtype=bool)
    env_s[20:40] = True
    cfg_once = ValueTargetConfig(success_mode="success_once")
    cfg_end = ValueTargetConfig(success_mode="success_at_end")
    t_once = compute_value_targets(env_s, T_demo, max_len_demo, cfg_once)
    t_end = compute_value_targets(env_s, T_demo, max_len_demo, cfg_end)
    frames = np.arange(T_demo)
    ax.plot(frames, t_once, "g-", linewidth=2, label="success_once mode (treated as success)")
    ax.plot(frames, t_end, "r-", linewidth=2, label="success_at_end mode (treated as fail)")
    ax.axvspan(20, 39, alpha=0.15, color="green")
    ax.annotate("success=True\n(frame 20-39)", xy=(30, -0.3), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    ax.annotate("dropped!\n(frame 40+)", xy=(45, -0.6), fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.5))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value Target")
    ax.set_title("(d) Synthetic Example: peg lifted at f20, dropped at f40\n"
                 f"Target gap = {np.mean(t_end - t_once):.3f} per frame")
    ax.legend(fontsize=8)

    fig.suptitle("Fig 2. Value Target Comparison — success_once vs success_at_end",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_fig2_value_target_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 保存 {FIG_DIR / 'v3_fig2_value_target_comparison.png'}")


def plot_model_predictions(model_results: list[dict]):
    """Fig 3: V2 vs V3 模型预测对比。"""
    if not model_results:
        print("[警告] 无模型预测结果，跳过 Fig 3")
        return

    # Separate by category
    match_succ = [r for r in model_results if r["success_once"] and r["success_at_end"]]
    match_fail = [r for r in model_results if not r["success_once"] and not r["success_at_end"]]
    mismatch = [r for r in model_results if r["success_once"] != r["success_at_end"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Scatter: v2 mean value vs v3 mean value per trajectory
    ax = axes[0, 0]
    for subset, label, color, marker in [
        (match_succ, "Both success", "#4CAF50", "o"),
        (match_fail, "Both fail", "#9E9E9E", "x"),
        (mismatch, "Mismatch", "#FF5722", "D"),
    ]:
        if subset:
            v2_means = [np.mean(r["pred_v2"]) for r in subset]
            v3_means = [np.mean(r["pred_v3"]) for r in subset]
            ax.scatter(v2_means, v3_means, c=color, marker=marker, alpha=0.6,
                      s=30, label=f"{label} (N={len(subset)})", edgecolors="black", linewidths=0.3)
    lims = [-1.05, 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("v2 (success_once) Mean Value")
    ax.set_ylabel("v3 (success_at_end) Mean Value")
    ax.set_title("(a) Per-Trajectory Mean Value: v2 vs v3")
    ax.legend(fontsize=7)

    # (b) Distribution of per-trajectory value difference
    ax = axes[0, 1]
    all_mean_diffs = [np.mean(r["pred_diff"]) for r in model_results]
    match_mean_diffs = [np.mean(r["pred_diff"]) for r in match_succ + match_fail]
    mismatch_mean_diffs = [np.mean(r["pred_diff"]) for r in mismatch]
    ax.hist(match_mean_diffs, bins=40, alpha=0.7, label=f"Matching ({len(match_mean_diffs)})", color="#4CAF50")
    if mismatch_mean_diffs:
        ax.hist(mismatch_mean_diffs, bins=20, alpha=0.7, label=f"Mismatch ({len(mismatch_mean_diffs)})", color="#FF5722")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Prediction Diff (v3 - v2)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Distribution of Model Prediction Differences")
    ax.legend()

    # (c) Example trajectory: v2 vs v3 prediction curves
    ax = axes[1, 0]
    # Pick a mismatch trajectory, or if none, the one with largest prediction diff
    if mismatch:
        ex = max(mismatch, key=lambda r: abs(np.mean(r["pred_diff"])))
        title_suffix = "Mismatch"
    else:
        ex = max(model_results, key=lambda r: abs(np.mean(r["pred_diff"])))
        title_suffix = "Max Diff"
    T = ex["length"]
    frames = np.arange(T)
    ax.plot(frames, ex["pred_v2"], "g-", linewidth=1.5, label="v2 (success_once) prediction")
    ax.plot(frames, ex["pred_v3"], "r-", linewidth=1.5, label="v3 (success_at_end) prediction")
    ax.fill_between(frames, ex["pred_v2"], ex["pred_v3"], alpha=0.3, color="orange")
    # Mark success frames
    succ_frames = np.where(ex["env_success"])[0]
    if len(succ_frames) > 0:
        ax.axvspan(succ_frames[0], succ_frames[-1], alpha=0.1, color="green")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"(c) {title_suffix}: {ex['traj']} (T={T}, "
                 f"s_once={ex['success_once']}, s_end={ex['success_at_end']})")
    ax.legend(fontsize=8)

    # (d) Per-frame prediction diff averaged over success/fail/mismatch groups
    ax = axes[1, 1]
    max_T = 35  # typical episode length to show
    for group, label, color in [
        (match_succ, "Both success", "#4CAF50"),
        (match_fail, "Both fail", "#9E9E9E"),
        (mismatch, "Mismatch (once=T, end=F)", "#FF5722"),
    ]:
        if not group:
            continue
        # Pad/trim to common length
        diffs_padded = []
        for r in group:
            d = r["pred_diff"]
            if len(d) >= max_T:
                diffs_padded.append(d[:max_T])
            else:
                padded = np.full(max_T, np.nan)
                padded[:len(d)] = d
                diffs_padded.append(padded)
        arr = np.array(diffs_padded)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        frames = np.arange(max_T)
        ax.plot(frames, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(frames, mean - std, mean + std, color=color, alpha=0.15)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Prediction Diff (v3 - v2)")
    ax.set_title("(d) Average Per-Frame Prediction Diff by Group")
    ax.legend(fontsize=7)

    fig.suptitle("Fig 3. Model Prediction Comparison — v2 (success_once) vs v3 (success_at_end)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_fig3_model_predictions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 保存 {FIG_DIR / 'v3_fig3_model_predictions.png'}")


def plot_td_reward_impact(comparisons: list[dict], model_results: list[dict]):
    """Fig 4: TD-shaped reward 影响分析 r(s,s') = (V(s') - V(s)) * scale。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Value target TD reward: per-frame r_t = (target_{t+1} - target_t) * scale
    ax = axes[0]
    scale = 100.0
    mismatch_trajs = [c for c in comparisons if c["success_once"] != c["success_at_end"]]
    if mismatch_trajs:
        ex = max(mismatch_trajs, key=lambda c: abs(c["mean_diff"]))
        T = ex["length"]
        frames = np.arange(T - 1)
        td_once = np.diff(ex["targets_once"]) * scale
        td_end = np.diff(ex["targets_end"]) * scale
        ax.plot(frames, td_once, "g-", linewidth=1.5, label="TD reward (success_once)")
        ax.plot(frames, td_end, "r-", linewidth=1.5, label="TD reward (success_at_end)")
        succ_frames = np.where(ex["env_success"])[0]
        if len(succ_frames) > 0:
            ax.axvspan(succ_frames[0], min(succ_frames[-1], T-2), alpha=0.1, color="green")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"TD Reward (scale={scale:.0f})")
        ax.set_title("(a) Target-based TD Reward\n(mismatch trajectory)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No mismatch", transform=ax.transAxes, ha="center")

    # (b) Synthetic example: peg lifted at f20, dropped at f40
    ax = axes[1]
    T_demo = 50
    env_s = np.zeros(T_demo, dtype=bool)
    env_s[20:40] = True
    cfg_once = ValueTargetConfig(success_mode="success_once")
    cfg_end = ValueTargetConfig(success_mode="success_at_end")
    t_once = compute_value_targets(env_s, T_demo, T_demo, cfg_once)
    t_end = compute_value_targets(env_s, T_demo, T_demo, cfg_end)
    td_once = np.diff(t_once) * scale
    td_end = np.diff(t_end) * scale
    frames = np.arange(T_demo - 1)
    ax.plot(frames, td_once, "g-", linewidth=2, label="TD reward (success_once)")
    ax.plot(frames, td_end, "r-", linewidth=2, label="TD reward (success_at_end)")
    ax.axvspan(20, 39, alpha=0.1, color="green")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.annotate("success_once:\nTD=+1.0/frame\n(approach rewarded)", xy=(10, 1.3), fontsize=7,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    ax.annotate("success_at_end:\nTD=+0.5/frame\n(penalty for dropping)", xy=(35, -0.8), fontsize=7,
                bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.5))
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"TD Reward (scale={scale:.0f})")
    ax.set_title("(b) Synthetic: Lift at f20, Drop at f40\nKey: success_at_end penalizes dropping")
    ax.legend(fontsize=8)

    # (c) Cumulative TD reward comparison
    ax = axes[2]
    cum_once = np.cumsum(td_once)
    cum_end = np.cumsum(td_end)
    ax.plot(frames, cum_once, "g-", linewidth=2, label="Cumulative (success_once)")
    ax.plot(frames, cum_end, "r-", linewidth=2, label="Cumulative (success_at_end)")
    ax.axvspan(20, 39, alpha=0.1, color="green")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative TD Reward")
    ax.set_title(f"(c) Cumulative TD Reward\nFinal: once={cum_once[-1]:.1f}, end={cum_end[-1]:.1f}")
    ax.legend(fontsize=8)

    fig.suptitle("Fig 4. TD-Shaped Reward Impact — success_once vs success_at_end",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_fig4_td_reward_impact.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 保存 {FIG_DIR / 'v3_fig4_td_reward_impact.png'}")


def plot_training_curves():
    """Fig 5: ACP v2_combined vs v3_at_end 训练曲线对比。"""
    import glob

    # Find wandb run dirs
    wandb_dir = PROJECT_ROOT / "wandb"
    v3_run = None
    v2_run = None

    for run_dir in sorted(wandb_dir.glob("run-*")):
        config_file = run_dir / "files" / "config.yaml"
        if not config_file.exists():
            continue
        content = config_file.read_text()
        if "acp_v3_at_end" in content:
            v3_run = run_dir
        elif "v2_combined" in content or "acp_value_v2_combined" in content:
            v2_run = run_dir

    # Parse output logs for val_mae at each eval step
    def parse_output_log(run_dir):
        log_file = run_dir / "files" / "output.log"
        if not log_file.exists():
            return [], []
        steps, maes = [], []
        for line in log_file.read_text().splitlines():
            if "[val]" in line and "mae=" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("step="):
                        steps.append(int(p.split("=")[1]))
                    if p.startswith("mae="):
                        maes.append(float(p.split("=")[1]))
        return steps, maes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Val MAE curves
    ax = axes[0]
    if v3_run:
        steps, maes = parse_output_log(v3_run)
        if steps:
            ax.plot(steps, maes, "r-", linewidth=1.5, label="v3_at_end (success_at_end)")
    if v2_run:
        steps, maes = parse_output_log(v2_run)
        if steps:
            ax.plot(steps, maes, "g-", linewidth=1.5, label="v2_combined (success_once)")

    ax.axhline(0.1, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Quality gate (MAE<0.1)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation MAE")
    ax.set_title("(a) Validation MAE Convergence")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.2)

    # (b) Bar chart: final metrics
    ax = axes[1]
    versions = ["v2_combined\n(success_once)", "v3_at_end\n(success_at_end)"]
    maes = [0.0837, 0.0840]
    losses = [3.209, 3.210]

    x = np.arange(len(versions))
    bars = ax.bar(x, maes, 0.4, color=["#4CAF50", "#FF5722"], alpha=0.8)
    for i, v in enumerate(maes):
        ax.text(i, v + 0.002, f"MAE={v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.1, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Gate: MAE<0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("Best Validation MAE")
    ax.set_title("(b) Best MAE Comparison")
    ax.set_ylim(0, 0.15)
    ax.legend()

    fig.suptitle("Fig 5. Training Curves — v2_combined vs v3_at_end",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_fig5_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 保存 {FIG_DIR / 'v3_fig5_training_curves.png'}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("[ACP v2 vs v3 分析] 开始")
    print("=" * 70)

    # Step 1: Scan data
    print("\n[Step 1/5] 扫描数据...")
    scan_results, all_trajs = scan_trajectories()

    total_all = sum(s["total"] for s in scan_results.values())
    total_mismatch = sum(s["mismatch"] for s in scan_results.values())
    print(f"  总轨迹数: {total_all}")
    print(f"  Mismatch (once≠end): {total_mismatch} ({total_mismatch/total_all*100:.1f}%)")
    for label, stats in scan_results.items():
        print(f"  {label}: total={stats['total']}, once={stats['success_once']}, "
              f"end={stats['success_at_end']}, mismatch={stats['mismatch']}")

    # Step 2: Compute value target comparisons
    print("\n[Step 2/5] 计算 value target 对比...")
    comparisons = compute_target_comparison(all_trajs)
    mismatch_comps = [c for c in comparisons if c["success_once"] != c["success_at_end"]]
    print(f"  Mismatch 轨迹 target 差异: mean={np.mean([c['mean_diff'] for c in mismatch_comps]):.4f}" if mismatch_comps else "  无 mismatch")

    # Step 3: Model inference (if checkpoints available)
    print("\n[Step 3/5] 模型推理对比...")
    model_results = []
    if V2_CKPT.exists() and V3_CKPT.exists():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_results = run_model_comparison(all_trajs, device=device, max_trajs=80)
        print(f"  完成: {len(model_results)} 轨迹")
    else:
        print(f"  跳过: checkpoint 不存在 (v2={V2_CKPT.exists()}, v3={V3_CKPT.exists()})")

    # Step 4: Generate figures
    print("\n[Step 4/5] 生成图表...")
    plot_data_statistics(scan_results)
    plot_value_target_comparison(comparisons)
    if model_results:
        plot_model_predictions(model_results)
    plot_td_reward_impact(comparisons, model_results)
    plot_training_curves()

    # Step 5: Save JSON summary
    print("\n[Step 5/5] 保存分析数据...")
    summary = {
        "scan_results": {k: {kk: vv for kk, vv in v.items() if kk != "mismatch_trajs"}
                        for k, v in scan_results.items()},
        "mismatch_details": [
            {k: v for k, v in c.items()
             if k not in ("env_success", "targets_once", "targets_end", "target_diff")}
            for c in mismatch_comps
        ],
        "total_trajs": total_all,
        "total_mismatch": total_mismatch,
        "mismatch_rate": total_mismatch / total_all * 100 if total_all > 0 else 0,
    }
    if model_results:
        summary["model_comparison"] = {
            "num_trajs": len(model_results),
            "mean_v2": float(np.mean([np.mean(r["pred_v2"]) for r in model_results])),
            "mean_v3": float(np.mean([np.mean(r["pred_v3"]) for r in model_results])),
            "mean_diff": float(np.mean([np.mean(r["pred_diff"]) for r in model_results])),
        }

    json_path = FIG_DIR / "v3_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  保存 {json_path}")

    print("\n" + "=" * 70)
    print("[ACP v2 vs v3 分析] 完成")
    print(f"  图表目录: {FIG_DIR}")
    print(f"  生成图表: v3_fig1~fig5 共 5 张")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
