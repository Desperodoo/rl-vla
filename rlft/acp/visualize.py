"""ACP 可视化模块 — Value 预测 vs Ground Truth 对比 + Advantage 分布分析.

生成以下可视化图表：
  1. Value prediction vs ground truth 散点图
  2. Per-trajectory value prediction 曲线 (抽样)
  3. Advantage 分布直方图
  4. Per-trajectory advantage 热力图
  5. Weight 分布与阈值分析
  6. 成功/失败轨迹的 value 预测对比
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_acp_annotations(hdf5_paths: list[Path]) -> dict[str, Any]:
    """从 HDF5 文件加载 ACP 标注数据。

    Returns:
        dict 包含:
          - targets: (N,) value targets
          - preds: (N,) value predictions
          - advantages: (N,) advantages
          - indicators: (N,) binary indicators
          - weights: (N,) continuous weights
          - traj_data: list[dict] 每条轨迹的数据
    """
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_advantages: list[np.ndarray] = []
    all_indicators: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    traj_data: list[dict] = []

    for hp in hdf5_paths:
        with h5py.File(str(hp), "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            for tk in traj_keys:
                grp = f[tk]
                if "acp_value_target" not in grp:
                    continue

                targets = grp["acp_value_target"][:].astype(np.float32)
                preds = grp["acp_value_pred"][:].astype(np.float32)
                advantages = grp["acp_advantage"][:].astype(np.float32)
                indicators = grp["acp_indicator"][:].astype(np.int32)
                weights = grp["acp_weight"][:].astype(np.float32)

                success = bool(grp.attrs.get("success", False))
                env_success = grp["env_success"][:] if "env_success" in grp else None

                all_targets.append(targets)
                all_preds.append(preds)
                all_advantages.append(advantages)
                all_indicators.append(indicators)
                all_weights.append(weights)
                traj_data.append({
                    "traj_key": tk,
                    "targets": targets,
                    "preds": preds,
                    "advantages": advantages,
                    "indicators": indicators,
                    "weights": weights,
                    "success": success,
                    "env_success": env_success,
                    "length": len(targets),
                })

    if not all_targets:
        raise RuntimeError("未找到包含 ACP 标注的轨迹")

    return {
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "advantages": np.concatenate(all_advantages),
        "indicators": np.concatenate(all_indicators),
        "weights": np.concatenate(all_weights),
        "traj_data": traj_data,
    }


def plot_value_scatter(
    targets: np.ndarray, preds: np.ndarray, output_path: Path
) -> dict[str, float]:
    """散点图: value prediction vs ground truth."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(targets, preds, alpha=0.1, s=4, c="steelblue")
    ax.plot([-1, 0], [-1, 0], "r--", linewidth=1.5, label="Perfect prediction")

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    corr = float(np.corrcoef(targets, preds)[0, 1])

    ax.set_xlabel("Ground Truth Value Target", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(
        f"ACP Value: Prediction vs Ground Truth\n"
        f"MAE={mae:.4f}  RMSE={rmse:.4f}  r={corr:.4f}",
        fontsize=13,
    )
    ax.set_xlim(-1.05, 0.05)
    ax.set_ylim(-1.05, 0.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return {"mae": mae, "rmse": rmse, "correlation": corr}


def plot_trajectory_values(
    traj_data: list[dict],
    output_path: Path,
    num_samples: int = 8,
    seed: int = 42,
) -> None:
    """每条轨迹的 value target vs prediction 折线图（抽样）."""
    rng = np.random.default_rng(seed)
    # 选择 num_samples/2 成功 + num_samples/2 失败
    success_trajs = [t for t in traj_data if t["success"]]
    fail_trajs = [t for t in traj_data if not t["success"]]
    n_suc = min(num_samples // 2, len(success_trajs))
    n_fail = min(num_samples - n_suc, len(fail_trajs))
    if n_suc < num_samples // 2:
        n_fail = min(num_samples - n_suc, len(fail_trajs))

    selected = []
    if success_trajs:
        idxs = rng.choice(len(success_trajs), size=n_suc, replace=False)
        selected.extend([success_trajs[i] for i in idxs])
    if fail_trajs:
        idxs = rng.choice(len(fail_trajs), size=n_fail, replace=False)
        selected.extend([fail_trajs[i] for i in idxs])

    if not selected:
        return

    nrows = (len(selected) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (ax, traj) in enumerate(zip(axes.flat, selected)):
        t = np.arange(traj["length"])
        ax.plot(t, traj["targets"], "b-", linewidth=1.5, label="GT Target")
        ax.plot(t, traj["preds"], "r--", linewidth=1.5, label="Predicted")

        if traj["env_success"] is not None:
            success_frames = np.where(traj["env_success"])[0]
            if len(success_frames) > 0:
                ax.axvline(success_frames[0], color="green", linestyle=":",
                           alpha=0.7, label=f"First success (t={success_frames[0]})")

        status = "Success" if traj["success"] else "Fail"
        mae_i = float(np.mean(np.abs(traj["preds"] - traj["targets"])))
        ax.set_title(f"{traj['traj_key']} [{status}] MAE={mae_i:.4f}", fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.set_ylim(-1.05, 0.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)

    fig.suptitle("Per-Trajectory Value Predictions vs Ground Truth", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_advantage_distribution(
    advantages: np.ndarray,
    indicators: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """Advantage 分布直方图 + weight 分布."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Advantage histogram
    ax = axes[0]
    ax.hist(advantages, bins=80, alpha=0.7, color="steelblue", edgecolor="white")
    threshold_idx = np.where(indicators == 1)[0]
    if len(threshold_idx) > 0:
        threshold_approx = float(np.min(advantages[threshold_idx]))
        ax.axvline(threshold_approx, color="red", linestyle="--",
                   label=f"Threshold ≈ {threshold_approx:.4f}")
    positive_ratio = float(np.mean(indicators.astype(np.float32)))
    ax.set_title(f"Advantage Distribution\n"
                 f"mean={float(np.mean(advantages)):.4f}, "
                 f"std={float(np.std(advantages)):.4f}", fontsize=11)
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Indicator pie chart
    ax = axes[1]
    pos = int(np.sum(indicators))
    neg = len(indicators) - pos
    ax.pie([pos, neg], labels=[f"Positive ({pos})", f"Negative ({neg})"],
           autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    ax.set_title(f"Positive Ratio: {positive_ratio:.1%}\n({pos}/{len(indicators)} frames)")

    # 3. Weight distribution
    ax = axes[2]
    ax.hist(weights, bins=60, alpha=0.7, color="coral", edgecolor="white")
    ax.set_title(f"Continuous Weight Distribution\n"
                 f"mean={float(np.mean(weights)):.4f}, "
                 f"max={float(np.max(weights)):.4f}", fontsize=11)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.suptitle("ACP Advantage & Weight Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    return {
        "positive_ratio": positive_ratio,
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "weight_mean": float(np.mean(weights)),
        "weight_max": float(np.max(weights)),
    }


def plot_success_vs_fail_comparison(
    traj_data: list[dict], output_path: Path
) -> dict[str, float]:
    """成功 vs 失败轨迹的 value prediction 分布对比."""
    success_preds = np.concatenate(
        [t["preds"] for t in traj_data if t["success"]]
    ) if any(t["success"] for t in traj_data) else np.array([])
    fail_preds = np.concatenate(
        [t["preds"] for t in traj_data if not t["success"]]
    ) if any(not t["success"] for t in traj_data) else np.array([])

    success_targets = np.concatenate(
        [t["targets"] for t in traj_data if t["success"]]
    ) if any(t["success"] for t in traj_data) else np.array([])
    fail_targets = np.concatenate(
        [t["targets"] for t in traj_data if not t["success"]]
    ) if any(not t["success"] for t in traj_data) else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predictions
    ax = axes[0]
    if len(success_preds) > 0:
        ax.hist(success_preds, bins=60, alpha=0.6, color="#4CAF50",
                label=f"Success ({len(success_preds)} frames)", density=True)
    if len(fail_preds) > 0:
        ax.hist(fail_preds, bins=60, alpha=0.6, color="#F44336",
                label=f"Fail ({len(fail_preds)} frames)", density=True)
    ax.set_title("Value Predictions by Outcome", fontsize=12)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # MAE by group
    ax = axes[1]
    success_maes = [float(np.mean(np.abs(t["preds"] - t["targets"])))
                    for t in traj_data if t["success"]]
    fail_maes = [float(np.mean(np.abs(t["preds"] - t["targets"])))
                 for t in traj_data if not t["success"]]

    data_to_plot = []
    labels = []
    if success_maes:
        data_to_plot.append(success_maes)
        labels.append(f"Success\n(n={len(success_maes)})")
    if fail_maes:
        data_to_plot.append(fail_maes)
        labels.append(f"Fail\n(n={len(fail_maes)})")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ["#4CAF50", "#F44336"][:len(data_to_plot)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax.set_title("Per-Trajectory MAE by Outcome", fontsize=12)
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Success vs Fail Trajectory Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    n_success = sum(1 for t in traj_data if t["success"])
    n_fail = sum(1 for t in traj_data if not t["success"])
    return {
        "n_success_trajs": n_success,
        "n_fail_trajs": n_fail,
        "success_mae_mean": float(np.mean(success_maes)) if success_maes else 0.0,
        "fail_mae_mean": float(np.mean(fail_maes)) if fail_maes else 0.0,
    }


def plot_error_by_timestep(
    traj_data: list[dict], output_path: Path
) -> None:
    """误差随时间步变化的趋势图."""
    max_len = max(t["length"] for t in traj_data)
    errors_by_t: dict[int, list[float]] = {}

    for traj in traj_data:
        for t_idx in range(traj["length"]):
            err = abs(float(traj["preds"][t_idx]) - float(traj["targets"][t_idx]))
            if t_idx not in errors_by_t:
                errors_by_t[t_idx] = []
            errors_by_t[t_idx].append(err)

    timesteps = sorted(errors_by_t.keys())
    means = [np.mean(errors_by_t[t]) for t in timesteps]
    stds = [np.std(errors_by_t[t]) for t in timesteps]
    counts = [len(errors_by_t[t]) for t in timesteps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax1.plot(timesteps, means, "b-", linewidth=1.5, label="Mean |error|")
    ax1.fill_between(timesteps, means_arr - stds_arr, means_arr + stds_arr,
                     alpha=0.2, color="blue", label="±1 std")
    ax1.set_title("Prediction Error by Timestep", fontsize=13)
    ax1.set_ylabel("Absolute Error")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.bar(timesteps, counts, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Timestep (frame index within trajectory)")
    ax2.set_ylabel("# Trajectories")
    ax2.set_title("Sample Count per Timestep", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def generate_all_visualizations(
    hdf5_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """生成所有 ACP 可视化图表。

    Args:
        hdf5_paths: 包含 ACP 标注的 HDF5 文件路径
        output_dir: 输出目录

    Returns:
        dict: 所有图表的统计信息
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ACP-VIZ] 加载 ACP 标注数据...")
    data = load_acp_annotations(hdf5_paths)
    n_frames = len(data["targets"])
    n_trajs = len(data["traj_data"])
    print(f"[ACP-VIZ] 加载 {n_trajs} 条轨迹, {n_frames} 帧")

    results: dict[str, Any] = {
        "n_frames": n_frames,
        "n_trajs": n_trajs,
    }

    # 1. Value scatter plot
    print("[ACP-VIZ] 生成 06_value_scatter.png ...")
    scatter_stats = plot_value_scatter(
        data["targets"], data["preds"],
        output_dir / "06_value_scatter.png",
    )
    results.update(scatter_stats)

    # 2. Per-trajectory value curves
    print("[ACP-VIZ] 生成 07_trajectory_values.png ...")
    plot_trajectory_values(
        data["traj_data"],
        output_dir / "07_trajectory_values.png",
        num_samples=8,
    )

    # 3. Advantage distribution
    print("[ACP-VIZ] 生成 08_advantage_distribution.png ...")
    adv_stats = plot_advantage_distribution(
        data["advantages"], data["indicators"], data["weights"],
        output_dir / "08_advantage_distribution.png",
    )
    results.update(adv_stats)

    # 4. Success vs fail comparison
    print("[ACP-VIZ] 生成 09_success_vs_fail.png ...")
    sf_stats = plot_success_vs_fail_comparison(
        data["traj_data"],
        output_dir / "09_success_vs_fail.png",
    )
    results.update(sf_stats)

    # 5. Error by timestep
    print("[ACP-VIZ] 生成 10_error_by_timestep.png ...")
    plot_error_by_timestep(
        data["traj_data"],
        output_dir / "10_error_by_timestep.png",
    )

    print(f"[ACP-VIZ] 完成! {len(results)} 项统计, 5 张图表保存至 {output_dir}")
    return results
