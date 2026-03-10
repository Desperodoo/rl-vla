"""Generate ACP documentation figures from wandb training logs.

Usage:
    python docs/vlaw/gen_acp_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WANDB_DIR = Path("/home/wjz/rl-vla/wandb")
OUT_DIR = Path("/home/wjz/rl-vla/docs/vlaw/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_MAP: dict[str, str] = {
    "v2_demo_only": "run-20260310_010600-met1w4i1",
    "v2_pretrained_pol": "run-20260310_010558-csc0i3u9",
    "v2_teleop_sim": "run-20260310_010600-7n1u0yh3",
    "v2_rl_prior": "run-20260310_010559-gl8fwb3a",
    "v2_combined": "run-20260310_010557-dezvqsjl",
}

COLORS = {
    "v2_demo_only": "#e74c3c",
    "v2_pretrained_pol": "#3498db",
    "v2_teleop_sim": "#2ecc71",
    "v2_rl_prior": "#9b59b6",
    "v2_combined": "#e67e22",
}

LABELS = {
    "v2_demo_only": "A: Demo Only (50 traj)",
    "v2_pretrained_pol": "B: Pretrained Policy (400 traj)",
    "v2_teleop_sim": "C: Teleop Sim / OU (400 traj)",
    "v2_rl_prior": "D: RL Prior / Gauss (400 traj)",
    "v2_combined": "A+B+C+D: Combined (1250 traj)",
}

# Regex patterns for log parsing
TRAIN_PAT = re.compile(
    r"\[ACP\] step=(\d+)/\d+ loss=([\d.]+) mae=([\d.]+) lr=([\d.eE+\-]+)"
)
VAL_PAT = re.compile(r"\[ACP\] \[val\] step=(\d+) loss=([\d.]+) mae=([\d.]+)")
BEST_PAT = re.compile(r"\[ACP\] 新 best MAE=([\d.]+)")


def parse_log(version: str) -> dict:
    """Parse output.log for a given version."""
    run_dir = RUN_MAP[version]
    log_path = WANDB_DIR / run_dir / "files" / "output.log"
    if not log_path.exists():
        print(f"  WARNING: {log_path} not found")
        return {"train": [], "val": [], "best_mae": None}

    train_steps, train_losses, train_maes, lrs = [], [], [], []
    val_steps, val_losses, val_maes = [], [], []
    best_mae = None

    for line in log_path.read_text().splitlines():
        m = TRAIN_PAT.search(line)
        if m:
            train_steps.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            train_maes.append(float(m.group(3)))
            lrs.append(float(m.group(4)))
            continue
        m = VAL_PAT.search(line)
        if m:
            val_steps.append(int(m.group(1)))
            val_losses.append(float(m.group(2)))
            val_maes.append(float(m.group(3)))
            continue
        m = BEST_PAT.search(line)
        if m:
            best_mae = float(m.group(1))

    return {
        "train_steps": np.array(train_steps),
        "train_loss": np.array(train_losses),
        "train_mae": np.array(train_maes),
        "lr": np.array(lrs),
        "val_steps": np.array(val_steps),
        "val_loss": np.array(val_losses),
        "val_mae": np.array(val_maes),
        "best_mae": best_mae,
    }


def plot_val_mae_curves(data: dict[str, dict]) -> None:
    """Fig 1: Validation MAE curves for all 5 versions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for version in ["v2_demo_only", "v2_pretrained_pol", "v2_teleop_sim", "v2_rl_prior", "v2_combined"]:
        d = data[version]
        if len(d["val_steps"]) == 0:
            continue
        best = d["best_mae"]
        label = f"{LABELS[version]}  (best={best:.4f})"
        ax.plot(d["val_steps"], d["val_mae"], color=COLORS[version], label=label, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Validation MAE", fontsize=13)
    ax.set_title("ACP v2: Validation MAE across Data Distributions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_val_mae_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_val_mae_curves.png'}")


def plot_val_loss_curves(data: dict[str, dict]) -> None:
    """Fig 2: Validation loss curves for all 5 versions."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for version in ["v2_demo_only", "v2_pretrained_pol", "v2_teleop_sim", "v2_rl_prior", "v2_combined"]:
        d = data[version]
        if len(d["val_steps"]) == 0:
            continue
        ax.plot(d["val_steps"], d["val_loss"], color=COLORS[version], label=LABELS[version], linewidth=2)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Validation Loss (Distributional CE)", fontsize=13)
    ax.set_title("ACP v2: Validation Loss across Data Distributions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_val_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_val_loss_curves.png'}")


def plot_train_mae_curves(data: dict[str, dict]) -> None:
    """Fig 3: Training MAE curves (smoothed)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for version in ["v2_demo_only", "v2_pretrained_pol", "v2_teleop_sim", "v2_rl_prior", "v2_combined"]:
        d = data[version]
        if len(d["train_steps"]) == 0:
            continue
        # Light raw curve
        ax.plot(d["train_steps"], d["train_mae"], color=COLORS[version], alpha=0.15, linewidth=0.8)
        # Smoothed with moving average
        window = min(20, len(d["train_mae"]))
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(d["train_mae"], kernel, mode="valid")
            steps_smooth = d["train_steps"][window - 1 :]
            ax.plot(steps_smooth, smoothed, color=COLORS[version], label=LABELS[version], linewidth=2)
        else:
            ax.plot(d["train_steps"], d["train_mae"], color=COLORS[version], label=LABELS[version], linewidth=2)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Training MAE (smoothed)", fontsize=13)
    ax.set_title("ACP v2: Training MAE across Data Distributions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_train_mae_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_train_mae_curves.png'}")


def plot_best_mae_bar(data: dict[str, dict]) -> None:
    """Fig 4: Best validation MAE bar chart comparison."""
    versions = ["v2_demo_only", "v2_pretrained_pol", "v2_teleop_sim", "v2_rl_prior", "v2_combined"]
    maes = [data[v]["best_mae"] or 0 for v in versions]
    short_labels = ["A: Demo\nOnly", "B: Pretrained\nPolicy", "C: Teleop\nSim (OU)", "D: RL Prior\n(Gauss)", "A+B+C+D\nCombined"]
    colors = [COLORS[v] for v in versions]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(range(len(versions)), maes, color=colors, edgecolor="black", linewidth=0.8, width=0.65)

    # Add value labels on top of bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.003,
            f"{mae:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylabel("Best Validation MAE", fontsize=13)
    ax.set_title("ACP v2: Best Validation MAE by Data Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add threshold line
    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Quality threshold: MAE < 0.1")
    ax.axhline(y=0.05, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Target: MAE < 0.05")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, max(maes) * 1.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_best_mae_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_best_mae_bar.png'}")


def plot_data_distribution(data: dict[str, dict]) -> None:
    """Fig 5: Training data distribution pie + bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart — trajectory count
    labels = ["A: Demo (50)", "B: Pretrained (400)", "C: Teleop (400)", "D: RL Prior (400)", "E: Random (100)"]
    sizes = [50, 400, 400, 400, 100]
    colors_pie = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#95a5a6"]
    explode = (0.05, 0, 0, 0, 0)
    wedges, texts, autotexts = ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 10},
    )
    ax1.set_title("Trajectory Count by Type", fontsize=13, fontweight="bold")

    # Bar chart — success rate + frame count
    types = ["A: Demo", "B: Pretrained", "C: Teleop", "D: RL Prior", "E: Random"]
    success_rates = [96.0, 30.5, 7.0, 3.5, 0.0]
    frame_counts = [510, 11395, 13040, 13243, 3500]

    x = np.arange(len(types))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, success_rates, width, label="Success Rate (%)", color="#3498db", alpha=0.8)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width / 2, [f / 1000 for f in frame_counts], width, label="Frame Count (K)", color="#e67e22", alpha=0.8)

    ax2.set_xlabel("Data Type", fontsize=12)
    ax2.set_ylabel("Success Rate (%)", fontsize=12, color="#3498db")
    ax2_twin.set_ylabel("Frame Count (K)", fontsize=12, color="#e67e22")
    ax2.set_xticks(x)
    ax2.set_xticklabels(types, fontsize=10, rotation=15)
    ax2.set_title("Data Statistics by Type", fontsize=13, fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_data_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_data_distribution.png'}")


def plot_lr_schedule(data: dict[str, dict]) -> None:
    """Fig 6: Learning rate schedule (from v2_combined, 12000 steps)."""
    d = data["v2_combined"]
    if len(d["train_steps"]) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(d["train_steps"], d["lr"], color="#2c3e50", linewidth=2)
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Learning Rate", fontsize=13)
    ax.set_title("ACP Learning Rate Schedule (warmup 500 + cosine decay)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=500, color="red", linestyle="--", alpha=0.5, label="Warmup end (step 500)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_lr_schedule.png'}")


def plot_value_target_illustration() -> None:
    """Fig 7: Value target illustration for success vs failure trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    T = 35
    c_fail = 35.0
    max_len = 35
    denom = max_len + c_fail  # 70

    frames = np.arange(T)
    remaining = T - frames - 1

    # Success trajectory
    targets_success = np.clip(-remaining / denom, -1, 0)
    # Failure trajectory
    targets_failure = np.clip((-remaining - c_fail) / denom, -1, 0)

    # Left: value target curves
    ax = axes[0]
    ax.plot(frames, targets_success, color="#2ecc71", linewidth=2.5, label="Success trajectory")
    ax.plot(frames, targets_failure, color="#e74c3c", linewidth=2.5, label="Failure trajectory")
    ax.fill_between(frames, targets_failure, targets_success, alpha=0.15, color="#3498db")
    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_ylabel("Value Target V(t)", fontsize=12)
    ax.set_title("Per-Frame Value Targets", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.05, 0.10)
    ax.annotate("Gap = 0.5", xy=(17, -0.5), fontsize=11, color="#3498db",
                ha="center", fontweight="bold")

    # Right: dense rewards derived from targets (exclude last frame which has
    # r[T-1]=target[T-1], a large negative outlier that distorts the scale)
    rewards_succ2 = np.zeros(T)
    for t in range(T - 1):
        rewards_succ2[t] = targets_success[t] - targets_success[t + 1]
    rewards_succ2[T - 1] = targets_success[T - 1]

    rewards_fail2 = np.zeros(T)
    for t in range(T - 1):
        rewards_fail2[t] = targets_failure[t] - targets_failure[t + 1]
    rewards_fail2[T - 1] = targets_failure[T - 1]

    ax2 = axes[1]
    # Plot only t=0..T-2 for clean visualization; annotate last frame separately
    ax2.bar(frames[:-1] - 0.2, rewards_succ2[:-1], width=0.4, color="#2ecc71", alpha=0.7, label="Success r(t)")
    ax2.bar(frames[:-1] + 0.2, rewards_fail2[:-1], width=0.4, color="#e74c3c", alpha=0.7, label="Failure r(t)")
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.set_ylabel("Dense Reward r(t)", fontsize=12)
    ax2.set_title("Dense Rewards (t=0..T-2, excluding boundary)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.annotate(
        f"Note: r[T-1]={rewards_succ2[-1]:.3f} (succ)\n           {rewards_fail2[-1]:.3f} (fail)\nomitted for scale",
        xy=(0.98, 0.02), xycoords="axes fraction", fontsize=9,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_value_target_illustration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_value_target_illustration.png'}")


def plot_architecture_summary() -> None:
    """Fig 8: Model parameter breakdown."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    components = ["SigLIP\n(frozen)", "Gemma\n(frozen)", "Projectors +\nValue Head\n(trainable)"]
    params = [428, 268, 1.55]
    colors = ["#bdc3c7", "#95a5a6", "#e67e22"]
    explode = (0, 0, 0.1)

    wedges, texts, autotexts = ax.pie(
        params,
        explode=explode,
        labels=components,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({pct * sum(params) / 100:.0f}M)",
        startangle=90,
        textprops={"fontsize": 11},
        pctdistance=0.6,
    )
    autotexts[-1].set_fontweight("bold")
    autotexts[-1].set_color("white")
    ax.set_title("Pistar06 Value Model — Parameter Distribution (~697M total)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_model_params.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_model_params.png'}")


def plot_convergence_comparison(data: dict[str, dict]) -> None:
    """Fig 9: Convergence comparison — val MAE with log scale to show demo_only detail."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: linear scale, excluding demo_only
    for version in ["v2_pretrained_pol", "v2_teleop_sim", "v2_rl_prior", "v2_combined"]:
        d = data[version]
        if len(d["val_steps"]) == 0:
            continue
        ax1.plot(d["val_steps"], d["val_mae"], color=COLORS[version], label=LABELS[version], linewidth=2)
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Validation MAE", fontsize=12)
    ax1.set_title("Validation MAE (excl. Demo Only)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.1, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Threshold 0.1")
    ax1.axhline(y=0.05, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Target 0.05")
    ax1.set_ylim(0, 0.25)

    # Right: log scale, all
    for version in ["v2_demo_only", "v2_pretrained_pol", "v2_teleop_sim", "v2_rl_prior", "v2_combined"]:
        d = data[version]
        if len(d["val_steps"]) == 0:
            continue
        ax2.plot(d["val_steps"], d["val_mae"], color=COLORS[version], label=LABELS[version], linewidth=2)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Validation MAE (log scale)", fontsize=12)
    ax2.set_title("Validation MAE (log scale, all versions)", fontsize=13, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "acp_convergence_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'acp_convergence_comparison.png'}")


def main() -> None:
    print("Parsing training logs...")
    data: dict[str, dict] = {}
    for version in RUN_MAP:
        print(f"  Parsing {version}...")
        data[version] = parse_log(version)
        d = data[version]
        print(f"    train: {len(d['train_steps'])} pts, val: {len(d['val_steps'])} pts, best={d['best_mae']}")

    print("\nGenerating figures...")
    plot_val_mae_curves(data)
    plot_val_loss_curves(data)
    plot_train_mae_curves(data)
    plot_best_mae_bar(data)
    plot_data_distribution(data)
    plot_lr_schedule(data)
    plot_value_target_illustration()
    plot_architecture_summary()
    plot_convergence_comparison(data)
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
