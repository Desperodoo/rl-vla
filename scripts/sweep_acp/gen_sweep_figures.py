#!/usr/bin/env python3
"""Generate visualization figures for AWSC+ACP Sweep v2 report.

Usage:
    conda run -n rlft_ms3 python scripts/sweep_acp/gen_sweep_figures.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path("runs/acp_sweep/awsc_acp")
OUT_DIR = Path("logs/vlaw/acp_sweep_v2_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette — visually distinct
COLORS = {
    "baseline": "#2196F3",
    "scale_500": "#FF9800",
    "scale_1000": "#F44336",
    "scale_2000": "#9C27B0",
    "bc_4.0": "#4CAF50",
    "bc_8.0": "#00BCD4",
    "or_0.3": "#FF5722",
    "or_0.5": "#795548",
    "gamma_0.5": "#E91E63",
    "gamma_0.7": "#607D8B",
    "combined_amp_anchor": "#3F51B5",
    "combined_rebalance": "#009688",
    "combined_short_fast": "#CDDC39",
    "combined_max_acp": "#FF4081",
    "combined_balanced": "#00E676",
}

# Group definitions for subplot organization
GROUPS = {
    "scale": ["baseline", "scale_500", "scale_1000", "scale_2000"],
    "bc": ["baseline", "bc_4.0", "bc_8.0"],
    "or": ["baseline", "or_0.3", "or_0.5"],
    "gamma": ["baseline", "gamma_0.5"],
    "combined": [
        "baseline",
        "combined_amp_anchor",
        "combined_rebalance",
        "combined_short_fast",
        "combined_max_acp",
        "combined_balanced",
    ],
}

# Short display names
SHORT_NAMES = {
    "baseline": "baseline\n(s100,bc2,or0.15,γ0.9)",
    "scale_500": "scale=500",
    "scale_1000": "scale=1000",
    "scale_2000": "scale=2000",
    "bc_4.0": "bc=4.0",
    "bc_8.0": "bc=8.0",
    "or_0.3": "or=0.3",
    "or_0.5": "or=0.5",
    "gamma_0.5": "γ=0.5",
    "gamma_0.7": "γ=0.7",
    "combined_amp_anchor": "amp_anchor\n(s1k,bc4)",
    "combined_rebalance": "rebalance\n(s1k,bc4,or0.3)",
    "combined_short_fast": "short_fast\n(s1k,bc4,γ0.7)",
    "combined_max_acp": "max_acp\n(s2k,or0.5,γ0.7)",
    "combined_balanced": "balanced\n(s1k,bc4,or0.3,γ0.7)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_runs() -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Load tensorboard data. Returns {config: {tag: [(step, value), ...]}}."""
    all_data: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for run_dir in sorted(glob.glob(str(BASE_DIR / "*__*"))):
        name = os.path.basename(run_dir)
        config_name = name.split("__")[0]

        tf_files = glob.glob(f"{run_dir}/events.out.tfevents*")
        if not tf_files:
            continue

        ea = EventAccumulator(run_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])

        data: dict[str, list[tuple[int, float]]] = {}
        for tag in tags:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]

        all_data[config_name] = data

    return all_data


def get_series(
    data: dict[str, list[tuple[int, float]]], tag: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract steps and values arrays for a tag."""
    if tag not in data:
        return np.array([]), np.array([])
    pairs = data[tag]
    steps = np.array([p[0] for p in pairs])
    vals = np.array([p[1] for p in pairs])
    return steps, vals


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def apply_style(ax: plt.Axes, title: str, ylabel: str, pct: bool = True) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, loc="best", ncol=1, framealpha=0.8)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))


# ---------------------------------------------------------------------------
# Figure 1: Success rate curves by axis group (4 panels)
# ---------------------------------------------------------------------------
def fig1_success_curves_by_group(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    group_order = ["scale", "bc", "or", "gamma"]
    group_titles = {
        "scale": "A) Scale Axis (acp_reward_scale)",
        "bc": "B) BC Weight Axis (awsc_bc_weight)",
        "or": "C) Online Ratio Axis",
        "gamma": "D) Gamma Axis (discount factor)",
    }

    for idx, gname in enumerate(group_order):
        ax = axes_flat[idx]
        configs = GROUPS[gname]

        for cn in configs:
            if cn not in all_data:
                continue
            steps, vals = get_series(all_data[cn], "eval/success_at_end")
            if len(steps) == 0:
                continue
            label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
            lw = 2.5 if cn == "baseline" else 1.8
            ls = "--" if cn == "baseline" else "-"
            ax.plot(steps, vals, label=label, color=COLORS.get(cn, "gray"),
                    linewidth=lw, linestyle=ls, alpha=0.9)

        # Reference line: sim-reward baseline
        ax.axhline(y=0.72, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.6, label="sim-reward baseline (72%)")

        apply_style(ax, group_titles[gname], "success_at_end")

    fig.suptitle("Fig 1. Success-at-End Curves by Sweep Axis",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig1_sae_curves_by_axis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Combined group success curves (SO + SAE side by side)
# ---------------------------------------------------------------------------
def fig2_combined_group_curves(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    configs = GROUPS["combined"]
    for cn in configs:
        if cn not in all_data:
            continue
        label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
        color = COLORS.get(cn, "gray")
        lw = 2.5 if cn == "baseline" else 1.5
        ls = "--" if cn == "baseline" else "-"

        steps_so, vals_so = get_series(all_data[cn], "eval/success_once")
        steps_sae, vals_sae = get_series(all_data[cn], "eval/success_at_end")

        if len(steps_so) > 0:
            ax1.plot(steps_so, vals_so, label=label, color=color,
                     linewidth=lw, linestyle=ls, alpha=0.9)
        if len(steps_sae) > 0:
            ax2.plot(steps_sae, vals_sae, label=label, color=color,
                     linewidth=lw, linestyle=ls, alpha=0.9)

    ax1.axhline(y=0.80, color="black", linestyle=":", linewidth=1, alpha=0.5,
                label="pretrained (80%)")
    ax2.axhline(y=0.72, color="black", linestyle=":", linewidth=1.2, alpha=0.6,
                label="sim-reward baseline (72%)")

    apply_style(ax1, "success_once (Grasp Ability)", "success_once")
    apply_style(ax2, "success_at_end (Hold Ability)", "success_at_end")

    fig.suptitle("Fig 2. Combined Configs — success_once vs success_at_end",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "fig2_combined_so_sae.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Bar chart — Best SAE, Final SAE, Final Gap
# ---------------------------------------------------------------------------
def fig3_bar_chart(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    configs_ordered = [
        "baseline", "scale_500", "scale_1000", "scale_2000",
        "bc_4.0", "bc_8.0", "or_0.3", "or_0.5", "gamma_0.5",
        "combined_amp_anchor", "combined_rebalance", "combined_short_fast",
        "combined_max_acp", "combined_balanced",
    ]
    configs_present = [c for c in configs_ordered if c in all_data]

    best_sae = []
    final_sae = []
    final_gap = []
    labels = []

    for cn in configs_present:
        _, vals_sae = get_series(all_data[cn], "eval/success_at_end")
        _, vals_so = get_series(all_data[cn], "eval/success_once")
        if len(vals_sae) == 0:
            continue
        labels.append(SHORT_NAMES.get(cn, cn).replace("\n", " "))
        best_sae.append(vals_sae.max())
        final_sae.append(vals_sae[-1])
        final_gap.append(vals_so[-1] - vals_sae[-1] if len(vals_so) > 0 else 0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 7))
    bars1 = ax.bar(x - width, best_sae, width, label="Best SAE",
                   color="#2196F3", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x, final_sae, width, label="Final SAE",
                   color="#4CAF50", alpha=0.85, edgecolor="white")
    bars3 = ax.bar(x + width, final_gap, width, label="Final SO-SAE Gap",
                   color="#FF9800", alpha=0.85, edgecolor="white")

    # Reference line
    ax.axhline(y=0.72, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="sim-reward best SAE (72%)")

    # Value labels on bars
    for bar_set in [bars1, bars2, bars3]:
        for bar in bar_set:
            h = bar.get_height()
            ax.annotate(f"{h:.0%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title("Fig 3. Best SAE / Final SAE / Final Gap — All Configs",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = OUT_DIR / "fig3_bar_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Critic health — Q_mean + critic_loss by axis
# ---------------------------------------------------------------------------
def fig4_critic_health(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top row: Q_mean for scale axis and combined
    for ax, gname, title in [
        (axes[0, 0], "scale", "Q Mean — Scale Axis"),
        (axes[0, 1], "combined", "Q Mean — Combined Configs"),
    ]:
        for cn in GROUPS[gname]:
            if cn not in all_data:
                continue
            steps, vals = get_series(all_data[cn], "train/critic/q_mean")
            if len(steps) == 0:
                continue
            label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
            lw = 2.0 if cn == "baseline" else 1.3
            ls = "--" if cn == "baseline" else "-"
            ax.plot(steps, vals, label=label, color=COLORS.get(cn, "gray"),
                    linewidth=lw, linestyle=ls, alpha=0.8)
        apply_style(ax, title, "Q Mean", pct=False)

    # Bottom row: critic_loss for scale axis and combined
    for ax, gname, title in [
        (axes[1, 0], "scale", "Critic Loss — Scale Axis"),
        (axes[1, 1], "combined", "Critic Loss — Combined Configs"),
    ]:
        for cn in GROUPS[gname]:
            if cn not in all_data:
                continue
            steps, vals = get_series(all_data[cn], "train/critic_loss")
            if len(steps) == 0:
                continue
            label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
            lw = 2.0 if cn == "baseline" else 1.3
            ls = "--" if cn == "baseline" else "-"
            ax.plot(steps, vals, label=label, color=COLORS.get(cn, "gray"),
                    linewidth=lw, linestyle=ls, alpha=0.8)
        apply_style(ax, title, "Critic Loss", pct=False)

    fig.suptitle("Fig 4. Critic Health — Q Values & Loss",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig4_critic_health.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5: SO-SAE Gap evolution for key configs
# ---------------------------------------------------------------------------
def fig5_gap_evolution(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    key_configs = [
        "baseline", "gamma_0.5", "bc_4.0",
        "combined_balanced", "combined_max_acp", "or_0.3",
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    for cn in key_configs:
        if cn not in all_data:
            continue
        steps_so, vals_so = get_series(all_data[cn], "eval/success_once")
        steps_sae, vals_sae = get_series(all_data[cn], "eval/success_at_end")
        if len(vals_so) == 0 or len(vals_sae) == 0:
            continue

        n = min(len(vals_so), len(vals_sae))
        gap = vals_so[:n] - vals_sae[:n]
        steps = steps_so[:n]

        label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
        lw = 2.5 if cn == "baseline" else 1.8
        ls = "--" if cn == "baseline" else "-"
        ax.plot(steps, gap, label=label, color=COLORS.get(cn, "gray"),
                linewidth=lw, linestyle=ls, alpha=0.9)

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
    ax.fill_between([0, 500000], 0, 0.1, color="green", alpha=0.06, label="ideal zone (<10%)")
    apply_style(ax, "Fig 5. SO-SAE Gap Evolution (lower = better hold behavior)",
                "success_once − success_at_end")
    ax.set_ylim(-0.05, 0.85)

    fig.tight_layout()
    out = OUT_DIR / "fig5_gap_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6: Reward signal analysis
# ---------------------------------------------------------------------------
def fig6_reward_signals(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: online_cum_reward_mean for scale axis
    ax = axes[0]
    for cn in GROUPS["scale"]:
        if cn not in all_data:
            continue
        steps, vals = get_series(all_data[cn], "train/smdp/online_cum_reward_mean")
        if len(steps) == 0:
            continue
        label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
        lw = 2.0 if cn == "baseline" else 1.5
        ls = "--" if cn == "baseline" else "-"
        ax.plot(steps, vals, label=label, color=COLORS.get(cn, "gray"),
                linewidth=lw, linestyle=ls, alpha=0.8)
    apply_style(ax, "Online Cumulative Reward — Scale Axis",
                "online_cum_reward_mean", pct=False)

    # Right: advantage_mean for all single-axis configs
    ax = axes[1]
    single_axis = ["baseline", "scale_1000", "bc_4.0", "or_0.3", "gamma_0.5"]
    for cn in single_axis:
        if cn not in all_data:
            continue
        steps, vals = get_series(all_data[cn], "train/actor/advantage_mean")
        if len(steps) == 0:
            continue
        label = SHORT_NAMES.get(cn, cn).replace("\n", " ")
        lw = 2.0 if cn == "baseline" else 1.5
        ls = "--" if cn == "baseline" else "-"
        # Smooth for readability
        if len(vals) > 10:
            sm = smooth(vals, 5)
            sm_steps = steps[:len(sm)]
            ax.plot(sm_steps, sm, label=label, color=COLORS.get(cn, "gray"),
                    linewidth=lw, linestyle=ls, alpha=0.8)
        else:
            ax.plot(steps, vals, label=label, color=COLORS.get(cn, "gray"),
                    linewidth=lw, linestyle=ls, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
    apply_style(ax, "Advantage Mean — Key Configs",
                "advantage_mean", pct=False)

    fig.suptitle("Fig 6. Reward Signal & Advantage Analysis",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "fig6_reward_advantage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 7: Scatter — Best SAE vs Final SO (Pareto frontier)
# ---------------------------------------------------------------------------
def fig7_pareto_scatter(
    all_data: dict[str, dict[str, list[tuple[int, float]]]]
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))

    for cn, data in all_data.items():
        _, vals_so = get_series(data, "eval/success_once")
        _, vals_sae = get_series(data, "eval/success_at_end")
        if len(vals_so) == 0 or len(vals_sae) == 0:
            continue

        best_sae = vals_sae.max()
        final_so = vals_so[-1]

        color = COLORS.get(cn, "gray")
        label = SHORT_NAMES.get(cn, cn).replace("\n", " ")

        ax.scatter(final_so, best_sae, s=120, c=color, edgecolors="black",
                   linewidth=0.8, zorder=5)
        ax.annotate(label, (final_so, best_sae), fontsize=6.5,
                    textcoords="offset points", xytext=(5, 5),
                    ha="left", va="bottom")

    # Reference point: sim reward
    ax.scatter([], [], s=1, label="configs")  # dummy for legend
    ax.axhline(y=0.72, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="sim-reward best SAE (72%)")
    ax.axvline(x=0.80, color="blue", linestyle="--", linewidth=1, alpha=0.5,
               label="pretrained SO (80%)")

    # Ideal zone
    ax.fill_between([0.80, 1.0], 0.72, 1.0, color="green", alpha=0.06,
                    label="target zone")

    ax.set_xlabel("Final success_once", fontsize=11)
    ax.set_ylabel("Best success_at_end", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(0.50, 0.95)
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Fig 7. Pareto Frontier — Final SO vs Best SAE",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = OUT_DIR / "fig7_pareto_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading tensorboard data...")
    all_data = load_all_runs()
    print(f"  Loaded {len(all_data)} configs: {sorted(all_data.keys())}")

    generators = [
        ("Fig 1: SAE curves by axis", fig1_success_curves_by_group),
        ("Fig 2: Combined SO vs SAE", fig2_combined_group_curves),
        ("Fig 3: Bar comparison", fig3_bar_chart),
        ("Fig 4: Critic health", fig4_critic_health),
        ("Fig 5: Gap evolution", fig5_gap_evolution),
        ("Fig 6: Reward & advantage", fig6_reward_signals),
        ("Fig 7: Pareto scatter", fig7_pareto_scatter),
    ]

    for desc, func in generators:
        print(f"  Generating {desc}...", end=" ", flush=True)
        path = func(all_data)
        print(f"-> {path}")

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
