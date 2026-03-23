"""ACP v3 RLPD 实验结果分析脚本。

分析 3 算法 × 2 ACP 版本 = 6 组 RLPD 在线实验结果，
对比 sim-reward 基线，生成图表到 docs/vlaw/figures/v3_rlpd/。

用法：
    PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_acp_v3_rlpd_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path("/home/wjz/rl-vla")
FIG_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures" / "v3_rlpd"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (14, 8),
})

# ── Log parsing ──────────────────────────────────────────────────────────
LOG_FILES = {
    "AWSC+v3_so": PROJECT_ROOT / "logs/vlaw/acp_v3_awsc_so_s42.log",
    "AWSC+v3_sae": PROJECT_ROOT / "logs/vlaw/acp_v3_awsc_sae_s42.log",
    "PLD+v3_so": PROJECT_ROOT / "logs/vlaw/acp_v3_pld_so_s42.log",
    "PLD+v3_sae": PROJECT_ROOT / "logs/vlaw/acp_v3_pld_sae_s42.log",
    "DSRL+v3_so": PROJECT_ROOT / "logs/vlaw/acp_v3_dsrl_so_s42.log",
    "DSRL+v3_sae": PROJECT_ROOT / "logs/vlaw/acp_v3_dsrl_sae_s42.log",
}

# Sim-reward baselines (seed 42, from fair_comparison and ACP mirror report)
SIM_BASELINES = {
    "AWSC": {"best_so": 0.92, "best_sae": 0.72},
    "PLD":  {"best_so": 1.00, "best_sae": 0.86},
    "DSRL": {"best_so": 0.92, "best_sae": 0.60},
}

# ACP v2 mirror baselines (from acp_mirror_experiments.md)
ACP_V2_BASELINES = {
    "AWSC": {"best_so": 0.90, "best_sae": 0.66},
    "PLD":  {"best_so": 0.82, "best_sae": 0.02},
    "DSRL": {"best_so": 0.92, "best_sae": 0.06},
}

COLORS = {
    "v3_so": "#4CAF50",   # Green
    "v3_sae": "#FF5722",  # Red-orange
    "sim": "#2196F3",     # Blue
    "v2": "#9E9E9E",      # Gray
}


def parse_log(log_path: Path) -> dict:
    """解析 RLPD 训练日志，提取 eval 指标。"""
    if not log_path.exists():
        return {"steps": [], "so": [], "sae": [], "reward": [], "completed": False}

    steps, so_vals, sae_vals, rewards = [], [], [], []
    completed = False

    text = log_path.read_text()

    # Parse eval blocks: look for patterns like "success_once: 0.8000" and "success_at_end: 0.5000"
    # and "step=XXXXX" or step indicators
    # Format varies by algorithm. Let's look for eval blocks.

    # AWSC format: eval lines after "Evaluation at step XXXXX" or in progress bar
    # Look for "success_once:" and "success_at_end:" pairs

    lines = text.splitlines()
    current_step = None
    current_so = None
    current_sae = None
    current_reward = None

    for line in lines:
        # Detect step from progress bar
        step_match = re.search(r"(\d+)/\d+.*steps=(\d+)", line)
        if step_match:
            current_step = int(step_match.group(1))

        # Detect evaluation step
        eval_step = re.search(r"[Ee]val.*step[= ](\d+)", line)
        if eval_step:
            current_step = int(eval_step.group(1))

        # success_once
        so_match = re.search(r"success_once:\s*([\d.]+)", line)
        if so_match:
            current_so = float(so_match.group(1))

        # success_at_end
        sae_match = re.search(r"success_at_end:\s*([\d.]+)", line)
        if sae_match:
            current_sae = float(sae_match.group(1))

        # reward
        rew_match = re.search(r"(?:eval.*reward|reward_mean):\s*([\d.]+)", line)
        if rew_match:
            current_reward = float(rew_match.group(1))

        # When we have both SO and SAE, record the eval point
        if current_so is not None and current_sae is not None:
            step = current_step if current_step is not None else len(steps) * 10000
            steps.append(step)
            so_vals.append(current_so)
            sae_vals.append(current_sae)
            rewards.append(current_reward if current_reward is not None else 0.0)
            current_so = None
            current_sae = None
            current_reward = None

        if "Training complete" in line or "Done." in line:
            completed = True

    return {
        "steps": steps,
        "so": so_vals,
        "sae": sae_vals,
        "reward": rewards,
        "completed": completed,
    }


def parse_all_logs() -> dict:
    """解析所有日志文件。"""
    results = {}
    for name, path in LOG_FILES.items():
        data = parse_log(path)
        if data["steps"]:
            results[name] = data
            best_so = max(data["so"]) if data["so"] else 0
            best_sae = max(data["sae"]) if data["sae"] else 0
            print(f"  {name}: {len(data['steps'])} evals, best SO={best_so:.0%}, "
                  f"best SAE={best_sae:.0%}, completed={data['completed']}")
        else:
            print(f"  {name}: No evaluation data found")
    return results


# ── Figure 1: Best SO & SAE bar chart ────────────────────────────────────
def plot_best_metrics_bars(results: dict):
    """Fig 1: 各实验 best success_once 和 best success_at_end 柱状图。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    algos = ["AWSC", "PLD", "DSRL"]

    for ax_idx, (metric, metric_label) in enumerate([("so", "Best success_once"), ("sae", "Best success_at_end")]):
        ax = axes[ax_idx]
        x = np.arange(len(algos))
        width = 0.18

        # Sim baseline
        sim_vals = [SIM_BASELINES[a][f"best_{metric}"] for a in algos]
        ax.bar(x - 1.5*width, sim_vals, width, label="Sim reward", color=COLORS["sim"], alpha=0.8)

        # ACP v2 baseline
        v2_vals = [ACP_V2_BASELINES[a][f"best_{metric}"] for a in algos]
        ax.bar(x - 0.5*width, v2_vals, width, label="ACP v2", color=COLORS["v2"], alpha=0.8)

        # ACP v3_so
        v3_so_vals = []
        for a in algos:
            key = f"{a}+v3_so"
            if key in results and results[key][metric]:
                v3_so_vals.append(max(results[key][metric]))
            else:
                v3_so_vals.append(0)
        ax.bar(x + 0.5*width, v3_so_vals, width, label="ACP v3_so", color=COLORS["v3_so"], alpha=0.8)

        # ACP v3_sae
        v3_sae_vals = []
        for a in algos:
            key = f"{a}+v3_sae"
            if key in results and results[key][metric]:
                v3_sae_vals.append(max(results[key][metric]))
            else:
                v3_sae_vals.append(0)
        ax.bar(x + 1.5*width, v3_sae_vals, width, label="ACP v3_sae", color=COLORS["v3_sae"], alpha=0.8)

        # Value labels
        for i, (s, v2, v3s, v3e) in enumerate(zip(sim_vals, v2_vals, v3_so_vals, v3_sae_vals)):
            for offset, val in [(-1.5, s), (-0.5, v2), (0.5, v3s), (1.5, v3e)]:
                if val > 0:
                    ax.text(i + offset*width, val + 0.015, f"{val:.0%}", ha="center", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=11)
        ax.set_ylabel(metric_label)
        ax.set_title(f"(a) {metric_label}" if ax_idx == 0 else f"(b) {metric_label}")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, loc="upper right")
        ax.axhline(y=0.82, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(2.4, 0.83, "pretrained=82%", fontsize=7, color="gray")

    fig.suptitle("Fig 1. ACP v3 RLPD Results — Best Metrics Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rlpd_fig1_best_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] {FIG_DIR / 'rlpd_fig1_best_metrics.png'}")


# ── Figure 2: AWSC training curves ──────────────────────────────────────
def plot_awsc_curves(results: dict):
    """Fig 2: AWSC + v3_so vs v3_sae 训练曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key, label, color, ls in [
        ("AWSC+v3_so", "AWSC+v3_so", COLORS["v3_so"], "-"),
        ("AWSC+v3_sae", "AWSC+v3_sae", COLORS["v3_sae"], "-"),
    ]:
        if key not in results:
            continue
        data = results[key]
        steps_k = [s/1000 for s in data["steps"]]

        axes[0].plot(steps_k, data["so"], color=color, linestyle=ls, linewidth=1.5,
                     marker="o", markersize=2, label=label)
        axes[1].plot(steps_k, data["sae"], color=color, linestyle=ls, linewidth=1.5,
                     marker="o", markersize=2, label=label)
        axes[2].plot(steps_k, data["reward"], color=color, linestyle=ls, linewidth=1.5,
                     marker="o", markersize=2, label=label)

    # Baselines
    axes[0].axhline(SIM_BASELINES["AWSC"]["best_so"], color=COLORS["sim"], linestyle="--",
                    linewidth=1, label=f"Sim best SO={SIM_BASELINES['AWSC']['best_so']:.0%}")
    axes[1].axhline(SIM_BASELINES["AWSC"]["best_sae"], color=COLORS["sim"], linestyle="--",
                    linewidth=1, label=f"Sim best SAE={SIM_BASELINES['AWSC']['best_sae']:.0%}")

    axes[0].axhline(0.82, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    titles = ["(a) success_once", "(b) success_at_end", "(c) Eval Reward"]
    ylabels = ["success_once", "success_at_end", "Reward"]
    for i, ax in enumerate(axes):
        ax.set_xlabel("Steps (K)")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.legend(fontsize=7)
        if i < 2:
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 2. AWSC Training Curves — v3_so vs v3_sae (500K steps)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rlpd_fig2_awsc_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] {FIG_DIR / 'rlpd_fig2_awsc_curves.png'}")


# ── Figure 3: PLD/DSRL training curves ──────────────────────────────────
def plot_pld_dsrl_curves(results: dict):
    """Fig 3: PLD 和 DSRL 训练曲线（71K steps）。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, algo in enumerate(["PLD", "DSRL"]):
        for col, (metric, metric_label) in enumerate([("so", "success_once"), ("sae", "success_at_end")]):
            ax = axes[row, col]

            for key_suffix, label_suffix, color in [
                ("v3_so", "v3_so", COLORS["v3_so"]),
                ("v3_sae", "v3_sae", COLORS["v3_sae"]),
            ]:
                key = f"{algo}+{key_suffix}"
                if key not in results:
                    continue
                data = results[key]
                steps_k = [s/1000 for s in data["steps"]]
                ax.plot(steps_k, data[metric], color=color, linewidth=1.5,
                        marker="o", markersize=3, label=f"{algo}+{label_suffix}")

            # Baselines
            ax.axhline(SIM_BASELINES[algo][f"best_{metric}"], color=COLORS["sim"], linestyle="--",
                        linewidth=1, alpha=0.7, label=f"Sim best={SIM_BASELINES[algo][f'best_{metric}']:.0%}")
            ax.axhline(ACP_V2_BASELINES[algo][f"best_{metric}"], color=COLORS["v2"], linestyle=":",
                        linewidth=1, alpha=0.7, label=f"v2 best={ACP_V2_BASELINES[algo][f'best_{metric}']:.0%}")

            ax.set_xlabel("Steps (K)")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{algo} — {metric_label}")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 3. PLD-SAC & DSRL-SAC Training Curves — v3_so vs v3_sae (71K steps)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rlpd_fig3_pld_dsrl_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] {FIG_DIR / 'rlpd_fig3_pld_dsrl_curves.png'}")


# ── Figure 4: SAE improvement heatmap ────────────────────────────────────
def plot_sae_heatmap(results: dict):
    """Fig 4: success_at_end 改善热力图。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    algos = ["AWSC", "PLD", "DSRL"]
    versions = ["Sim", "ACP v2", "ACP v3_so", "ACP v3_sae"]

    data = np.zeros((len(algos), len(versions)))
    for i, algo in enumerate(algos):
        data[i, 0] = SIM_BASELINES[algo]["best_sae"]
        data[i, 1] = ACP_V2_BASELINES[algo]["best_sae"]
        key_so = f"{algo}+v3_so"
        key_sae = f"{algo}+v3_sae"
        data[i, 2] = max(results[key_so]["sae"]) if key_so in results and results[key_so]["sae"] else 0
        data[i, 3] = max(results[key_sae]["sae"]) if key_sae in results and results[key_sae]["sae"] else 0

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=11)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=11)

    # Value annotations
    for i in range(len(algos)):
        for j in range(len(versions)):
            val = data[i, j]
            color = "white" if val > 0.5 or val < 0.1 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_title("Fig 4. Best success_at_end (%) Comparison", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="success_at_end", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rlpd_fig4_sae_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] {FIG_DIR / 'rlpd_fig4_sae_heatmap.png'}")


# ── Figure 5: SO degradation analysis ────────────────────────────────────
def plot_so_degradation(results: dict):
    """Fig 5: success_once 退化分析（AWSC 长训练）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, label, color in [
        ("AWSC+v3_so", "AWSC+v3_so", COLORS["v3_so"]),
        ("AWSC+v3_sae", "AWSC+v3_sae", COLORS["v3_sae"]),
    ]:
        if key not in results:
            continue
        data = results[key]
        steps_k = [s/1000 for s in data["steps"]]
        so = np.array(data["so"])
        sae = np.array(data["sae"])

        # (a) SO over time with moving average
        ax = axes[0]
        ax.plot(steps_k, so, color=color, alpha=0.3, linewidth=0.8)
        if len(so) >= 5:
            smooth = np.convolve(so, np.ones(5)/5, mode="valid")
            ax.plot(steps_k[2:-2], smooth, color=color, linewidth=2, label=f"{label} (MA-5)")
        ax.axhline(0.82, color="gray", linestyle=":", linewidth=0.8, label="Pretrained=82%")

        # (b) SO vs SAE scatter
        ax = axes[1]
        ax.scatter(so, sae, c=color, alpha=0.5, s=20, label=label, edgecolors="black", linewidths=0.3)

    axes[0].set_xlabel("Steps (K)")
    axes[0].set_ylabel("success_once")
    axes[0].set_title("(a) success_once Degradation Over Training")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("success_once")
    axes[1].set_ylabel("success_at_end")
    axes[1].set_title("(b) SO vs SAE Correlation")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.3, label="SO=SAE")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Fig 5. success_once Degradation — AWSC 500K Steps",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rlpd_fig5_so_degradation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] {FIG_DIR / 'rlpd_fig5_so_degradation.png'}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("[ACP v3 RLPD 结果分析] 开始")
    print("=" * 70)

    print("\n[Step 1] 解析日志...")
    results = parse_all_logs()

    if not results:
        print("[ERROR] 无实验结果，退出")
        return

    print(f"\n[Step 2] 生成图表 ({len(results)} experiments)...")
    plot_best_metrics_bars(results)
    plot_awsc_curves(results)
    plot_pld_dsrl_curves(results)
    plot_sae_heatmap(results)
    plot_so_degradation(results)

    # Save summary JSON
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "best_so": max(data["so"]) if data["so"] else 0,
            "best_sae": max(data["sae"]) if data["sae"] else 0,
            "final_so": data["so"][-1] if data["so"] else 0,
            "final_sae": data["sae"][-1] if data["sae"] else 0,
            "num_evals": len(data["steps"]),
            "completed": data["completed"],
            "total_steps": data["steps"][-1] if data["steps"] else 0,
        }

    json_path = FIG_DIR / "rlpd_results_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Summary] {json_path}")

    print("\n" + "=" * 70)
    print("[ACP v3 RLPD 结果分析] 完成")
    print(f"  图表目录: {FIG_DIR}")
    print(f"  生成图表: rlpd_fig1~fig5")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
