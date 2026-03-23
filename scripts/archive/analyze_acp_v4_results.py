#!/usr/bin/env python3
"""ACP v4 RLPD experiment analysis — parse wandb output.log, generate figures + report.

Usage:
    python scripts/analyze_acp_v4_results.py
"""

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Run metadata ──────────────────────────────────────────────────────
RUNS = {
    # v4 experiments
    "AWSC bc=4 (v4)": {
        "output_log": "wandb/run-20260316_141331-jyj63bml/files/output.log",
        "summary": "wandb/run-20260316_141331-jyj63bml/files/wandb-summary.json",
        "algo": "AWSC", "version": "v4", "color": "#2196F3", "linestyle": "-",
        "total_steps": 500_000, "eval_interval": 10_000,
    },
    "AWSC bc=8 (v4)": {
        "output_log": "wandb/run-20260316_141332-gdokl47v/files/output.log",
        "summary": "wandb/run-20260316_141332-gdokl47v/files/wandb-summary.json",
        "algo": "AWSC", "version": "v4", "color": "#FF9800", "linestyle": "-",
        "total_steps": 500_000, "eval_interval": 10_000,
    },
    "PLD γ=0.7 (v4)": {
        "output_log": "wandb/run-20260316_142710-229ntt4w/files/output.log",
        "summary": "wandb/run-20260316_142710-229ntt4w/files/wandb-summary.json",
        "algo": "PLD", "version": "v4", "color": "#4CAF50", "linestyle": "-",
        "total_steps": 71_000, "eval_interval": 10_000,
    },
    "DSRL γ=0.7 (v4)": {
        "output_log": "wandb/run-20260316_142710-gwa4gbtu/files/output.log",
        "summary": "wandb/run-20260316_142710-gwa4gbtu/files/wandb-summary.json",
        "algo": "DSRL", "version": "v4", "color": "#E91E63", "linestyle": "-",
        "total_steps": 71_000, "eval_interval": 10_000,
    },
    # v3 baselines (success_once only)
    "AWSC + v3_so": {
        "output_log": "wandb/run-20260315_120903-7weycepc/files/output.log",
        "summary": "wandb/run-20260315_120903-7weycepc/files/wandb-summary.json",
        "algo": "AWSC", "version": "v3", "color": "#2196F3", "linestyle": "--",
        "total_steps": 500_000, "eval_interval": 10_000,
    },
    "PLD + v3_so": {
        "output_log": "wandb/run-20260315_120903-ynp44qlz/files/output.log",
        "summary": "wandb/run-20260315_120903-ynp44qlz/files/wandb-summary.json",
        "algo": "PLD", "version": "v3", "color": "#4CAF50", "linestyle": "--",
        "total_steps": 71_000, "eval_interval": 10_000,
    },
    "DSRL + v3_so": {
        "output_log": "wandb/run-20260315_120903-m4wgw4ku/files/output.log",
        "summary": "wandb/run-20260315_120903-m4wgw4ku/files/wandb-summary.json",
        "algo": "DSRL", "version": "v3", "color": "#E91E63", "linestyle": "--",
        "total_steps": 71_000, "eval_interval": 10_000,
    },
}

ROOT = Path("/home/wjz/rl-vla")
OUT_DIR = ROOT / "docs/vlaw/figures/rlpd_acp_v4"


def parse_eval_series(log_path: Path, eval_interval: int = 10_000) -> dict:
    """Parse output.log to extract eval time-series.

    Returns dict with keys: steps, success_once, success_at_end, return, reward.
    """
    text = log_path.read_text()

    # Find all eval blocks — each block has 5 lines of metrics
    so_vals = [float(x) for x in re.findall(r"success_once:\s+([\d.]+)", text)]
    sae_vals = [float(x) for x in re.findall(r"success_at_end:\s+([\d.]+)", text)]
    ret_vals = [float(x) for x in re.findall(r"return:\s+([\d.]+)", text)]
    rew_vals = [float(x) for x in re.findall(r"reward:\s+([\d.]+)", text)]

    # Align all series to the shortest length (first entry is pretrain_eval/baseline)
    n = min(len(so_vals), len(sae_vals), len(ret_vals), len(rew_vals))
    if n == 0:
        return {"steps": [], "success_once": [], "success_at_end": [], "return": [], "reward": []}

    # First entry is the pretrain eval (step 0), rest are at eval_interval intervals
    steps = [0] + [eval_interval * i for i in range(1, n)]
    steps = steps[:n]

    return {
        "steps": steps,
        "success_once": so_vals[:n],
        "success_at_end": sae_vals[:n],
        "return": ret_vals[:n],
        "reward": rew_vals[:n],
    }


def parse_critic_from_log(log_path: Path) -> dict:
    """Parse critic stats from output.log (if logged periodically)."""
    text = log_path.read_text()
    # Try to extract Q-mean values from training log lines
    q_means = re.findall(r"q_mean[=:]\s*([\d.e+-]+)", text)
    critic_losses = re.findall(r"critic_loss[=:]\s*([\d.e+-]+)", text)
    return {
        "q_means": [float(x) for x in q_means],
        "critic_losses": [float(x) for x in critic_losses],
    }


def load_all_data() -> dict:
    """Load eval time-series for all runs."""
    data = {}
    for name, meta in RUNS.items():
        log_path = ROOT / meta["output_log"]
        if not log_path.exists():
            print(f"  [SKIP] {name}: {log_path} not found")
            continue
        series = parse_eval_series(log_path, meta["eval_interval"])
        data[name] = {**meta, **series}
        print(f"  [OK] {name}: {len(series['steps'])} eval points, "
              f"best SO={max(series['success_once']):.0%}, "
              f"best SAE={max(series['success_at_end']):.0%}")
    return data


# ── Figure generators ─────────────────────────────────────────────────

def fig1_so_sae_curves(data: dict):
    """Figure 1: Success Once & Success At End learning curves — v4 only."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACP v4 RLPD — Eval Success Curves", fontsize=14, fontweight="bold")

    v4_runs = {k: v for k, v in data.items() if v["version"] == "v4"}

    for name, d in v4_runs.items():
        steps_k = [s / 1000 for s in d["steps"]]
        so_pct = [x * 100 for x in d["success_once"]]
        sae_pct = [x * 100 for x in d["success_at_end"]]
        ax1.plot(steps_k, so_pct, color=d["color"], linestyle=d["linestyle"],
                 marker="o", markersize=3, label=name, linewidth=2)
        ax2.plot(steps_k, sae_pct, color=d["color"], linestyle=d["linestyle"],
                 marker="o", markersize=3, label=name, linewidth=2)

    for ax, title in [(ax1, "Success Once (%)"), (ax2, "Success At End (%)")]:
        ax.set_xlabel("Training Steps (K)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    fig.tight_layout()
    path = OUT_DIR / "fig1_v4_so_sae_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_v3_vs_v4_comparison(data: dict):
    """Figure 2: v3 vs v4 side-by-side comparison per algorithm."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ACP v3 vs v4 — Success Once & Success At End Comparison",
                 fontsize=14, fontweight="bold")

    algos = ["AWSC", "PLD", "DSRL"]
    for col, algo in enumerate(algos):
        # Get v3 and v4 runs for this algo
        v3_runs = {k: v for k, v in data.items()
                   if v["algo"] == algo and v["version"] == "v3"}
        v4_runs = {k: v for k, v in data.items()
                   if v["algo"] == algo and v["version"] == "v4"}

        # Success Once
        ax_so = axes[0, col]
        ax_so.set_title(f"{algo} — Success Once (%)")
        for name, d in {**v3_runs, **v4_runs}.items():
            steps_k = [s / 1000 for s in d["steps"]]
            so_pct = [x * 100 for x in d["success_once"]]
            label = name
            ax_so.plot(steps_k, so_pct, color=d["color"], linestyle=d["linestyle"],
                       marker="o", markersize=3, label=label, linewidth=2)
        ax_so.legend(fontsize=8)
        ax_so.grid(True, alpha=0.3)
        ax_so.set_ylim(-5, 105)
        ax_so.set_xlabel("Steps (K)")

        # Success At End
        ax_sae = axes[1, col]
        ax_sae.set_title(f"{algo} — Success At End (%)")
        for name, d in {**v3_runs, **v4_runs}.items():
            steps_k = [s / 1000 for s in d["steps"]]
            sae_pct = [x * 100 for x in d["success_at_end"]]
            label = name
            ax_sae.plot(steps_k, sae_pct, color=d["color"], linestyle=d["linestyle"],
                        marker="o", markersize=3, label=label, linewidth=2)
        ax_sae.legend(fontsize=8)
        ax_sae.grid(True, alpha=0.3)
        ax_sae.set_ylim(-5, 105)
        ax_sae.set_xlabel("Steps (K)")

    fig.tight_layout()
    path = OUT_DIR / "fig2_v3_vs_v4_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_best_metrics_bar(data: dict):
    """Figure 3: Best SO & SAE bar chart — all runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Best Success Metrics — All ACP Versions", fontsize=14, fontweight="bold")

    # Collect best values
    names = []
    best_so = []
    best_sae = []
    colors = []
    for name, d in data.items():
        if len(d["success_once"]) == 0:
            continue
        names.append(name)
        best_so.append(max(d["success_once"]) * 100)
        best_sae.append(max(d["success_at_end"]) * 100)
        colors.append(d["color"])

    x = np.arange(len(names))
    width = 0.6

    # Best SO
    bars1 = ax1.bar(x, best_so, width, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Best Success Once (%)")
    ax1.set_title("Best Success Once (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, best_so):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Best SAE
    bars2 = ax2.bar(x, best_sae, width, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Best Success At End (%)")
    ax2.set_title("Best Success At End (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, best_sae):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = OUT_DIR / "fig3_best_metrics_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_awsc_comparison(data: dict):
    """Figure 4: AWSC deep dive — bc=4 vs bc=8 on SO & SAE."""
    awsc_runs = {k: v for k, v in data.items()
                 if v["algo"] == "AWSC" and v["version"] == "v4"}
    v3_awsc = {k: v for k, v in data.items()
               if v["algo"] == "AWSC" and v["version"] == "v3"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AWSC Deep Dive — bc_weight Ablation (v4) vs v3 Baseline",
                 fontsize=14, fontweight="bold")

    for name, d in {**v3_awsc, **awsc_runs}.items():
        steps_k = [s / 1000 for s in d["steps"]]
        so_pct = [x * 100 for x in d["success_once"]]
        sae_pct = [x * 100 for x in d["success_at_end"]]
        ax1.plot(steps_k, so_pct, color=d["color"], linestyle=d["linestyle"],
                 marker="o", markersize=3, label=name, linewidth=2)
        ax2.plot(steps_k, sae_pct, color=d["color"], linestyle=d["linestyle"],
                 marker="o", markersize=3, label=name, linewidth=2)

    for ax, title in [(ax1, "Success Once (%)"), (ax2, "Success At End (%)")]:
        ax.set_xlabel("Training Steps (K)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    fig.tight_layout()
    path = OUT_DIR / "fig4_awsc_bc_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_critic_diagnosis(data: dict):
    """Figure 5: Q-value and critic loss comparison from summary data."""
    # Read summary JSONs to get critic stats
    summaries = {}
    for name, meta in RUNS.items():
        if meta["version"] != "v4":
            continue
        summary_path = ROOT / meta["summary"]
        if summary_path.exists():
            with open(summary_path) as f:
                summaries[name] = json.load(f)

    # Also load v3 summaries for comparison
    for name, meta in RUNS.items():
        if meta["version"] != "v3":
            continue
        summary_path = ROOT / meta["summary"]
        if summary_path.exists():
            with open(summary_path) as f:
                summaries[name] = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Critic Diagnosis — v3 vs v4 (Final Snapshot)", fontsize=14, fontweight="bold")

    names = []
    q_means = []
    critic_losses = []
    colors = []
    for name, s in summaries.items():
        q_mean = s.get("train/critic/q_mean", 0)
        c_loss = s.get("train/critic/critic_loss", 0)
        names.append(name)
        q_means.append(q_mean)
        critic_losses.append(c_loss)
        colors.append(RUNS[name]["color"])

    x = np.arange(len(names))
    width = 0.6

    bars1 = ax1.bar(x, q_means, width, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Q Mean")
    ax1.set_title("Final Q Mean")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, q_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, critic_losses, width, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Critic Loss")
    ax2.set_title("Final Critic Loss")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_yscale("log")
    for bar, val in zip(bars2, critic_losses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "fig5_critic_diagnosis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_return_curves(data: dict):
    """Figure 6: Return (cumulative reward) learning curves — v4 only."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle("ACP v4 RLPD — Eval Return Curves", fontsize=14, fontweight="bold")

    v4_runs = {k: v for k, v in data.items() if v["version"] == "v4"}

    for name, d in v4_runs.items():
        steps_k = [s / 1000 for s in d["steps"]]
        ax.plot(steps_k, d["return"], color=d["color"], linestyle=d["linestyle"],
                marker="o", markersize=3, label=name, linewidth=2)

    ax.set_xlabel("Training Steps (K)")
    ax.set_ylabel("Eval Return")
    ax.set_title("Eval Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "fig6_return_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_summary_table(data: dict) -> str:
    """Generate markdown summary table."""
    lines = []
    lines.append("| 实验 | Version | Best SO | Best SAE | Final SO | Final SAE | Steps |")
    lines.append("|------|---------|---------|----------|----------|-----------|-------|")
    for name, d in data.items():
        if len(d["success_once"]) == 0:
            continue
        best_so = max(d["success_once"]) * 100
        best_sae = max(d["success_at_end"]) * 100
        final_so = d["success_once"][-1] * 100
        final_sae = d["success_at_end"][-1] * 100
        last_step = d["steps"][-1] if d["steps"] else 0
        lines.append(f"| {name} | {d['version']} | {best_so:.0f}% | {best_sae:.0f}% | "
                     f"{final_so:.0f}% | {final_sae:.0f}% | {last_step/1000:.0f}K |")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ACP v4 RLPD Experiment Analysis")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    data = load_all_data()

    print(f"\n[2/7] Generating Fig 1: v4 SO/SAE curves...")
    fig1_so_sae_curves(data)

    print(f"[3/7] Generating Fig 2: v3 vs v4 comparison...")
    fig2_v3_vs_v4_comparison(data)

    print(f"[4/7] Generating Fig 3: Best metrics bar chart...")
    fig3_best_metrics_bar(data)

    print(f"[5/7] Generating Fig 4: AWSC bc ablation...")
    fig4_awsc_comparison(data)

    print(f"[6/7] Generating Fig 5: Critic diagnosis...")
    fig5_critic_diagnosis(data)

    print(f"[7/7] Generating Fig 6: Return curves...")
    fig6_return_curves(data)

    print("\n" + "=" * 60)
    print("Summary Table:")
    print("=" * 60)
    print(generate_summary_table(data))

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
