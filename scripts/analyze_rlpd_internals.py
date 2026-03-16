"""ACP v3 RLPD 训练内科分析（Internal Diagnosis）。

从 WandB CSV 数据出发，分析 6 组实验的训练内部指标：
- Critic health: Q-value dynamics, loss convergence, td-target stability
- Actor loss dynamics: flow_loss, shortcut_loss, advantage weighting (AWSC)
- Entropy / temperature evolution (PLD/DSRL)
- Reward signal: ACP vs sim, online vs offline cumulative reward gap
- Cross-metric correlation: SO/SAE vs internal metrics

目标：用训练时的 loss / Q-value / entropy 数据严谨诊断：
1. PLD+v3_sae 灾难性崩溃的根因
2. DSRL/PLD SAE≈0% 的根因
3. AWSC SO 退化的根因

用法：
    PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_rlpd_internals.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/wjz/rl-vla")
DATA_DIR = PROJECT_ROOT / "logs" / "vlaw" / "wandb_analysis" / "rlpd_acp_v3"
FIG_DIR = PROJECT_ROOT / "docs" / "vlaw" / "figures" / "v3_rlpd_internals"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "figure.figsize": (16, 8),
})

RUNS = {
    "AWSC+v3_so":  {"id": "7weycepc", "algo": "awsc", "acp": "v3_so"},
    "AWSC+v3_sae": {"id": "d6wfjs2f", "algo": "awsc", "acp": "v3_sae"},
    "PLD+v3_so":   {"id": "ynp44qlz", "algo": "pld",  "acp": "v3_so"},
    "PLD+v3_sae":  {"id": "4hjfih2f", "algo": "pld",  "acp": "v3_sae"},
    "DSRL+v3_so":  {"id": "m4wgw4ku", "algo": "dsrl", "acp": "v3_so"},
    "DSRL+v3_sae": {"id": "1blrmq2r", "algo": "dsrl", "acp": "v3_sae"},
}

COLORS = {
    "AWSC+v3_so": "#4CAF50", "AWSC+v3_sae": "#FF5722",
    "PLD+v3_so": "#2196F3", "PLD+v3_sae": "#E91E63",
    "DSRL+v3_so": "#FF9800", "DSRL+v3_sae": "#9C27B0",
}


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for name, meta in RUNS.items():
        csv_path = DATA_DIR / f"{meta['id']}_history.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data[name] = df
            print(f"  {name}: {len(df)} rows, {len(df.columns)} cols")
        else:
            print(f"  {name}: CSV not found")
    return data


def safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col in df.columns:
        s = df[col].dropna()
        return s if len(s) > 0 else None
    return None


def smooth(arr, window=5):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode="valid")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Critic Health — Q-value evolution across all 6 experiments
# ═══════════════════════════════════════════════════════════════════════════
def plot_critic_health(data: dict):
    """Fig I-1: Q-value and critic loss evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: q_mean
    # Row 1: critic_loss
    for name, df in data.items():
        meta = RUNS[name]
        color = COLORS[name]

        # Determine q_mean column name
        q_col = "train/critic/q_mean"
        cl_col = "train/critic/critic_loss"
        td_col = "train/critic/td_target_mean"

        q_mean = safe_col(df, q_col)
        critic_loss = safe_col(df, cl_col)
        td_target = safe_col(df, td_col)

        col_idx = {"awsc": 0, "pld": 1, "dsrl": 2}[meta["algo"]]

        if q_mean is not None:
            steps = q_mean.index.values
            vals = q_mean.values
            axes[0, col_idx].plot(steps, vals, color=color, alpha=0.3, linewidth=0.5)
            if len(vals) > 10:
                sm = smooth(vals, 10)
                axes[0, col_idx].plot(steps[4:-5], sm, color=color, linewidth=1.5, label=name)
            else:
                axes[0, col_idx].plot(steps, vals, color=color, linewidth=1.5, label=name)

        if td_target is not None:
            steps = td_target.index.values
            vals = td_target.values
            axes[0, col_idx].plot(steps, vals, color=color, linestyle=":", alpha=0.5, linewidth=1)

        if critic_loss is not None:
            steps = critic_loss.index.values
            vals = critic_loss.values
            axes[1, col_idx].plot(steps, vals, color=color, alpha=0.3, linewidth=0.5)
            if len(vals) > 10:
                sm = smooth(vals, 10)
                axes[1, col_idx].plot(steps[4:-5], sm, color=color, linewidth=1.5, label=name)

    for col_idx, algo in enumerate(["AWSC", "PLD", "DSRL"]):
        axes[0, col_idx].set_title(f"{algo} — Q-value (mean + TD target)")
        axes[0, col_idx].set_ylabel("Q-value")
        axes[0, col_idx].legend()
        axes[0, col_idx].grid(True, alpha=0.3)

        axes[1, col_idx].set_title(f"{algo} — Critic Loss")
        axes[1, col_idx].set_xlabel("Step Index")
        axes[1, col_idx].set_ylabel("Critic Loss")
        axes[1, col_idx].legend()
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_yscale("log")

    fig.suptitle("Fig I-1. Critic Health — Q-value & Loss Dynamics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig1_critic_health.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-1] {FIG_DIR / 'intern_fig1_critic_health.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: PLD v3_so vs v3_sae — deep dive into catastrophic collapse
# ═══════════════════════════════════════════════════════════════════════════
def plot_pld_collapse_diagnosis(data: dict):
    """Fig I-2: PLD catastrophic collapse diagnosis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_map = [
        ("train/critic/q_mean", "Q-value Mean"),
        ("train/actor/actor_entropy", "Actor Entropy"),
        ("train/temp/temperature", "Temperature (alpha)"),
        ("train/critic/critic_loss", "Critic Loss"),
        ("train/actor/actor_loss", "Actor Loss"),
        ("eval/success_once", "success_once (eval)"),
    ]

    for i, (col_name, label) in enumerate(metrics_map):
        ax = axes[i // 3, i % 3]

        for name in ["PLD+v3_so", "PLD+v3_sae"]:
            if name not in data:
                continue
            df = data[name]
            s = safe_col(df, col_name)
            if s is None:
                continue
            steps = s.index.values
            vals = s.values
            color = COLORS[name]

            if "eval" in col_name:
                ax.plot(steps, vals, color=color, linewidth=1.5, marker="o",
                        markersize=4, label=name)
            else:
                ax.plot(steps, vals, color=color, alpha=0.3, linewidth=0.5)
                if len(vals) > 10:
                    sm = smooth(vals, 10)
                    ax.plot(steps[4:-5], sm, color=color, linewidth=1.5, label=name)

        ax.set_title(label)
        ax.set_xlabel("Step Index")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if "loss" in col_name.lower() and "actor" not in col_name.lower():
            ax.set_yscale("log")

    fig.suptitle("Fig I-2. PLD-SAC Collapse Diagnosis — v3_so (stable) vs v3_sae (collapse)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig2_pld_collapse.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-2] {FIG_DIR / 'intern_fig2_pld_collapse.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: AWSC Actor — flow_loss, shortcut_loss, advantage dynamics
# ═══════════════════════════════════════════════════════════════════════════
def plot_awsc_actor_diagnosis(data: dict):
    """Fig I-3: AWSC actor loss decomposition + advantage evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_map = [
        ("train/actor/flow_loss", "Flow BC Loss"),
        ("train/actor/shortcut_loss", "Shortcut Consistency Loss"),
        ("train/actor/advantage_mean", "Advantage Mean"),
        ("train/actor/advantage_std", "Advantage Std"),
        ("train/actor/weight_max", "Weight Max"),
        ("train/critic/q_std", "Critic Q-value Std"),
    ]

    for i, (col_name, label) in enumerate(metrics_map):
        ax = axes[i // 3, i % 3]

        for name in ["AWSC+v3_so", "AWSC+v3_sae"]:
            if name not in data:
                continue
            df = data[name]
            s = safe_col(df, col_name)
            if s is None:
                continue
            steps = s.index.values
            vals = s.values
            color = COLORS[name]

            ax.plot(steps, vals, color=color, alpha=0.3, linewidth=0.5)
            if len(vals) > 20:
                sm = smooth(vals, 20)
                ax.plot(steps[9:-10], sm, color=color, linewidth=1.5, label=name)

        ax.set_title(label)
        ax.set_xlabel("Step Index")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig I-3. AWSC Actor Internals — Loss Decomposition & Advantage Dynamics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig3_awsc_actor.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-3] {FIG_DIR / 'intern_fig3_awsc_actor.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: AWSC SMDP Reward gap — online vs offline cumulative reward
# ═══════════════════════════════════════════════════════════════════════════
def plot_awsc_reward_gap(data: dict):
    """Fig I-4: AWSC online vs offline cumulative reward dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name in ["AWSC+v3_so", "AWSC+v3_sae"]:
        if name not in data:
            continue
        df = data[name]
        color = COLORS[name]

        # Online cumulative reward
        online_rew = safe_col(df, "train/smdp/online_cum_reward_mean")
        offline_rew = safe_col(df, "train/smdp/offline_cum_reward_mean")
        acp_rew = safe_col(df, "train/reward/acp_step_mean")

        if online_rew is not None:
            axes[0].plot(online_rew.index.values, online_rew.values,
                        color=color, linewidth=1.5, label=f"{name} (online)")
        if offline_rew is not None:
            axes[0].plot(offline_rew.index.values, offline_rew.values,
                        color=color, linestyle="--", linewidth=1.5, label=f"{name} (offline)")

        # Reward ratio
        if online_rew is not None and offline_rew is not None:
            # Align indices
            common = online_rew.index.intersection(offline_rew.index)
            if len(common) > 0:
                ratio = online_rew.loc[common].values / (offline_rew.loc[common].values + 1e-8)
                axes[1].plot(common.values, ratio, color=color, linewidth=1.5, label=name)

        if acp_rew is not None:
            axes[2].plot(acp_rew.index.values, acp_rew.values,
                        color=color, alpha=0.3, linewidth=0.5)
            if len(acp_rew) > 20:
                sm = smooth(acp_rew.values, 20)
                axes[2].plot(acp_rew.index.values[9:-10], sm, color=color,
                            linewidth=1.5, label=name)

    axes[0].set_title("(a) Online vs Offline Cumulative Reward")
    axes[0].set_xlabel("Step Index")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("(b) Online/Offline Reward Ratio")
    axes[1].set_xlabel("Step Index")
    axes[1].set_ylabel("Ratio")
    axes[1].axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Parity")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("(c) ACP Step Reward (per-step mean)")
    axes[2].set_xlabel("Step Index")
    axes[2].set_ylabel("ACP Reward")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Fig I-4. AWSC Reward Signal Diagnosis — Online vs Offline Gap",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig4_awsc_reward_gap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-4] {FIG_DIR / 'intern_fig4_awsc_reward_gap.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Entropy & Temperature — PLD/DSRL comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_entropy_temperature(data: dict):
    """Fig I-5: Entropy and temperature evolution for PLD/DSRL."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, algo in enumerate(["PLD", "DSRL"]):
        for name_suffix, color_name in [("v3_so", f"{algo}+v3_so"), ("v3_sae", f"{algo}+v3_sae")]:
            name = f"{algo}+{name_suffix}"
            if name not in data:
                continue
            df = data[name]
            color = COLORS[name]

            # Temperature
            temp = safe_col(df, "train/temp/temperature")
            if temp is not None:
                steps = temp.index.values
                axes[row, 0].plot(steps, temp.values, color=color, linewidth=1.5, label=name)

            # Entropy
            ent = safe_col(df, "train/temp/entropy")
            if ent is not None:
                steps = ent.index.values
                vals = ent.values
                axes[row, 1].plot(steps, vals, color=color, alpha=0.3, linewidth=0.5)
                if len(vals) > 10:
                    sm = smooth(vals, 10)
                    axes[row, 1].plot(steps[4:-5], sm, color=color, linewidth=1.5, label=name)

        axes[row, 0].set_title(f"{algo} — Temperature (alpha)")
        axes[row, 0].set_xlabel("Step Index")
        axes[row, 0].set_ylabel("Temperature")
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].set_title(f"{algo} — Actor Entropy")
        axes[row, 1].set_xlabel("Step Index")
        axes[row, 1].set_ylabel("Entropy")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)

    fig.suptitle("Fig I-5. PLD/DSRL Temperature & Entropy Evolution",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig5_entropy_temp.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-5] {FIG_DIR / 'intern_fig5_entropy_temp.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: AWSC flow_loss vs eval SO/SAE — correlation across training
# ═══════════════════════════════════════════════════════════════════════════
def plot_awsc_loss_vs_eval(data: dict):
    """Fig I-6: AWSC flow_loss trend aligned with eval success metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, name in enumerate(["AWSC+v3_so", "AWSC+v3_sae"]):
        if name not in data:
            continue
        df = data[name]
        color = COLORS[name]

        # Flow loss over time
        flow_loss = safe_col(df, "train/actor/flow_loss")
        so = safe_col(df, "eval/success_once")
        sae = safe_col(df, "eval/success_at_end")

        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]

        if flow_loss is not None:
            steps = flow_loss.index.values
            vals = flow_loss.values
            ax_top.plot(steps, vals, color=color, alpha=0.2, linewidth=0.5)
            if len(vals) > 20:
                sm = smooth(vals, 20)
                ax_top.plot(steps[9:-10], sm, color=color, linewidth=2, label="Flow Loss (MA-20)")
            ax_top.set_ylabel("Flow BC Loss")
            ax_top.set_title(f"{name} — Flow Loss Over Training")
            ax_top.legend()
            ax_top.grid(True, alpha=0.3)

            # Twin axis for SO
            ax_twin = ax_top.twinx()
            if so is not None:
                ax_twin.plot(so.index.values, so.values, color="blue", linewidth=1.5,
                           marker="o", markersize=3, linestyle="--", label="SO", alpha=0.7)
            if sae is not None:
                ax_twin.plot(sae.index.values, sae.values, color="red", linewidth=1.5,
                           marker="s", markersize=3, linestyle="--", label="SAE", alpha=0.7)
            ax_twin.set_ylabel("Success Rate")
            ax_twin.set_ylim(-0.05, 1.05)
            ax_twin.legend(loc="lower right")

        # Shortcut loss
        sc_loss = safe_col(df, "train/actor/shortcut_loss")
        if sc_loss is not None:
            steps = sc_loss.index.values
            vals = sc_loss.values
            ax_bot.plot(steps, vals, color=color, alpha=0.2, linewidth=0.5)
            if len(vals) > 20:
                sm = smooth(vals, 20)
                ax_bot.plot(steps[9:-10], sm, color=color, linewidth=2, label="Shortcut Loss (MA-20)")
            ax_bot.set_ylabel("Shortcut Consistency Loss")
            ax_bot.set_xlabel("Step Index")
            ax_bot.set_title(f"{name} — Shortcut Loss Over Training")
            ax_bot.legend()
            ax_bot.grid(True, alpha=0.3)

    fig.suptitle("Fig I-6. AWSC Loss Decomposition — Flow BC & Shortcut vs Eval Metrics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig6_awsc_loss_eval.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-6] {FIG_DIR / 'intern_fig6_awsc_loss_eval.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Q-value scale comparison across algorithms
# ═══════════════════════════════════════════════════════════════════════════
def plot_q_value_scale(data: dict):
    """Fig I-7: Q-value scale comparison — explains why PLD/DSRL ACP reward gets drowned."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    algo_groups = {"AWSC": [], "PLD": [], "DSRL": []}
    for name in data:
        algo = RUNS[name]["algo"].upper()
        algo_groups[algo].append(name)

    for col_idx, (algo, names) in enumerate(algo_groups.items()):
        ax = axes[col_idx]
        for name in names:
            df = data[name]
            color = COLORS[name]
            q = safe_col(df, "train/critic/q_mean")
            if q is not None:
                ax.plot(q.index.values, q.values, color=color, linewidth=1.5, label=name)

        ax.set_title(f"{algo} — Q-value Scale")
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Q-value Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig I-7. Q-value Scale Comparison — ACP Reward Signal Drowning",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intern_fig7_q_scale.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig I-7] {FIG_DIR / 'intern_fig7_q_scale.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Quantitative Summary
# ═══════════════════════════════════════════════════════════════════════════
def compute_summary(data: dict) -> dict:
    """Compute quantitative summary for all runs."""
    summary = {}
    for name, df in data.items():
        meta = RUNS[name]
        s = {"algo": meta["algo"], "acp": meta["acp"]}

        # Q-value stats
        q = safe_col(df, "train/critic/q_mean")
        if q is not None:
            s["q_mean_avg"] = float(q.mean())
            s["q_mean_final_20pct"] = float(q.iloc[-max(1, len(q)//5):].mean())
            s["q_mean_max"] = float(q.max())
            s["q_mean_min"] = float(q.min())
            s["q_range"] = float(q.max() - q.min())

        # Critic loss
        cl = safe_col(df, "train/critic/critic_loss")
        if cl is not None:
            s["critic_loss_avg"] = float(cl.mean())
            s["critic_loss_final_20pct"] = float(cl.iloc[-max(1, len(cl)//5):].mean())

        # TD target
        td = safe_col(df, "train/critic/td_target_mean")
        if td is not None:
            s["td_target_avg"] = float(td.mean())
            s["td_target_std"] = float(td.std())

        # Actor-specific
        if meta["algo"] == "awsc":
            fl = safe_col(df, "train/actor/flow_loss")
            if fl is not None:
                s["flow_loss_avg"] = float(fl.mean())
                # First vs last 20%
                n = max(1, len(fl) // 5)
                s["flow_loss_first_20pct"] = float(fl.iloc[:n].mean())
                s["flow_loss_last_20pct"] = float(fl.iloc[-n:].mean())
                s["flow_loss_ratio"] = s["flow_loss_last_20pct"] / max(s["flow_loss_first_20pct"], 1e-8)

            adv = safe_col(df, "train/actor/advantage_mean")
            if adv is not None:
                s["advantage_mean_avg"] = float(adv.mean())
                s["advantage_mean_std"] = float(adv.std())

            # SMDP reward gap
            online_rew = safe_col(df, "train/smdp/online_cum_reward_mean")
            offline_rew = safe_col(df, "train/smdp/offline_cum_reward_mean")
            if online_rew is not None and offline_rew is not None:
                s["online_cum_reward_avg"] = float(online_rew.mean())
                s["offline_cum_reward_avg"] = float(offline_rew.mean())
                s["reward_gap_ratio"] = s["offline_cum_reward_avg"] / max(abs(s["online_cum_reward_avg"]), 1e-8)
        else:
            # PLD / DSRL
            temp = safe_col(df, "train/temp/temperature")
            if temp is not None:
                s["temperature_avg"] = float(temp.mean())
                s["temperature_final"] = float(temp.iloc[-1])
                s["temperature_max"] = float(temp.max())

            ent = safe_col(df, "train/temp/entropy")
            if ent is not None:
                s["entropy_avg"] = float(ent.mean())
                s["entropy_final"] = float(ent.iloc[-1])
                s["entropy_min"] = float(ent.min())

            al = safe_col(df, "train/actor/actor_loss")
            if al is not None:
                s["actor_loss_avg"] = float(al.mean())
                s["actor_loss_final_20pct"] = float(al.iloc[-max(1, len(al)//5):].mean())

        summary[name] = s

    return summary


def main():
    print("=" * 70)
    print("[RLPD Internal Diagnosis] Start")
    print("=" * 70)

    print("\n[Step 1] Loading WandB CSV data...")
    data = load_data()
    if not data:
        print("[ERROR] No data found, exiting")
        return

    print(f"\n[Step 2] Generating diagnostic figures ({len(data)} experiments)...")
    plot_critic_health(data)
    plot_pld_collapse_diagnosis(data)
    plot_awsc_actor_diagnosis(data)
    plot_awsc_reward_gap(data)
    plot_entropy_temperature(data)
    plot_awsc_loss_vs_eval(data)
    plot_q_value_scale(data)

    print("\n[Step 3] Computing quantitative summary...")
    summary = compute_summary(data)

    json_path = FIG_DIR / "internals_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")

    # Print key findings
    print("\n" + "=" * 70)
    print("[Key Diagnostic Findings]")
    print("=" * 70)

    # 1. Q-value scale comparison
    print("\n1. Q-value Scale (Average):")
    for name, s in sorted(summary.items()):
        q = s.get("q_mean_avg", None)
        if q is not None:
            print(f"   {name:20s}: Q={q:8.2f}  range=[{s.get('q_mean_min',0):.1f}, {s.get('q_mean_max',0):.1f}]")

    # 2. AWSC reward gap
    print("\n2. AWSC Online/Offline Reward Gap:")
    for name in ["AWSC+v3_so", "AWSC+v3_sae"]:
        s = summary.get(name, {})
        if "offline_cum_reward_avg" in s:
            print(f"   {name}: online={s['online_cum_reward_avg']:.3f}, "
                  f"offline={s['offline_cum_reward_avg']:.3f}, "
                  f"gap_ratio={s['reward_gap_ratio']:.1f}x")

    # 3. PLD temperature divergence
    print("\n3. PLD/DSRL Temperature:")
    for name in ["PLD+v3_so", "PLD+v3_sae", "DSRL+v3_so", "DSRL+v3_sae"]:
        s = summary.get(name, {})
        if "temperature_avg" in s:
            print(f"   {name:20s}: avg={s['temperature_avg']:.4f}, "
                  f"max={s['temperature_max']:.4f}, final={s['temperature_final']:.4f}")

    # 4. AWSC flow loss degradation
    print("\n4. AWSC Flow Loss Degradation:")
    for name in ["AWSC+v3_so", "AWSC+v3_sae"]:
        s = summary.get(name, {})
        if "flow_loss_first_20pct" in s:
            print(f"   {name}: first 20%={s['flow_loss_first_20pct']:.4f}, "
                  f"last 20%={s['flow_loss_last_20pct']:.4f}, "
                  f"ratio={s['flow_loss_ratio']:.2f}x")

    # 5. PLD entropy collapse
    print("\n5. PLD/DSRL Entropy:")
    for name in ["PLD+v3_so", "PLD+v3_sae", "DSRL+v3_so", "DSRL+v3_sae"]:
        s = summary.get(name, {})
        if "entropy_avg" in s:
            print(f"   {name:20s}: avg={s['entropy_avg']:.2f}, "
                  f"min={s['entropy_min']:.2f}, final={s['entropy_final']:.2f}")

    print(f"\n[Done] Figures: {FIG_DIR}")
    print(f"       Summary: {json_path}")


if __name__ == "__main__":
    main()
