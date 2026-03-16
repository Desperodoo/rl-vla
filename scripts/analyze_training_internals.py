"""Generalized RLPD/ACP training internals diagnosis.

Analyzes ANY set of RLPD/ACP experiments from WandB data with a five-dimension
diagnostic framework: Critic health, Actor drift, Exploration, Reward signal,
and Advantage weighting.

Auto-detects algorithm type (AWSC/PLD/DSRL) per run from available metrics,
generates algorithm-appropriate diagnostic figures, and produces a graded
markdown report with prescriptions.

Usage:
    # Analyze from pre-downloaded CSV data
    PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_training_internals.py \
        --data_dir logs/vlaw/wandb_analysis/rlpd_acp_v3

    # Fetch from WandB and analyze
    http_proxy=http://10.20.93.149:7890 https_proxy=http://10.20.93.149:7890 \
    PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_training_internals.py \
        --project rlpd-acp-v4

    # Analyze specific runs
    PYTHONPATH=/home/wjz/rl-vla python scripts/analyze_training_internals.py \
        --project rlpd-acp-v4 --run_ids abc123,def456
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tyro
except ImportError:
    tyro = None

PROJECT_ROOT = Path("/home/wjz/rl-vla")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
    "figure.figsize": (16, 8),
})

# Color palette for up to 12 runs
PALETTE = [
    "#4CAF50", "#FF5722", "#2196F3", "#E91E63",
    "#FF9800", "#9C27B0", "#00BCD4", "#795548",
    "#607D8B", "#8BC34A", "#3F51B5", "#FFEB3B",
]


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionScore:
    grade: str  # A/B/C/D/F
    score: float  # 0-100
    findings: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass
class RunInfo:
    df: pd.DataFrame
    algo: str  # awsc, pld, dsrl, unknown
    config: dict
    run_id: str
    color: str = "#666666"


# ═══════════════════════════════════════════════════════════════════════════
# CLI Args
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Args:
    """Training internals diagnosis arguments."""
    project: str = ""
    """WandB project name."""
    run_ids: str = ""
    """Comma-separated WandB run IDs. If empty, fetch all runs from project."""
    data_dir: str = ""
    """Directory with pre-downloaded CSV files ({run_id}_history.csv).
    If empty, will fetch from WandB and save to auto-generated dir."""
    output_dir: str = ""
    """Output directory for figures and report. Auto-generated if empty."""
    fetch_wandb: bool = True
    """Whether to fetch data from WandB. Set False with --data_dir for offline use."""


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col in df.columns:
        s = df[col].dropna()
        return s if len(s) > 0 else None
    return None


def smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def score_to_grade(score: float) -> str:
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    return "F"


def detect_algorithm(df: pd.DataFrame) -> str:
    """Detect algorithm type from available metric columns."""
    cols = set(df.columns)
    if "train/actor/flow_loss" in cols or "train/actor/shortcut_loss" in cols:
        return "awsc"
    if "probe/total_probe_steps" in cols:
        return "pld"
    if "train/temp/temperature" in cols or "train/actor/actor_entropy" in cols:
        return "dsrl"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def fetch_wandb_data(project: str, run_ids: list[str] | None, output_dir: Path) -> None:
    """Fetch WandB data using fetch_wandb.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "sweep_acp" / "fetch_wandb.py"),
        "--project", project,
        "--save_csv",
        "--output_dir", str(output_dir),
    ]
    if run_ids:
        cmd.extend(["--run_ids", ",".join(run_ids)])

    print(f"[Fetch] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def load_data(data_dir: Path) -> dict[str, RunInfo]:
    """Load pre-downloaded CSV data and auto-detect algorithm per run."""
    runs: dict[str, RunInfo] = {}
    csv_files = sorted(data_dir.glob("*_history.csv"))

    if not csv_files:
        print(f"[WARN] No CSV files found in {data_dir}")
        return runs

    for idx, csv_path in enumerate(csv_files):
        run_id = csv_path.stem.replace("_history", "")
        df = pd.read_csv(csv_path)
        algo = detect_algorithm(df)

        config_path = data_dir / f"{run_id}_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        run_name = config.get("exp_name", run_id)
        color = PALETTE[idx % len(PALETTE)]

        runs[run_name] = RunInfo(df=df, algo=algo, config=config, run_id=run_id, color=color)
        print(f"  {run_name}: {len(df)} rows, algo={algo}")

    return runs


# ═══════════════════════════════════════════════════════════════════════════
# Five-Dimension Auto-Grading
# ═══════════════════════════════════════════════════════════════════════════

def grade_critic(df: pd.DataFrame, algo: str) -> DimensionScore:
    """Dimension 1: Critic health — Q-value stability, loss convergence, TD target."""
    findings: list[str] = []
    evidence: dict = {}
    score = 100.0

    q = safe_col(df, "train/critic/q_mean")
    if q is not None:
        q_range = float(q.max() - q.min())
        q_avg = float(q.mean())
        evidence["q_mean_avg"] = round(q_avg, 2)
        evidence["q_range"] = round(q_range, 1)
        if q_range > 50:
            score -= 40
            findings.append(f"Q-value range {q_range:.1f} >> 50: critic oscillating severely")
        elif q_range > 10:
            score -= 20
            findings.append(f"Q-value range {q_range:.1f} > 10: mild instability")

    cl = safe_col(df, "train/critic/critic_loss")
    if cl is not None:
        n = max(1, len(cl) // 5)
        final_cl = float(cl.iloc[-n:].mean())
        evidence["critic_loss_final"] = round(final_cl, 2)
        if final_cl > 50:
            score -= 30
            findings.append(f"Critic loss final 20%={final_cl:.1f}: not converging")
        elif final_cl > 1.0:
            score -= 15
            findings.append(f"Critic loss final 20%={final_cl:.2f}: slow convergence")

    td = safe_col(df, "train/critic/td_target_mean")
    if td is not None:
        td_std = float(td.std())
        evidence["td_target_std"] = round(td_std, 3)
        if td_std > 10:
            score -= 20
            findings.append(f"TD target std={td_std:.1f}: value estimation unstable")
        elif td_std > 1.0:
            score -= 10
            findings.append(f"TD target std={td_std:.2f}: moderate instability")

    return DimensionScore(grade=score_to_grade(score), score=score, findings=findings, evidence=evidence)


def grade_actor(df: pd.DataFrame, algo: str) -> DimensionScore:
    """Dimension 2: Actor drift — policy degradation detection."""
    findings: list[str] = []
    evidence: dict = {}
    score = 100.0

    if algo == "awsc":
        fl = safe_col(df, "train/actor/flow_loss")
        so = safe_col(df, "eval/success_once")
        if fl is not None:
            n = max(1, len(fl) // 5)
            first = float(fl.iloc[:n].mean())
            last = float(fl.iloc[-n:].mean())
            ratio = last / max(first, 1e-8)
            evidence["flow_loss_first"] = round(first, 4)
            evidence["flow_loss_last"] = round(last, 4)
            evidence["flow_loss_ratio"] = round(ratio, 3)

            if ratio < 0.3 and so is not None:
                so_arr = so.values
                if len(so_arr) >= 4:
                    first_so = float(np.mean(so_arr[:len(so_arr) // 4]))
                    last_so = float(np.mean(so_arr[-len(so_arr) // 4:]))
                    if last_so < first_so * 0.8:
                        score -= 40
                        findings.append(
                            f"Flow loss ↓{(1-ratio)*100:.0f}% but SO declined "
                            f"{first_so:.2%}→{last_so:.2%}: overfitting demo"
                        )
                        evidence["so_decline"] = f"{first_so:.2%}→{last_so:.2%}"
    else:
        ent = safe_col(df, "train/temp/entropy")
        if ent is None:
            ent = safe_col(df, "train/actor/actor_entropy")
        if ent is not None:
            ent_min = float(ent.min())
            ent_final = float(ent.iloc[-1])
            evidence["entropy_min"] = round(ent_min, 2)
            evidence["entropy_final"] = round(ent_final, 2)
            if ent_min < -50:
                score -= 30
                findings.append(f"Entropy min={ent_min:.0f}: policy collapsed at some point")
            if ent_final < -10:
                score -= 15
                findings.append(f"Entropy final={ent_final:.1f}: exploration severely limited")

    return DimensionScore(grade=score_to_grade(score), score=score, findings=findings, evidence=evidence)


def grade_exploration(df: pd.DataFrame, algo: str) -> DimensionScore:
    """Dimension 3: Exploration health — temperature and entropy."""
    findings: list[str] = []
    evidence: dict = {}
    score = 100.0

    if algo == "awsc":
        # AWSC has no temperature/entropy; grade as N/A
        return DimensionScore(grade="N/A", score=100, findings=["N/A (AWSC has no SAC entropy)"], evidence={})

    temp = safe_col(df, "train/temp/temperature")
    if temp is not None:
        t_avg = float(temp.mean())
        t_final = float(temp.iloc[-1])
        evidence["temperature_avg"] = round(t_avg, 4)
        evidence["temperature_final"] = round(t_final, 4)
        if t_final < 0.05:
            score -= 25
            findings.append(f"Temperature final={t_final:.4f}: exploration over-compressed")
        elif t_final > 1.0:
            score -= 15
            findings.append(f"Temperature final={t_final:.2f}: entropy far from target")

    ent = safe_col(df, "train/temp/entropy")
    if ent is not None:
        e_min = float(ent.min())
        evidence["entropy_min"] = round(e_min, 2)
        if e_min < -50:
            score -= 30
            findings.append(f"Entropy min={e_min:.0f}: historical policy collapse")

    return DimensionScore(grade=score_to_grade(score), score=score, findings=findings, evidence=evidence)


def grade_reward(df: pd.DataFrame, algo: str) -> DimensionScore:
    """Dimension 4: Reward signal — online/offline gap, ACP signal strength."""
    findings: list[str] = []
    evidence: dict = {}
    score = 100.0

    if algo == "awsc":
        online = safe_col(df, "train/smdp/online_cum_reward_mean")
        offline = safe_col(df, "train/smdp/offline_cum_reward_mean")
        if online is not None and offline is not None:
            on_avg = float(online.mean())
            off_avg = float(offline.mean())
            gap = off_avg / max(abs(on_avg), 1e-8)
            evidence["online_cum_reward_avg"] = round(on_avg, 4)
            evidence["offline_cum_reward_avg"] = round(off_avg, 4)
            evidence["reward_gap_ratio"] = round(gap, 1)
            if gap > 100:
                score -= 40
                findings.append(f"Online/offline reward gap {gap:.0f}x: critic dominated by offline")
            elif gap > 10:
                score -= 20
                findings.append(f"Online/offline reward gap {gap:.0f}x: significant imbalance")

        acp = safe_col(df, "train/reward/acp_step_mean")
        if acp is not None:
            acp_avg = float(acp.mean())
            evidence["acp_step_mean"] = round(acp_avg, 4)
            if abs(acp_avg) < 0.001:
                score -= 15
                findings.append(f"ACP step reward avg={acp_avg:.4f}: signal nearly dead")
    else:
        # For PLD/DSRL, infer reward signal from Q-value scale
        q = safe_col(df, "train/critic/q_mean")
        if q is not None:
            q_avg = float(q.mean())
            evidence["q_mean_avg"] = round(q_avg, 2)
            if q_avg > 30:
                score -= 20
                findings.append(f"Q-value avg={q_avg:.1f}: Q-value inflated (likely high gamma compounding ACP rewards)")

    return DimensionScore(grade=score_to_grade(score), score=score, findings=findings, evidence=evidence)


def grade_advantage(df: pd.DataFrame, algo: str) -> DimensionScore:
    """Dimension 5: Advantage weighting (AWSC only)."""
    findings: list[str] = []
    evidence: dict = {}
    score = 100.0

    if algo != "awsc":
        return DimensionScore(grade="N/A", score=100, findings=["N/A (non-AWSC algorithm)"], evidence={})

    adv = safe_col(df, "train/actor/advantage_mean")
    if adv is not None:
        adv_avg = float(adv.mean())
        evidence["advantage_mean_avg"] = round(adv_avg, 3)
        if abs(adv_avg) > 1.0:
            score -= 30
            findings.append(f"Advantage mean={adv_avg:.2f}: critic unable to discriminate good/bad actions")
        elif abs(adv_avg) > 0.5:
            score -= 15
            findings.append(f"Advantage mean={adv_avg:.2f}: moderate positive bias")

    wm = safe_col(df, "train/actor/weight_max")
    if wm is not None:
        wm_max = float(wm.max())
        evidence["weight_max_peak"] = round(wm_max, 1)
        if wm_max > 20:
            score -= 20
            findings.append(f"Weight max peak={wm_max:.0f}: few samples over-amplified")
        elif wm_max > 5:
            score -= 10
            findings.append(f"Weight max peak={wm_max:.1f}: moderate amplification")

    return DimensionScore(grade=score_to_grade(score), score=score, findings=findings, evidence=evidence)


def diagnose_all(runs: dict[str, RunInfo]) -> dict[str, dict[str, DimensionScore]]:
    """Run five-dimension diagnosis on all runs."""
    scores: dict[str, dict[str, DimensionScore]] = {}
    for name, info in runs.items():
        scores[name] = {
            "critic": grade_critic(info.df, info.algo),
            "actor": grade_actor(info.df, info.algo),
            "exploration": grade_exploration(info.df, info.algo),
            "reward": grade_reward(info.df, info.algo),
            "advantage": grade_advantage(info.df, info.algo),
        }
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════════════════════════

def plot_critic_health(runs: dict[str, RunInfo], fig_dir: Path) -> str | None:
    """Critic Q-value and loss per algorithm group."""
    algo_groups: dict[str, list[tuple[str, RunInfo]]] = {}
    for name, info in runs.items():
        algo_groups.setdefault(info.algo.upper(), []).append((name, info))

    n_algos = len(algo_groups)
    if n_algos == 0:
        return None

    fig, axes = plt.subplots(2, max(n_algos, 1), figsize=(6 * max(n_algos, 1), 10), squeeze=False)

    for col_idx, (algo, items) in enumerate(sorted(algo_groups.items())):
        for name, info in items:
            q = safe_col(info.df, "train/critic/q_mean")
            cl = safe_col(info.df, "train/critic/critic_loss")

            if q is not None:
                vals = q.values
                axes[0, col_idx].plot(q.index.values, vals, color=info.color, alpha=0.3, linewidth=0.5)
                if len(vals) > 10:
                    sm = smooth(vals, 10)
                    axes[0, col_idx].plot(q.index.values[4:-5], sm, color=info.color, linewidth=1.5, label=name)

            if cl is not None:
                vals = cl.values
                axes[1, col_idx].plot(cl.index.values, vals, color=info.color, alpha=0.3, linewidth=0.5)
                if len(vals) > 10:
                    sm = smooth(vals, 10)
                    axes[1, col_idx].plot(cl.index.values[4:-5], sm, color=info.color, linewidth=1.5, label=name)

        axes[0, col_idx].set_title(f"{algo} — Q-value (mean)")
        axes[0, col_idx].set_ylabel("Q-value")
        axes[0, col_idx].legend()
        axes[0, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_title(f"{algo} — Critic Loss")
        axes[1, col_idx].set_xlabel("Step Index")
        axes[1, col_idx].set_ylabel("Critic Loss")
        axes[1, col_idx].legend()
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_yscale("log")

    fig.suptitle("Critic Health — Q-value & Loss Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "fig_critic_health.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {out}")
    return str(out)


def plot_q_value_scale(runs: dict[str, RunInfo], fig_dir: Path) -> str | None:
    """Cross-algorithm Q-value scale comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    any_data = False
    for name, info in runs.items():
        q = safe_col(info.df, "train/critic/q_mean")
        if q is not None:
            ax.plot(q.index.values, q.values, color=info.color, linewidth=1.5, label=f"{name} ({info.algo})")
            any_data = True

    if not any_data:
        plt.close(fig)
        return None

    ax.set_title("Q-value Scale Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Q-value Mean")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "fig_q_scale.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {out}")
    return str(out)


def plot_awsc_actor(awsc_runs: dict[str, RunInfo], fig_dir: Path) -> str | None:
    """AWSC actor loss decomposition + advantage dynamics."""
    metrics = [
        ("train/actor/flow_loss", "Flow BC Loss"),
        ("train/actor/shortcut_loss", "Shortcut Consistency Loss"),
        ("train/actor/advantage_mean", "Advantage Mean"),
        ("train/actor/advantage_std", "Advantage Std"),
        ("train/actor/weight_max", "Weight Max"),
        ("train/critic/q_std", "Critic Q-value Std"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (col_name, label) in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        for name, info in awsc_runs.items():
            s = safe_col(info.df, col_name)
            if s is None:
                continue
            vals = s.values
            ax.plot(s.index.values, vals, color=info.color, alpha=0.3, linewidth=0.5)
            if len(vals) > 20:
                sm = smooth(vals, 20)
                ax.plot(s.index.values[9:-10], sm, color=info.color, linewidth=1.5, label=name)
        ax.set_title(label)
        ax.set_xlabel("Step Index")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("AWSC Actor Internals — Loss & Advantage Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "fig_awsc_actor.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {out}")
    return str(out)


def plot_awsc_reward_gap(awsc_runs: dict[str, RunInfo], fig_dir: Path) -> str | None:
    """AWSC online vs offline cumulative reward dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, info in awsc_runs.items():
        online = safe_col(info.df, "train/smdp/online_cum_reward_mean")
        offline = safe_col(info.df, "train/smdp/offline_cum_reward_mean")
        acp = safe_col(info.df, "train/reward/acp_step_mean")

        if online is not None:
            axes[0].plot(online.index.values, online.values, color=info.color, linewidth=1.5, label=f"{name} (online)")
        if offline is not None:
            axes[0].plot(offline.index.values, offline.values, color=info.color, linestyle="--", linewidth=1.5, label=f"{name} (offline)")

        if online is not None and offline is not None:
            common = online.index.intersection(offline.index)
            if len(common) > 0:
                ratio = online.loc[common].values / (offline.loc[common].values + 1e-8)
                axes[1].plot(common.values, ratio, color=info.color, linewidth=1.5, label=name)

        if acp is not None:
            vals = acp.values
            axes[2].plot(acp.index.values, vals, color=info.color, alpha=0.3, linewidth=0.5)
            if len(vals) > 20:
                sm = smooth(vals, 20)
                axes[2].plot(acp.index.values[9:-10], sm, color=info.color, linewidth=1.5, label=name)

    axes[0].set_title("Online vs Offline Cumulative Reward")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Online/Offline Reward Ratio")
    axes[1].axhline(1.0, color="red", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title("ACP Step Reward (per-step)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("AWSC Reward Signal Diagnosis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "fig_awsc_reward_gap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {out}")
    return str(out)


def plot_awsc_loss_vs_eval(awsc_runs: dict[str, RunInfo], fig_dir: Path) -> str | None:
    """AWSC flow_loss aligned with eval success metrics (overfitting check)."""
    n = len(awsc_runs)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, max(n, 1), figsize=(7 * max(n, 1), 5), squeeze=False)

    for idx, (name, info) in enumerate(awsc_runs.items()):
        ax = axes[0, idx]
        fl = safe_col(info.df, "train/actor/flow_loss")
        so = safe_col(info.df, "eval/success_once")
        sae = safe_col(info.df, "eval/success_at_end")

        if fl is not None:
            vals = fl.values
            ax.plot(fl.index.values, vals, color=info.color, alpha=0.2, linewidth=0.5)
            if len(vals) > 20:
                sm = smooth(vals, 20)
                ax.plot(fl.index.values[9:-10], sm, color=info.color, linewidth=2, label="Flow Loss")
            ax.set_ylabel("Flow BC Loss")
            ax.set_title(f"{name}")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            ax2 = ax.twinx()
            if so is not None:
                ax2.plot(so.index.values, so.values, color="blue", linewidth=1.5, marker="o", markersize=3,
                         linestyle="--", label="SO", alpha=0.7)
            if sae is not None:
                ax2.plot(sae.index.values, sae.values, color="red", linewidth=1.5, marker="s", markersize=3,
                         linestyle="--", label="SAE", alpha=0.7)
            ax2.set_ylabel("Success Rate")
            ax2.set_ylim(-0.05, 1.05)
            ax2.legend(loc="lower right")

    fig.suptitle("AWSC Flow Loss vs Eval — Overfitting Detection", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "fig_awsc_loss_eval.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {out}")
    return str(out)


def plot_entropy_temperature(sac_runs: dict[str, RunInfo], fig_dir: Path) -> str | None:
    """PLD/DSRL temperature and entropy evolution."""
    algo_groups: dict[str, list[tuple[str, RunInfo]]] = {}
    for name, info in sac_runs.items():
        algo_groups.setdefault(info.algo.upper(), []).append((name, info))

    n = len(algo_groups)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

    for row, (algo, items) in enumerate(sorted(algo_groups.items())):
        for name, info in items:
            temp = safe_col(info.df, "train/temp/temperature")
            if temp is not None:
                axes[row, 0].plot(temp.index.values, temp.values, color=info.color, linewidth=1.5, label=name)

            ent = safe_col(info.df, "train/temp/entropy")
            if ent is not None:
                vals = ent.values
                axes[row, 1].plot(ent.index.values, vals, color=info.color, alpha=0.3, linewidth=0.5)
                if len(vals) > 10:
                    sm = smooth(vals, 10)
                    axes[row, 1].plot(ent.index.values[4:-5], sm, color=info.color, linewidth=1.5, label=name)

        axes[row, 0].set_title(f"{algo} — Temperature")
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].set_title(f"{algo} — Entropy")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)

    fig.suptitle("PLD/DSRL Temperature & Entropy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "fig_entropy_temp.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {out}")
    return str(out)


def generate_figures(runs: dict[str, RunInfo], fig_dir: Path) -> list[str]:
    """Generate diagnostic figures based on available algorithms."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str | None] = []

    generated.append(plot_critic_health(runs, fig_dir))
    generated.append(plot_q_value_scale(runs, fig_dir))

    awsc_runs = {k: v for k, v in runs.items() if v.algo == "awsc"}
    if awsc_runs:
        generated.append(plot_awsc_actor(awsc_runs, fig_dir))
        generated.append(plot_awsc_reward_gap(awsc_runs, fig_dir))
        generated.append(plot_awsc_loss_vs_eval(awsc_runs, fig_dir))

    sac_runs = {k: v for k, v in runs.items() if v.algo in ("pld", "dsrl")}
    if sac_runs:
        generated.append(plot_entropy_temperature(sac_runs, fig_dir))

    return [p for p in generated if p is not None]


# ═══════════════════════════════════════════════════════════════════════════
# Prescription Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_prescriptions(scores: dict[str, dict[str, DimensionScore]]) -> list[str]:
    """Auto-generate prescriptions based on diagnostic findings."""
    prescriptions: list[str] = []
    seen: set[str] = set()

    for name, dims in scores.items():
        # Critic issues
        if dims["critic"].score < 55:
            key = "lower_gamma"
            if key not in seen:
                seen.add(key)
                prescriptions.append("**Lower gamma**: Reduce discount factor to shrink Q-value scale and improve critic stability.")

        # Actor overfitting (AWSC)
        for f in dims["actor"].findings:
            if "overfitting" in f.lower():
                key = "increase_bc_weight"
                if key not in seen:
                    seen.add(key)
                    prescriptions.append("**Increase BC weight**: Raise awsc_bc_weight (e.g., 4-8) to resist policy drift from pretrained distribution.")
                key = "early_stop"
                if key not in seen:
                    seen.add(key)
                    prescriptions.append("**Enable early stopping**: Use --early_stop to halt training when SO degrades while flow_loss drops.")

        # Reward signal issues
        if dims["reward"].score < 70:
            for f in dims["reward"].findings:
                if "gap" in f.lower():
                    key = "increase_scale"
                    if key not in seen:
                        seen.add(key)
                        prescriptions.append("**Increase ACP reward scale**: Raise --acp_reward_scale (e.g., 500-2000) to strengthen online ACP signal relative to offline demo reward.")
                if "inflated" in f.lower() or "drowned" in f.lower():
                    key = "lower_gamma"
                    if key not in seen:
                        seen.add(key)
                        prescriptions.append("**Lower gamma**: Reduce discount factor (e.g., 0.7) to shrink Q-value scale and improve critic stability.")

        # Advantage issues (AWSC)
        if dims["advantage"].score < 70:
            key = "increase_online_ratio"
            if key not in seen:
                seen.add(key)
                prescriptions.append("**Increase online_ratio**: Raise --online_ratio (e.g., 0.3-0.5) to give critic more diverse training data.")

    if not prescriptions:
        prescriptions.append("No critical issues detected. Consider running longer or with different seeds for confirmation.")

    return prescriptions


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def overall_grade(dims: dict[str, DimensionScore]) -> str:
    """Compute overall grade from dimension grades."""
    grade_vals = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0, "N/A": -1}
    valid = [(d, s) for d, s in dims.items() if s.grade != "N/A"]
    if not valid:
        return "N/A"
    avg = sum(grade_vals[s.grade] for _, s in valid) / len(valid)
    if avg >= 3.5:
        return "A"
    elif avg >= 2.5:
        return "B"
    elif avg >= 1.5:
        return "C"
    elif avg >= 0.5:
        return "D"
    return "F"


def generate_report(
    runs: dict[str, RunInfo],
    scores: dict[str, dict[str, DimensionScore]],
    fig_dir: Path,
    report_path: Path,
) -> None:
    """Generate markdown diagnostic report."""
    lines = [
        "# Training Internals Diagnosis Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Experiments: {len(runs)}",
        f"> Algorithms: {', '.join(sorted(set(info.algo for info in runs.values())))}",
        "",
        "---",
        "",
        "## Five-Dimension Scorecard",
        "",
        "| Experiment | Algo | Critic | Actor | Exploration | Reward | Advantage | Overall |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name in scores:
        dims = scores[name]
        algo = runs[name].algo.upper()
        grades = [dims[d].grade for d in ["critic", "actor", "exploration", "reward", "advantage"]]
        og = overall_grade(dims)
        lines.append(f"| {name} | {algo} | {' | '.join(grades)} | **{og}** |")

    lines.extend(["", "---", "", "## Detailed Findings", ""])

    for name, dims in scores.items():
        algo = runs[name].algo.upper()
        lines.append(f"### {name} ({algo})")
        lines.append("")
        for dim_name in ["critic", "actor", "exploration", "reward", "advantage"]:
            ds = dims[dim_name]
            if ds.grade == "N/A":
                continue
            lines.append(f"**{dim_name.title()}** ({ds.grade}, score={ds.score:.0f}):")
            if ds.findings:
                for f in ds.findings:
                    lines.append(f"- {f}")
            else:
                lines.append("- Healthy")
            if ds.evidence:
                ev_str = ", ".join(f"{k}={v}" for k, v in ds.evidence.items())
                lines.append(f"- Evidence: {ev_str}")
            lines.append("")

    # Prescriptions
    lines.extend(["---", "", "## Auto-Generated Prescriptions", ""])
    prescriptions = generate_prescriptions(scores)
    for p in prescriptions:
        lines.append(f"- {p}")

    lines.extend(["", "---", "", f"*Report generated by `scripts/analyze_training_internals.py`*"])

    report_path.write_text("\n".join(lines))
    print(f"  [Report] {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if tyro is not None:
        args = tyro.cli(Args)
    else:
        # Fallback for environments without tyro
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--project", default="")
        parser.add_argument("--run_ids", default="")
        parser.add_argument("--data_dir", default="")
        parser.add_argument("--output_dir", default="")
        parser.add_argument("--fetch_wandb", action="store_true", default=True)
        parser.add_argument("--no_fetch_wandb", dest="fetch_wandb", action="store_false")
        parsed = parser.parse_args()
        args = Args(**vars(parsed))

    print("=" * 70)
    print("[Training Internals Diagnosis] Start")
    print("=" * 70)

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.project:
        data_dir = PROJECT_ROOT / "logs" / "vlaw" / "wandb_analysis" / args.project
    else:
        print("[ERROR] Must provide either --project or --data_dir")
        sys.exit(1)

    # Fetch WandB data if needed
    if args.fetch_wandb and args.project and not list(data_dir.glob("*_history.csv")):
        run_id_list = [r.strip() for r in args.run_ids.split(",") if r.strip()] if args.run_ids else None
        fetch_wandb_data(args.project, run_id_list, data_dir)

    # Load data
    print(f"\n[Step 1] Loading data from {data_dir}")
    runs = load_data(data_dir)
    if not runs:
        print("[ERROR] No data loaded, exiting")
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        name = args.project or data_dir.name
        out_dir = PROJECT_ROOT / "docs" / "vlaw" / "figures" / f"{name}_internals"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diagnose
    print(f"\n[Step 2] Running five-dimension diagnosis ({len(runs)} experiments)")
    scores = diagnose_all(runs)

    # Print scorecard
    print("\n  Scorecard:")
    for name, dims in scores.items():
        grades = " ".join(f"{d[0].upper()}:{dims[d].grade}" for d in ["critic", "actor", "exploration", "reward", "advantage"])
        og = overall_grade(dims)
        print(f"    {name:40s} {grades}  Overall: {og}")

    # Generate figures
    print(f"\n[Step 3] Generating figures → {out_dir}")
    figs = generate_figures(runs, out_dir)
    print(f"  Generated {len(figs)} figures")

    # Generate report
    print(f"\n[Step 4] Generating report")
    report_path = out_dir / "diagnosis_report.md"
    generate_report(runs, scores, out_dir, report_path)

    # Dump summary JSON
    summary: dict = {}
    for name, dims in scores.items():
        summary[name] = {
            "algo": runs[name].algo,
            "overall": overall_grade(dims),
            "dimensions": {
                d: {"grade": s.grade, "score": s.score, "evidence": s.evidence}
                for d, s in dims.items()
            },
        }
    json_path = out_dir / "diagnosis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  [JSON] {json_path}")

    print(f"\n{'='*70}")
    print(f"[Done] Report: {report_path}")
    print(f"       Figures: {out_dir}")
    print(f"       Summary: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
