#!/usr/bin/env python3
"""
Data-Efficiency Fair Comparison — Multi-Seed Analysis Tool
==========================================================

Compares AWSC (RLPD) vs PLD-SAC vs DSRL-SAC across multiple seeds.

Features:
1. Discover all {algo}/best_s{seed}__* experiment directories
2. Extract training curves from TensorBoard event files
3. Normalize x-axis across algorithms (scheme A/B/C)
4. Plot mean ± std curves with shaded bands
5. Box-plots for final performance comparison
6. Summary statistics table (mean, std, min, max, median)
7. Markdown report with all figures

Usage:
    python analyze_sweep.py --sweep-dir runs/fair_comparison
    python analyze_sweep.py --sweep-dir runs/fair_comparison --output-dir results/analysis
"""

from __future__ import annotations

import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

# Optional imports
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tensorboard.backend.event_processing import event_accumulator

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from scipy import stats as sp_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =============================================================================
# Constants
# =============================================================================

ALGO_DISPLAY: dict[str, dict] = {
    "awsc": {"label": "AWSC (RLPD)", "color": "#2196F3", "marker": "o"},
    "pld":  {"label": "PLD-SAC",     "color": "#4CAF50", "marker": "s"},
    "dsrl": {"label": "DSRL-SAC",    "color": "#FF5722", "marker": "^"},
}

METRICS_OF_INTEREST = [
    "eval/success_once",
    "eval/success_at_end",
    "eval/episode_len",
    "charts/critic_loss",
]

# Plot style
PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Single experiment result (one seed)."""

    algorithm: str          # awsc, pld, dsrl
    config_name: str        # e.g. "best_s42"
    base_config: str = ""   # e.g. "best" (without seed suffix)
    seed: int = 0
    success_once: float | None = None
    success_at_end: float | None = None
    status: str = "unknown"
    exp_dir: str = ""
    # Training curves: tag -> list of (step, value)
    training_curves: dict[str, list[tuple[int, float]]] = field(default_factory=dict)


# =============================================================================
# Data Extraction
# =============================================================================

class DataExtractor:
    """Discover and extract multi-seed experiment data."""

    def __init__(self, sweep_dir: str, act_horizon: int = 7):
        self.sweep_dir = Path(sweep_dir)
        self.act_horizon = act_horizon

    # ── Discovery ──────────────────────────────────────────────────────────

    def find_all_experiments(self) -> list[ExperimentResult]:
        """Walk sweep_dir/{algo}/ and discover all seed experiments."""
        results: list[ExperimentResult] = []
        for algo in ALGO_DISPLAY:
            algo_dir = self.sweep_dir / algo
            if not algo_dir.exists():
                continue
            for exp_dir in self._find_exp_dirs(algo_dir):
                result = self._parse_experiment(algo, exp_dir)
                if result is not None:
                    results.append(result)
        return results

    def _find_exp_dirs(self, algo_dir: Path) -> list[Path]:
        """Find all experiment dirs under an algorithm directory.

        Handles both:
          - best_s42__1234567890/   (multi-seed new format)
          - best__1234567890/       (single-seed legacy)
        """
        dirs: list[Path] = []
        for item in sorted(algo_dir.iterdir()):
            if not item.is_dir():
                continue
            # Valid if has train.log or checkpoints/
            if (item / "train.log").exists() or (item / "checkpoints").exists():
                dirs.append(item)
        return dirs

    def _parse_experiment(self, algorithm: str, exp_dir: Path) -> ExperimentResult | None:
        """Parse algorithm, config_name, seed from directory and logs."""
        dir_name = exp_dir.name

        # Strip timestamp suffix: "best_s42__1770390417" → "best_s42"
        config_name = dir_name.rsplit("__", 1)[0] if "__" in dir_name else dir_name

        # Extract seed from config_name: "best_s42" → seed=42, base="best"
        seed_match = re.search(r"_s(\d+)$", config_name)
        if seed_match:
            seed = int(seed_match.group(1))
            base_config = config_name[: seed_match.start()]
        else:
            # Legacy single-seed experiments — try to detect seed from log
            seed = self._detect_seed_from_log(exp_dir)
            base_config = config_name

        result = ExperimentResult(
            algorithm=algorithm,
            config_name=config_name,
            base_config=base_config,
            seed=seed,
            exp_dir=str(exp_dir),
        )

        # Parse final metrics from log
        log_file = exp_dir / "train.log"
        if log_file.exists():
            self._parse_log_metrics(log_file, result)

        # Parse training curves from TensorBoard
        if HAS_TENSORBOARD:
            self._parse_tensorboard(exp_dir, result)

        # Determine status
        ckpt_dir = exp_dir / "checkpoints"
        if ckpt_dir.exists() and any(ckpt_dir.glob("*.pt")):
            result.status = "success"
        elif log_file.exists():
            text = log_file.read_text(errors="ignore")
            if any(err in text for err in ("CUDA error", "OOM", "out of memory")):
                result.status = "failed"
            elif any(
                k in text
                for k in ("Training complete", "Done.", "Saving final checkpoint")
            ):
                result.status = "success"
            else:
                result.status = "running"
        else:
            result.status = "not_started"

        return result

    def _detect_seed_from_log(self, exp_dir: Path) -> int:
        """Fallback: extract seed from train.log (wandb config or CLI args)."""
        log_file = exp_dir / "train.log"
        if not log_file.exists():
            return 0
        try:
            text = log_file.read_text(errors="ignore")[:20000]  # first 20KB
            # Pattern: "seed: 42" or "seed=42" or "--seed 42"
            m = re.search(r"(?:seed[=:\s]+)(\d+)", text)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return 0

    # ── Metric Parsing ─────────────────────────────────────────────────────

    def _parse_log_metrics(self, log_file: Path, result: ExperimentResult) -> None:
        """Parse final metrics from training log."""
        text = log_file.read_text(errors="ignore")

        # "Done. Best success rate: XX.XX%"  (PLD/DSRL)
        m = re.search(r"Best success rate: ([\d.]+)", text)
        if m:
            result.success_once = float(m.group(1)) / 100.0

        # "Training complete! Best success rate: XX.XX%"  (AWSC)
        if result.success_once is None:
            m = re.search(r"Training complete.*Best success rate: ([\d.]+)", text)
            if m:
                result.success_once = float(m.group(1)) / 100.0

        # wandb summary — eval/success_once
        for line in reversed(text.splitlines()):
            if "wandb:" not in line:
                continue
            if "eval/success_once" in line:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        val = float(parts[-1])
                        if result.success_once is None:
                            result.success_once = val
                    except ValueError:
                        pass
            if "eval/success_at_end" in line and result.success_at_end is None:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        result.success_at_end = float(parts[-1])
                    except ValueError:
                        pass
            # Early stop once both found
            if result.success_once is not None and result.success_at_end is not None:
                break

    def _parse_tensorboard(self, exp_dir: Path, result: ExperimentResult) -> None:
        """Extract training curves from TensorBoard event files."""
        event_files = list(exp_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            return
        try:
            ea = event_accumulator.EventAccumulator(
                str(exp_dir),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            ea.Reload()
            available_tags = ea.Tags().get("scalars", [])
            for tag in METRICS_OF_INTEREST:
                if tag in available_tags:
                    events = ea.Scalars(tag)
                    result.training_curves[tag] = [
                        (e.step, e.value) for e in events
                    ]
        except Exception:
            pass

    # ── X-Axis Normalization ───────────────────────────────────────────────

    def normalize_steps(
        self, results: list[ExperimentResult], scheme: str = "A"
    ) -> list[ExperimentResult]:
        """Normalize step counts based on comparison scheme.

        A: All → robot steps (PLD/DSRL steps × act_horizon)
        B: All → chunk decisions (AWSC steps ÷ act_horizon)
        C: No change
        """
        scheme = scheme.upper()
        for result in results:
            for tag in list(result.training_curves.keys()):
                curve = result.training_curves[tag]
                if scheme == "A" and result.algorithm in ("pld", "dsrl"):
                    result.training_curves[tag] = [
                        (s * self.act_horizon, v) for s, v in curve
                    ]
                elif scheme == "B" and result.algorithm == "awsc":
                    result.training_curves[tag] = [
                        (s // self.act_horizon, v) for s, v in curve
                    ]
        return results


# =============================================================================
# Aggregation Helpers
# =============================================================================

def group_by_algorithm(
    results: list[ExperimentResult],
) -> dict[str, list[ExperimentResult]]:
    """Group experiments by algorithm (only successful ones)."""
    groups: dict[str, list[ExperimentResult]] = {}
    for r in results:
        if r.status != "success":
            continue
        groups.setdefault(r.algorithm, []).append(r)
    return groups


def aggregate_curves(
    experiments: list[ExperimentResult],
    tag: str,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate training curves across seeds.

    Returns (steps, mean, std, n_seeds).
    Interpolates all curves onto a common set of steps.
    """
    curves = [e.training_curves.get(tag, []) for e in experiments]
    curves = [c for c in curves if len(c) > 0]
    n_seeds = len(curves)
    if n_seeds == 0:
        return np.array([]), np.array([]), np.array([]), 0

    # Find common step range
    min_step = max(c[0][0] for c in curves)
    max_step = min(c[-1][0] for c in curves)
    if min_step >= max_step:
        max_step = max(c[-1][0] for c in curves)

    common_steps = np.linspace(min_step, max_step, num=num_points)

    matrix = np.full((n_seeds, num_points), np.nan)
    for i, curve in enumerate(curves):
        s_arr = np.array([s for s, _ in curve], dtype=float)
        v_arr = np.array([v for _, v in curve], dtype=float)
        matrix[i] = np.interp(common_steps, s_arr, v_arr)

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return common_steps, mean, std, n_seeds


def compute_summary_stats(
    groups: dict[str, list[ExperimentResult]],
) -> dict[str, dict]:
    """Compute per-algorithm summary statistics."""
    summary: dict[str, dict] = {}
    for algo, exps in groups.items():
        s_once = np.array(
            [e.success_once for e in exps if e.success_once is not None]
        )
        s_end = np.array(
            [e.success_at_end for e in exps if e.success_at_end is not None]
        )
        summary[algo] = {
            "n_seeds": len(exps),
            "seeds": sorted(e.seed for e in exps),
            "success_once": _stats_dict(s_once),
            "success_at_end": _stats_dict(s_end),
        }
    return summary


def _stats_dict(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


# =============================================================================
# Visualization
# =============================================================================

def _apply_style() -> None:
    if HAS_MATPLOTLIB:
        plt.rcParams.update(PLT_STYLE)


def _human_steps(x: float, _pos: int | None = None) -> str:
    """Format step numbers for axis ticks: 100K, 1.5M, etc."""
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def plot_mean_std_curves(
    groups: dict[str, list[ExperimentResult]],
    output_dir: Path,
    scheme: str = "A",
) -> list[str]:
    """Plot mean ± std training curves for each metric."""
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available, skipping curve plots.")
        return []

    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    x_label = {
        "A": "Robot Steps",
        "B": "Chunk Decisions",
        "C": "Training Steps (raw)",
    }.get(scheme.upper(), "Steps")

    for tag in METRICS_OF_INTEREST:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for algo_key in ALGO_DISPLAY:
            exps = groups.get(algo_key, [])
            if not exps:
                continue

            meta = ALGO_DISPLAY[algo_key]
            steps, mean, std, n = aggregate_curves(exps, tag)
            if n == 0:
                continue

            has_data = True
            label = f"{meta['label']} (n={n})"

            ax.plot(
                steps,
                mean,
                label=label,
                color=meta["color"],
                marker=meta["marker"],
                markersize=4,
                markevery=max(1, len(steps) // 12),
                linewidth=2,
                zorder=3,
            )
            if n > 1:
                ax.fill_between(
                    steps,
                    mean - std,
                    mean + std,
                    alpha=0.2,
                    color=meta["color"],
                    zorder=2,
                )

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel(x_label)
        ax.set_ylabel(tag.replace("/", " / "))
        ax.set_title(f"Fair Comparison — {tag}  (mean ± std)")
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(FuncFormatter(_human_steps))

        safe_name = tag.replace("/", "_")
        fname = f"curve_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        saved.append(fname)
        print(f"  Saved: {fname}")

    return saved


def plot_box_whisker(
    groups: dict[str, list[ExperimentResult]],
    output_dir: Path,
) -> list[str]:
    """Box-and-whisker plot for final success_once and success_at_end."""
    if not HAS_MATPLOTLIB:
        return []

    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for metric_name, attr in [
        ("Best Success Rate (success_once)", "success_once"),
        ("Final Success Rate (success_at_end)", "success_at_end"),
    ]:
        data: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []

        for algo_key in ALGO_DISPLAY:
            exps = groups.get(algo_key, [])
            vals = np.array([getattr(e, attr) for e in exps if getattr(e, attr) is not None])
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(ALGO_DISPLAY[algo_key]["label"])
            colors.append(ALGO_DISPLAY[algo_key]["color"])

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)

        # Overlay individual points
        for i, (vals, c) in enumerate(zip(data, colors)):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full_like(vals, i + 1) + jitter,
                vals,
                color=c,
                edgecolors="black",
                linewidth=0.5,
                s=40,
                zorder=5,
                alpha=0.8,
            )

        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}  —  distribution across seeds")

        safe = attr.replace("/", "_")
        fname = f"boxplot_{safe}.png"
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        saved.append(fname)
        print(f"  Saved: {fname}")

    return saved


def plot_per_seed_bars(
    groups: dict[str, list[ExperimentResult]],
    output_dir: Path,
) -> list[str]:
    """Grouped bar chart: success_once per seed for each algorithm."""
    if not HAS_MATPLOTLIB:
        return []

    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all seeds
    all_seeds = sorted({e.seed for exps in groups.values() for e in exps})
    if not all_seeds:
        return []

    fig, ax = plt.subplots(figsize=(max(10, len(all_seeds) * 1.2), 5))
    x = np.arange(len(all_seeds))
    n_algos = len(ALGO_DISPLAY)
    width = 0.8 / n_algos

    for i, algo_key in enumerate(ALGO_DISPLAY):
        exps = groups.get(algo_key, [])
        seed_map = {e.seed: e.success_once for e in exps if e.success_once is not None}
        vals = [seed_map.get(s, 0.0) for s in all_seeds]
        meta = ALGO_DISPLAY[algo_key]
        ax.bar(
            x + i * width,
            vals,
            width=width,
            label=meta["label"],
            color=meta["color"],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels([f"s{s}" for s in all_seeds], rotation=45, ha="right")
    ax.set_ylabel("Best Success Rate")
    ax.set_title("success_once per Seed")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fname = "bar_per_seed.png"
    fig.tight_layout()
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")
    return [fname]


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    groups: dict[str, list[ExperimentResult]],
    summary: dict[str, dict],
    output_dir: Path,
    scheme: str = "A",
    act_horizon: int = 7,
    plot_files: list[str] | None = None,
) -> None:
    """Generate a comprehensive markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"

    lines = [
        "# Data-Efficiency Fair Comparison Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Comparison Scheme**: {scheme}",
        f"- **Act Horizon**: {act_horizon}",
        f"- **X-Axis**: {'Robot Steps' if scheme == 'A' else 'Chunk Decisions' if scheme == 'B' else 'Raw Steps'}",
        "",
    ]

    # ── Summary Table ──────────────────────────────────────────────────────
    lines += [
        "## Summary Statistics",
        "",
        "### Best Success Rate (success_once)",
        "",
        "| Algorithm | N seeds | Mean | Std | Min | Max | Median |",
        "|-----------|:-------:|:----:|:---:|:---:|:---:|:------:|",
    ]
    for algo_key in ALGO_DISPLAY:
        st = summary.get(algo_key, {}).get("success_once", {})
        n = summary.get(algo_key, {}).get("n_seeds", 0)
        if st.get("mean") is not None:
            lines.append(
                f"| {ALGO_DISPLAY[algo_key]['label']} | {n} "
                f"| {st['mean']:.4f} | {st['std']:.4f} "
                f"| {st['min']:.4f} | {st['max']:.4f} "
                f"| {st['median']:.4f} |"
            )
        else:
            lines.append(f"| {ALGO_DISPLAY[algo_key]['label']} | 0 | - | - | - | - | - |")

    lines += [
        "",
        "### Final Success Rate (success_at_end)",
        "",
        "| Algorithm | N seeds | Mean | Std | Min | Max | Median |",
        "|-----------|:-------:|:----:|:---:|:---:|:---:|:------:|",
    ]
    for algo_key in ALGO_DISPLAY:
        st = summary.get(algo_key, {}).get("success_at_end", {})
        n = summary.get(algo_key, {}).get("n_seeds", 0)
        if st.get("mean") is not None:
            lines.append(
                f"| {ALGO_DISPLAY[algo_key]['label']} | {n} "
                f"| {st['mean']:.4f} | {st['std']:.4f} "
                f"| {st['min']:.4f} | {st['max']:.4f} "
                f"| {st['median']:.4f} |"
            )
        else:
            lines.append(f"| {ALGO_DISPLAY[algo_key]['label']} | 0 | - | - | - | - | - |")

    # ── Per-seed detail table ──────────────────────────────────────────────
    lines += [
        "",
        "## Per-Seed Results",
        "",
        "| Seed | AWSC once | AWSC end | PLD once | PLD end | DSRL once | DSRL end |",
        "|:----:|:---------:|:--------:|:--------:|:-------:|:---------:|:--------:|",
    ]
    all_seeds = sorted(
        {e.seed for exps in groups.values() for e in exps}
    )
    for s in all_seeds:
        row = [f"| {s} "]
        for algo_key in ["awsc", "pld", "dsrl"]:
            exps = [e for e in groups.get(algo_key, []) if e.seed == s]
            if exps:
                e = exps[0]
                so = f"{e.success_once:.2%}" if e.success_once is not None else "-"
                se = f"{e.success_at_end:.2%}" if e.success_at_end is not None else "-"
                row.append(f"| {so} | {se} ")
            else:
                row.append("| - | - ")
        lines.append("".join(row) + "|")

    # ── Statistical Tests ──────────────────────────────────────────────────
    if HAS_SCIPY and len(groups) >= 2:
        lines += ["", "## Pairwise Statistical Tests (Welch's t-test)", ""]
        algo_keys = [k for k in ALGO_DISPLAY if k in groups]
        lines.append("| Pair | t-statistic | p-value | Significant (p<0.05)? |")
        lines.append("|------|:-----------:|:-------:|:---------------------:|")
        for i in range(len(algo_keys)):
            for j in range(i + 1, len(algo_keys)):
                a, b = algo_keys[i], algo_keys[j]
                va = np.array([e.success_once for e in groups[a] if e.success_once is not None])
                vb = np.array([e.success_once for e in groups[b] if e.success_once is not None])
                if len(va) >= 2 and len(vb) >= 2:
                    t_stat, p_val = sp_stats.ttest_ind(va, vb, equal_var=False)
                    sig = "Yes" if p_val < 0.05 else "No"
                    la = ALGO_DISPLAY[a]["label"]
                    lb = ALGO_DISPLAY[b]["label"]
                    lines.append(
                        f"| {la} vs {lb} | {t_stat:.4f} | {p_val:.4f} | {sig} |"
                    )

    # ── Plots ──────────────────────────────────────────────────────────────
    if plot_files:
        lines += ["", "## Plots", ""]
        for f in plot_files:
            lines.append(f"![{f}]({f})")
            lines.append("")

    # ── Scheme Notes ────────────────────────────────────────────────────────
    lines += ["", "## Comparison Scheme Notes", ""]
    if scheme.upper() == "A":
        lines += [
            "- **Scheme A**: X-axis = *robot steps*.",
            f"- AWSC reports robot steps natively; PLD/DSRL steps × act_horizon={act_horizon}.",
            "- Fairest comparison in terms of physical environment interaction.",
        ]
    elif scheme.upper() == "B":
        lines += [
            "- **Scheme B**: X-axis = *chunk decisions*.",
            f"- PLD/DSRL report chunks natively; AWSC steps ÷ act_horizon={act_horizon}.",
        ]
    else:
        lines += [
            "- **Scheme C**: Raw steps (NOT aligned between algorithms).",
            "- Not suitable for formal comparison.",
        ]

    report_path.write_text("\n".join(lines))
    print(f"  Report: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed fair comparison analysis"
    )
    parser.add_argument("--sweep-dir", required=True, help="Sweep base directory")
    parser.add_argument("--comparison-scheme", default="A", choices=["A", "B", "C"])
    parser.add_argument("--act-horizon", type=int, default=7)
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: sweep-dir/analysis_results)"
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else sweep_dir / "analysis_results"
    )

    print("=" * 60)
    print("  Multi-Seed Fair Comparison Analysis")
    print("=" * 60)
    print(f"  Sweep dir : {sweep_dir}")
    print(f"  Scheme    : {args.comparison_scheme}")
    print(f"  Output    : {output_dir}")
    print()

    # 1. Discover experiments
    extractor = DataExtractor(str(sweep_dir), act_horizon=args.act_horizon)
    results = extractor.find_all_experiments()
    if not results:
        print("No experiments found!")
        return

    print(f"Found {len(results)} experiments:")
    for algo_key in ALGO_DISPLAY:
        algo_exps = [r for r in results if r.algorithm == algo_key]
        n_success = sum(1 for r in algo_exps if r.status == "success")
        seeds = sorted(r.seed for r in algo_exps if r.status == "success")
        print(f"  {ALGO_DISPLAY[algo_key]['label']}: "
              f"{n_success}/{len(algo_exps)} success, seeds={seeds}")
    print()

    # 2. Normalize x-axis
    results = extractor.normalize_steps(results, scheme=args.comparison_scheme)

    # 3. Group and compute stats
    groups = group_by_algorithm(results)
    summary = compute_summary_stats(groups)

    # Print summary to terminal
    print("-" * 60)
    print("Summary (success_once, mean ± std):")
    for algo_key in ALGO_DISPLAY:
        st = summary.get(algo_key, {}).get("success_once", {})
        n = summary.get(algo_key, {}).get("n_seeds", 0)
        if st.get("mean") is not None:
            print(
                f"  {ALGO_DISPLAY[algo_key]['label']:15s}: "
                f"{st['mean']:.4f} ± {st['std']:.4f}  "
                f"(n={n}, range=[{st['min']:.4f}, {st['max']:.4f}])"
            )
    print("-" * 60)
    print()

    # 4. Generate plots
    plot_files: list[str] = []
    print("Generating plots...")
    plot_files += plot_mean_std_curves(groups, output_dir, scheme=args.comparison_scheme)
    plot_files += plot_box_whisker(groups, output_dir)
    plot_files += plot_per_seed_bars(groups, output_dir)
    print()

    # 5. Generate report
    print("Generating report...")
    generate_report(
        groups,
        summary,
        output_dir,
        scheme=args.comparison_scheme,
        act_horizon=args.act_horizon,
        plot_files=plot_files,
    )

    # 6. Export JSON summary
    json_summary = {
        "timestamp": datetime.now().isoformat(),
        "comparison_scheme": args.comparison_scheme,
        "act_horizon": args.act_horizon,
        "num_experiments": len(results),
        "algorithms": {},
    }
    for algo_key in ALGO_DISPLAY:
        st = summary.get(algo_key, {})
        json_summary["algorithms"][algo_key] = st

    json_path = output_dir / "analysis_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_summary, indent=2, default=str))
    print(f"  JSON: {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
