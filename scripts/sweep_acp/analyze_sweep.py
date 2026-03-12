#!/usr/bin/env python3
"""
ACP Sweep Experiment Analysis Tool
====================================

Analyzes ACP reward sweep experiments across AWSC, PLD-SAC, and DSRL-SAC.
Focus on success_at_end as the core metric (ACP's key weakness).

Features:
1. Data extraction from training logs and WandB summaries
2. Parameter sensitivity analysis
3. Training curve visualization
4. Comparison against sim baselines
5. Markdown report generation

Usage:
    python analyze_sweep.py --sweep-dir runs/acp_sweep
    python analyze_sweep.py --sweep-dir runs/acp_sweep --algorithm awsc_acp
    python analyze_sweep.py --sweep-dir runs/acp_sweep --output-dir analysis_results
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# SIM BASELINES (from fair_comparison, seed 42)
SIM_BASELINES = {
    "awsc_acp": {"best_success_at_end": 0.72, "best_success_once": 0.92},
    "pld_acp":  {"best_success_at_end": 0.86, "best_success_once": 1.00},
    "dsrl_acp": {"best_success_at_end": 0.60, "best_success_once": 0.98},
}

# ACP MIRROR BASELINES (from acp_mirror_experiments, seed 42)
ACP_MIRROR_BASELINES = {
    "awsc_acp": {"best_success_at_end": 0.66, "best_success_once": 0.90},
    "pld_acp":  {"best_success_at_end": 0.02, "best_success_once": 0.82},
    "dsrl_acp": {"best_success_at_end": 0.06, "best_success_once": 0.92},
}


@dataclass
class ExperimentResult:
    algorithm: str
    config_name: str
    status: str  # success, failed, not_started
    best_success_once: Optional[float] = None
    final_success_once: Optional[float] = None
    best_success_at_end: Optional[float] = None
    final_success_at_end: Optional[float] = None
    params: Dict[str, str] = field(default_factory=dict)
    training_curve_once: List[Tuple[int, float]] = field(default_factory=list)
    training_curve_end: List[Tuple[int, float]] = field(default_factory=list)


def parse_log_file(log_path: Path) -> dict:
    """Extract metrics from a training log file."""
    metrics = {}
    if not log_path.exists():
        return metrics

    text = log_path.read_text(errors='ignore')

    # Parse wandb summary metrics
    for line in text.split('\n'):
        if 'wandb:' not in line:
            continue

        # final/best_success_rate
        m = re.search(r'wandb:\s+final/best_success_rate\s+([\d.]+)', line)
        if m:
            metrics['best_success_once'] = float(m.group(1))

        # eval/success_at_end
        m = re.search(r'wandb:\s+eval/success_at_end\s+([\d.]+)', line)
        if m:
            metrics['final_success_at_end'] = float(m.group(1))

        # eval/success_once
        m = re.search(r'wandb:\s+eval/success_once\s+([\d.]+)', line)
        if m:
            metrics['final_success_once'] = float(m.group(1))

    # Parse "Done. Best success rate: XX.XX%"
    m = re.search(r'Done\. Best success rate: ([\d.]+)%', text)
    if m:
        metrics.setdefault('best_success_once', float(m.group(1)) / 100.0)

    # Parse best success_at_end from all eval lines
    ends = re.findall(r'success_at_end[:\s]+([\d.]+)', text)
    if ends:
        metrics['best_success_at_end'] = max(float(v) for v in ends)
        metrics.setdefault('final_success_at_end', float(ends[-1]))

    # Parse best success_once from all eval lines
    onces = re.findall(r'success_once[:\s]+([\d.]+)', text)
    if onces:
        metrics.setdefault('best_success_once', max(float(v) for v in onces))
        metrics.setdefault('final_success_once', float(onces[-1]))

    return metrics


def parse_training_curves(log_path: Path) -> Tuple[list, list]:
    """Extract step-by-step training curves from log."""
    curve_once = []
    curve_end = []

    if not log_path.exists():
        return curve_once, curve_end

    text = log_path.read_text(errors='ignore')

    # Try to parse eval lines with step numbers
    # Pattern: "Step XXXX | eval/success_once: 0.XX | eval/success_at_end: 0.XX"
    for line in text.split('\n'):
        step_match = re.search(r'(?:step|Step|global_step)[=:\s]+(\d+)', line)
        if not step_match:
            continue
        step = int(step_match.group(1))

        once_match = re.search(r'success_once[=:\s]+([\d.]+)', line)
        end_match = re.search(r'success_at_end[=:\s]+([\d.]+)', line)

        if once_match:
            curve_once.append((step, float(once_match.group(1))))
        if end_match:
            curve_end.append((step, float(end_match.group(1))))

    return curve_once, curve_end


def load_experiments(sweep_dir: Path, algorithm: Optional[str] = None) -> List[ExperimentResult]:
    """Load all experiment results from a sweep directory."""
    results = []

    algos = [algorithm] if algorithm else ["awsc_acp", "pld_acp", "dsrl_acp"]

    for algo in algos:
        algo_dir = sweep_dir / algo
        if not algo_dir.exists():
            continue

        for exp_dir in sorted(algo_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            config_name = exp_dir.name
            # Strip timestamp suffix
            if '__' in config_name:
                config_name = config_name.rsplit('__', 1)[0]

            # Find log file
            log_file = exp_dir / "train.log"
            if not log_file.exists():
                # Check parent (base dir without timestamp)
                log_file = algo_dir / config_name / "train.log"

            metrics = parse_log_file(log_file)
            curve_once, curve_end = parse_training_curves(log_file)

            # Determine status
            ckpt_dir = exp_dir / "checkpoints"
            if (ckpt_dir / "best.pt").exists() or \
               (ckpt_dir / "best_eval_success_once.pt").exists() or \
               (ckpt_dir / "final.pt").exists():
                status = "success"
            elif ckpt_dir.exists():
                status = "failed"
            else:
                status = "not_started"

            result = ExperimentResult(
                algorithm=algo,
                config_name=config_name,
                status=status,
                best_success_once=metrics.get('best_success_once'),
                final_success_once=metrics.get('final_success_once'),
                best_success_at_end=metrics.get('best_success_at_end'),
                final_success_at_end=metrics.get('final_success_at_end'),
                training_curve_once=curve_once,
                training_curve_end=curve_end,
            )
            results.append(result)

    return results


def generate_bar_chart(results: List[ExperimentResult], algo: str, output_dir: Path):
    """Generate bar chart comparing configs for a single algorithm."""
    if not HAS_MATPLOTLIB:
        return

    completed = [r for r in results if r.algorithm == algo and r.status == "success"
                 and r.best_success_at_end is not None]
    if not completed:
        return

    completed.sort(key=lambda r: r.best_success_at_end or 0, reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(completed) * 0.4)))

    # success_at_end bars
    ax1 = axes[0]
    names = [r.config_name for r in completed]
    ends = [r.best_success_at_end or 0 for r in completed]
    colors = ['#2196F3' if e > 0.1 else '#90CAF9' for e in ends]

    bars1 = ax1.barh(range(len(names)), ends, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Best success_at_end')
    ax1.set_title(f'{algo}: success_at_end (core metric)')
    ax1.invert_yaxis()

    # Add sim baseline line
    algo_key = algo
    if algo_key in SIM_BASELINES:
        ax1.axvline(x=SIM_BASELINES[algo_key]['best_success_at_end'],
                    color='red', linestyle='--', linewidth=2, label='Sim baseline')
    if algo_key in ACP_MIRROR_BASELINES:
        ax1.axvline(x=ACP_MIRROR_BASELINES[algo_key]['best_success_at_end'],
                    color='orange', linestyle=':', linewidth=2, label='ACP mirror')
    ax1.legend(fontsize=8)

    # success_once bars
    ax2 = axes[1]
    onces = [r.best_success_once or 0 for r in completed]
    colors2 = ['#4CAF50' if o > 0.5 else '#A5D6A7' for o in onces]

    bars2 = ax2.barh(range(len(names)), onces, color=colors2)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Best success_once')
    ax2.set_title(f'{algo}: success_once')
    ax2.invert_yaxis()

    if algo_key in SIM_BASELINES:
        ax2.axvline(x=SIM_BASELINES[algo_key]['best_success_once'],
                    color='red', linestyle='--', linewidth=2, label='Sim baseline')
    if algo_key in ACP_MIRROR_BASELINES:
        ax2.axvline(x=ACP_MIRROR_BASELINES[algo_key]['best_success_once'],
                    color='orange', linestyle=':', linewidth=2, label='ACP mirror')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{algo}_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{algo}_results.png'}")


def generate_training_curves(results: List[ExperimentResult], algo: str, output_dir: Path):
    """Generate training curves for an algorithm."""
    if not HAS_MATPLOTLIB:
        return

    completed = [r for r in results if r.algorithm == algo and r.status == "success"
                 and len(r.training_curve_end) > 1]
    if not completed:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # success_at_end curves
    ax1 = axes[0]
    for r in completed[:10]:  # top 10
        if r.training_curve_end:
            steps, vals = zip(*r.training_curve_end)
            ax1.plot(steps, vals, label=r.config_name, alpha=0.7)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('success_at_end')
    ax1.set_title(f'{algo}: success_at_end curves')
    ax1.legend(fontsize=6, loc='best')
    ax1.grid(True, alpha=0.3)

    # success_once curves
    ax2 = axes[1]
    for r in completed[:10]:
        if r.training_curve_once:
            steps, vals = zip(*r.training_curve_once)
            ax2.plot(steps, vals, label=r.config_name, alpha=0.7)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('success_once')
    ax2.set_title(f'{algo}: success_once curves')
    ax2.legend(fontsize=6, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{algo}_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{algo}_curves.png'}")


def generate_report(results: List[ExperimentResult], output_dir: Path):
    """Generate a markdown summary report."""
    report = []
    report.append("# ACP Sweep Analysis Report")
    report.append(f"\n> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for algo in ["awsc_acp", "pld_acp", "dsrl_acp"]:
        algo_results = [r for r in results if r.algorithm == algo]
        if not algo_results:
            continue

        completed = [r for r in algo_results if r.status == "success"]
        failed = [r for r in algo_results if r.status == "failed"]
        pending = [r for r in algo_results if r.status == "not_started"]

        report.append(f"\n## {algo.upper()}")
        report.append(f"\nTotal: {len(algo_results)} | "
                      f"Completed: {len(completed)} | "
                      f"Failed: {len(failed)} | "
                      f"Not Started: {len(pending)}")

        # Sim baselines
        sim = SIM_BASELINES.get(algo, {})
        mirror = ACP_MIRROR_BASELINES.get(algo, {})
        report.append(f"\n**Sim baseline**: best_s_end={sim.get('best_success_at_end', '?')}, "
                      f"best_s_once={sim.get('best_success_once', '?')}")
        report.append(f"**ACP mirror**: best_s_end={mirror.get('best_success_at_end', '?')}, "
                      f"best_s_once={mirror.get('best_success_once', '?')}")

        if completed:
            # Sort by best_success_at_end
            completed.sort(key=lambda r: r.best_success_at_end or 0, reverse=True)

            report.append("\n### Results (sorted by best success_at_end)")
            report.append("")
            report.append("| Config | best_s_end | final_s_end | best_s_once | final_s_once |")
            report.append("|--------|-----------|------------|------------|-------------|")

            for r in completed:
                b_end = f"{r.best_success_at_end:.2%}" if r.best_success_at_end else "-"
                f_end = f"{r.final_success_at_end:.2%}" if r.final_success_at_end else "-"
                b_once = f"{r.best_success_once:.2%}" if r.best_success_once else "-"
                f_once = f"{r.final_success_once:.2%}" if r.final_success_once else "-"
                report.append(f"| {r.config_name} | {b_end} | {f_end} | {b_once} | {f_once} |")

            # Best config
            best = completed[0]
            report.append(f"\n**Best config**: `{best.config_name}` — "
                          f"best_s_end={best.best_success_at_end or 0:.2%}, "
                          f"best_s_once={best.best_success_once or 0:.2%}")

            # Delta vs sim
            sim_end = sim.get('best_success_at_end', 0)
            if best.best_success_at_end and sim_end:
                delta = best.best_success_at_end - sim_end
                emoji = ">" if delta > 0 else "<"
                report.append(f"Delta vs sim: {delta:+.2%} ({emoji} sim baseline)")

        if failed:
            report.append("\n### Failed Configs")
            for r in failed:
                report.append(f"- {r.config_name}")

    # Summary
    report.append("\n---\n## Summary\n")

    all_completed = [r for r in results if r.status == "success" and r.best_success_at_end]
    if all_completed:
        all_completed.sort(key=lambda r: r.best_success_at_end or 0, reverse=True)
        best = all_completed[0]
        report.append(f"**Overall best**: `{best.algorithm}/{best.config_name}` — "
                      f"best_s_end={best.best_success_at_end:.2%}")
    else:
        report.append("No completed experiments yet.")

    report_text = '\n'.join(report)
    report_path = output_dir / "acp_sweep_report.md"
    report_path.write_text(report_text)
    print(f"  Report saved: {report_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="ACP Sweep Analysis")
    parser.add_argument('--sweep-dir', required=True, help='Sweep base directory')
    parser.add_argument('--algorithm', default=None, help='Analyze single algorithm')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots/reports')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / 'analysis_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading experiments from {sweep_dir}...")
    results = load_experiments(sweep_dir, args.algorithm)
    print(f"Found {len(results)} experiments")

    if not results:
        print("No experiments found.")
        return

    # Generate plots
    algos = set(r.algorithm for r in results)
    for algo in sorted(algos):
        print(f"\nAnalyzing {algo}...")
        generate_bar_chart(results, algo, output_dir)
        generate_training_curves(results, algo, output_dir)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, output_dir)
    print("\n" + report)


if __name__ == "__main__":
    main()
