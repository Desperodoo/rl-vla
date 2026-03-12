#!/usr/bin/env python3
"""Fetch wandb training data and generate analysis plots.

Usage:
    # Fetch a single run by wandb run ID
    python scripts/sweep_acp/fetch_wandb.py --run_id wa52z9ce --project rlpd-acp-mirror

    # Fetch all runs from a project
    python scripts/sweep_acp/fetch_wandb.py --project rlpd-acp-mirror

    # Fetch and compare multiple runs
    python scripts/sweep_acp/fetch_wandb.py --run_ids wa52z9ce,dzr0k50k --project rlpd-acp-mirror

    # Filter runs by name pattern
    python scripts/sweep_acp/fetch_wandb.py --project ACP-Sweep --filter "awsc"

    # Specify output directory
    python scripts/sweep_acp/fetch_wandb.py --run_id wa52z9ce --project rlpd-acp-mirror -o logs/vlaw/wandb_analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric groups for structured analysis
# ---------------------------------------------------------------------------
METRIC_GROUPS = {
    "eval": {
        "title": "Evaluation Metrics",
        "keys": [
            "eval/success_once",
            "eval/success_at_end",
            "eval/elapsed_steps",
            "eval/return",
        ],
    },
    "reward": {
        "title": "Reward Signals",
        "keys": [
            "train/reward/sim_step_mean",
            "train/reward/acp_step_mean",
            "train/smdp/online_cum_reward_mean",
            "train/smdp/offline_cum_reward_mean",
        ],
    },
    "critic": {
        "title": "Critic Metrics",
        "keys": [
            "train/critic_loss",
            "train/critic/q_mean",
            "train/critic/q_std",
            "train/critic/td_target_mean",
        ],
    },
    "actor": {
        "title": "Actor Metrics",
        "keys": [
            "train/actor_loss",
            "train/actor/flow_loss",
            "train/actor/shortcut_loss",
            "train/actor/q_mean",
        ],
    },
    "advantage": {
        "title": "Advantage Weighting",
        "keys": [
            "train/actor/advantage_mean",
            "train/actor/advantage_std",
            "train/actor/weight_mean",
            "train/actor/weight_std",
            "train/actor/weight_max",
        ],
    },
    "data": {
        "title": "Data Composition",
        "keys": [
            "train/actor/n_demo_samples",
            "train/actor/n_online_kept",
            "train/actor/n_online_filtered",
            "train/actor/policy_batch_size",
        ],
    },
    "success": {
        "title": "Success Rate Overview",
        "keys": [
            "eval/success_once",
            "eval/success_at_end",
            "train/success_rate",
        ],
    },
}


def fetch_run_history(
    project: str,
    run_id: str,
    entity: Optional[str] = None,
    keys: Optional[list[str]] = None,
) -> tuple[dict, pd.DataFrame]:
    """Fetch full history for a single wandb run."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(path)

    # Fetch history — wandb returns a list of dicts
    if keys:
        samples = 10000  # max
        history = run.scan_history(keys=["_step"] + keys, page_size=1000)
    else:
        history = run.scan_history(page_size=1000)

    rows = list(history)
    df = pd.DataFrame(rows)
    if "_step" in df.columns:
        df = df.sort_values("_step").reset_index(drop=True)

    config = dict(run.config)
    config["_run_name"] = run.name
    config["_run_id"] = run.id
    config["_state"] = run.state

    # Get summary for best metrics
    summary = dict(run.summary)
    config["_summary"] = {
        k: v for k, v in summary.items() if not k.startswith("_")
    }

    return config, df


def fetch_runs(
    project: str,
    run_ids: Optional[list[str]] = None,
    entity: Optional[str] = None,
    name_filter: Optional[str] = None,
) -> dict[str, tuple[dict, pd.DataFrame]]:
    """Fetch multiple runs, returns {run_id: (config, df)}."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    if run_ids:
        results = {}
        for rid in run_ids:
            print(f"  Fetching {rid}...", end=" ", flush=True)
            config, df = fetch_run_history(project, rid, entity)
            results[rid] = (config, df)
            print(f"({len(df)} rows)")
        return results

    # Fetch all runs from project
    runs = api.runs(path)
    results = {}
    for run in runs:
        if name_filter and name_filter not in run.name:
            continue
        print(f"  Fetching {run.id}: {run.name}...", end=" ", flush=True)
        config, df = fetch_run_history(project, run.id, entity)
        results[run.id] = (config, df)
        print(f"({len(df)} rows)")

    return results


def plot_metric_group(
    data: dict[str, tuple[dict, pd.DataFrame]],
    group_name: str,
    output_dir: Path,
    x_key: str = "_step",
) -> Optional[Path]:
    """Plot a metric group for all runs."""
    if group_name not in METRIC_GROUPS:
        print(f"  WARNING: Unknown group '{group_name}'")
        return None

    group = METRIC_GROUPS[group_name]
    keys = group["keys"]

    # Filter to keys that actually exist
    available_keys = []
    for k in keys:
        for _, (_, df) in data.items():
            if k in df.columns:
                available_keys.append(k)
                break

    if not available_keys:
        print(f"  No data for group '{group_name}'")
        return None

    n_keys = len(available_keys)
    n_cols = min(2, n_keys)
    n_rows = (n_keys + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    if n_keys == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, key in enumerate(available_keys):
        ax = axes[idx // n_cols, idx % n_cols]
        for run_id, (config, df) in data.items():
            if key not in df.columns:
                continue
            mask = df[key].notna()
            if mask.sum() == 0:
                continue
            label = config.get("_run_name", run_id)
            # Shorten label
            if len(label) > 30:
                label = label[:27] + "..."
            ax.plot(
                df.loc[mask, x_key],
                df.loc[mask, key],
                label=label,
                alpha=0.8,
                linewidth=1.2,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel(key.split("/")[-1])
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_keys, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(group["title"], fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = output_dir / f"{group_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_text_report(
    data: dict[str, tuple[dict, pd.DataFrame]],
    output_dir: Path,
) -> Path:
    """Generate a markdown text report with key statistics."""
    lines = ["# WandB Training Analysis Report\n"]

    for run_id, (config, df) in data.items():
        run_name = config.get("_run_name", run_id)
        lines.append(f"\n## Run: {run_name} (`{run_id}`)\n")

        # Config summary
        lines.append("### Key Config\n")
        important_keys = [
            "acp_reward_scale", "reward_mode",
            "awsc_beta", "awsc_bc_weight",
            "lr_actor", "lr_critic",
            "total_timesteps", "online_ratio", "utd_ratio",
            "num_qs", "num_min_qs", "gamma",
        ]
        for k in important_keys:
            if k in config:
                lines.append(f"- `{k}`: {config[k]}")

        # Eval metrics summary
        lines.append("\n### Evaluation Summary\n")
        lines.append("| Metric | Best | Final | Step@Best |")
        lines.append("|--------|------|-------|-----------|")

        for metric in ["eval/success_once", "eval/success_at_end", "eval/return"]:
            if metric not in df.columns:
                continue
            col = df[metric].dropna()
            if len(col) == 0:
                continue
            best_val = col.max()
            final_val = col.iloc[-1]
            best_idx = col.idxmax()
            best_step = df.loc[best_idx, "_step"] if "_step" in df.columns else "?"
            name = metric.split("/")[-1]
            lines.append(f"| {name} | {best_val:.4f} | {final_val:.4f} | {best_step} |")

        # Reward dynamics
        lines.append("\n### Reward Dynamics\n")
        for metric in ["train/reward/acp_step_mean", "train/reward/sim_step_mean"]:
            if metric not in df.columns:
                continue
            col = df[metric].dropna()
            if len(col) == 0:
                continue
            name = metric.split("/")[-1]
            lines.append(
                f"- **{name}**: mean={col.mean():.4f}, std={col.std():.4f}, "
                f"min={col.min():.4f}, max={col.max():.4f}"
            )

        # Critic health
        lines.append("\n### Critic Health\n")
        for metric in ["train/critic/q_mean", "train/critic/td_target_mean", "train/critic_loss"]:
            if metric not in df.columns:
                continue
            col = df[metric].dropna()
            if len(col) == 0:
                continue
            name = metric.split("/")[-1]
            # Check last 20% for trend
            n = len(col)
            last_20 = col.iloc[int(n * 0.8):]
            first_20 = col.iloc[:int(n * 0.2)]
            trend = "stable"
            if len(first_20) > 0 and len(last_20) > 0:
                ratio = last_20.mean() / (first_20.mean() + 1e-8)
                if ratio > 2:
                    trend = "INCREASING"
                elif ratio < 0.5:
                    trend = "DECREASING"
            lines.append(
                f"- **{name}**: mean={col.mean():.4f}, final_20%_mean={last_20.mean():.4f}, "
                f"trend={trend}"
            )

        # Advantage weighting analysis
        lines.append("\n### Advantage Weighting\n")
        for metric in [
            "train/actor/advantage_mean",
            "train/actor/weight_mean",
            "train/actor/weight_max",
        ]:
            if metric not in df.columns:
                continue
            col = df[metric].dropna()
            if len(col) == 0:
                continue
            name = metric.split("/")[-1]
            lines.append(
                f"- **{name}**: mean={col.mean():.4f}, std={col.std():.4f}, "
                f"min={col.min():.4f}, max={col.max():.4f}"
            )

        # Data composition
        lines.append("\n### Data Composition (Actor)\n")
        for metric in [
            "train/actor/n_demo_samples",
            "train/actor/n_online_kept",
            "train/actor/n_online_filtered",
        ]:
            if metric not in df.columns:
                continue
            col = df[metric].dropna()
            if len(col) == 0:
                continue
            name = metric.split("/")[-1]
            lines.append(
                f"- **{name}**: mean={col.mean():.1f}, min={col.min():.0f}, max={col.max():.0f}"
            )

        # Degradation analysis: success_once vs success_at_end gap
        if "eval/success_once" in df.columns and "eval/success_at_end" in df.columns:
            lines.append("\n### Success Gap Analysis (success_once - success_at_end)\n")
            s_once = df["eval/success_once"].dropna()
            s_end = df["eval/success_at_end"].dropna()
            if len(s_once) > 0 and len(s_end) > 0:
                # Align by step
                merged = df[["_step", "eval/success_once", "eval/success_at_end"]].dropna()
                merged["gap"] = merged["eval/success_once"] - merged["eval/success_at_end"]
                lines.append(f"- Mean gap: {merged['gap'].mean():.4f}")
                lines.append(f"- Max gap: {merged['gap'].max():.4f}")
                lines.append(f"- Final gap: {merged['gap'].iloc[-1]:.4f}")
                if merged["gap"].mean() > 0.2:
                    lines.append("- **WARNING**: Large success_once vs success_at_end gap suggests "
                                 "policy picks up peg but drops it before episode end")

    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and analyze wandb training data")
    parser.add_argument("--project", "-p", required=True, help="WandB project name")
    parser.add_argument("--run_id", help="Single run ID to fetch")
    parser.add_argument("--run_ids", help="Comma-separated run IDs")
    parser.add_argument("--entity", "-e", help="WandB entity/team")
    parser.add_argument("--filter", "-f", help="Filter runs by name substring")
    parser.add_argument("--output_dir", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=list(METRIC_GROUPS.keys()),
        help=f"Metric groups to plot. Available: {list(METRIC_GROUPS.keys())}",
    )
    parser.add_argument("--save_csv", action="store_true", help="Also save raw data as CSV")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Determine run IDs
    run_ids = None
    if args.run_id:
        run_ids = [args.run_id]
    elif args.run_ids:
        run_ids = [r.strip() for r in args.run_ids.split(",")]

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif run_ids and len(run_ids) == 1:
        output_dir = Path(f"logs/vlaw/wandb_analysis/{run_ids[0]}")
    else:
        output_dir = Path(f"logs/vlaw/wandb_analysis/{args.project}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching from project: {args.project}")
    data = fetch_runs(
        project=args.project,
        run_ids=run_ids,
        entity=args.entity,
        name_filter=args.filter,
    )

    if not data:
        print("No runs found!")
        sys.exit(1)

    print(f"\nFetched {len(data)} run(s)")

    # Save CSV
    if args.save_csv:
        for run_id, (config, df) in data.items():
            csv_path = output_dir / f"{run_id}_history.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

            cfg_path = output_dir / f"{run_id}_config.json"
            cfg_path.write_text(json.dumps(config, indent=2, default=str))

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        for group in args.groups:
            out = plot_metric_group(data, group, output_dir)
            if out:
                print(f"  Saved: {out}")

    # Generate text report
    print("\nGenerating analysis report...")
    report_path = generate_text_report(data, output_dir)
    print(f"  Report: {report_path}")

    # Print report to stdout
    print("\n" + "=" * 70)
    print(report_path.read_text())


if __name__ == "__main__":
    main()
